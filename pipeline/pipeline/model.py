"""Chat completions against an OpenAI-compatible endpoint, such as a vLLM server.

Every stage that calls a model goes through this client. Point it at your own endpoint
with --base-url, or with the VIDEO_HOPCHAIN_BASE_URL environment variable.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("VIDEO_HOPCHAIN_BASE_URL", "http://localhost:8000/v1")

_FENCE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.S)


def extract_json(text: str):
    """Parse the JSON object or array out of a model response."""
    if not text or not text.strip():
        raise ValueError("empty model response")
    body = text.strip()
    fence = _FENCE.search(body)
    if fence:
        body = fence.group(1).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = body.find(opener), body.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(body[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON in model response")


def video_part(path: str, *, fps: float, max_pixels: int) -> dict:
    """A video content part, as vLLM's OpenAI-compatible server accepts it."""
    url = path if "://" in path else "file://" + os.path.abspath(path)
    return {"type": "video_url", "video_url": {"url": url},
            "fps": fps, "max_pixels": max_pixels}


class Chat:
    """One model behind an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL,
                 api_key: str = "EMPTY", timeout: int = 1800):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, messages: list, *, max_tokens: int = 4096,
                 temperature: float = 0.0, n: int = 1) -> list:
        """Return n completions for a message list."""
        body = json.dumps({"model": self.model, "messages": messages, "n": n,
                           "max_tokens": max_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
        return [c["message"]["content"] for c in payload["choices"]]

    def text(self, system: str, user: str, *, max_tokens: int = 4096,
             temperature: float = 0.0) -> str:
        """One text-only completion."""
        return self.complete([{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                             max_tokens=max_tokens, temperature=temperature)[0]

    def json_text(self, system: str, user: str, *, max_tokens: int = 4096,
                  temperature: float = 0.0):
        """One text-only completion, parsed as JSON."""
        return extract_json(self.text(system, user, max_tokens=max_tokens,
                                      temperature=temperature))

    def on_video(self, system: str, user: str, video: str, *, fps: float = 1.0,
                 max_pixels: int = 50176, max_tokens: int = 2048,
                 temperature: float = 0.0, n: int = 1) -> list:
        """n completions for a prompt that carries one video."""
        content = [video_part(video, fps=fps, max_pixels=max_pixels),
                   {"type": "text", "text": user}]
        return self.complete([{"role": "system", "content": system},
                              {"role": "user", "content": content}],
                             max_tokens=max_tokens, temperature=temperature, n=n)


__all__ = ["Chat", "DEFAULT_BASE_URL", "extract_json", "video_part"]
