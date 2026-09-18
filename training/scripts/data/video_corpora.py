#!/usr/bin/env python3

from collections import OrderedDict
import hashlib
import os
import re

SCRATCH = "__SCRATCH__"
PROJECT_DATA = "__PROJECT__/data"
VRIPT = "__SCRATCH__/tmp/vript"

PROXY_BASE = f"{SCRATCH}/videoreason"
PROXY_FRAMES = 140
PROXY_MAX_PIXELS = 50176

CORPORA = OrderedDict([
    ("vript", f"{VRIPT}/videos"),
    ("video_res", f"{SCRATCH}/videoreason/video_res_1080p/videos_1080p"),
    ("longvila", f"{SCRATCH}/rl/data/onethinker/QA/longvila_videos"),
    ("finevideo", f"{PROJECT_DATA}/finevideo/videos"),
    ("onethinker_queryd", f"{PROJECT_DATA}/onethinker_extra/queryd/videos"),
    ("onethinker_activitynet", f"{PROJECT_DATA}/onethinker_extra/activitynet/videos"),
    ("onethinker_hirest", f"{PROJECT_DATA}/onethinker_extra/hirest/videos"),
])

PROXY_ROOTS = OrderedDict([
    ("vript", f"{PROXY_BASE}/proxy_vript_140f50k"),
    ("video_res", f"{PROXY_BASE}/proxy_140f50k"),
    ("longvila", f"{PROXY_BASE}/proxy_longvila_140f50k"),
    ("finevideo", f"{PROXY_BASE}/proxy_finevideo_140f50k"),
    ("onethinker_queryd", f"{PROXY_BASE}/proxy_onethinker_queryd_140f50k"),
    ("onethinker_activitynet", f"{PROXY_BASE}/proxy_onethinker_activitynet_140f50k"),
    ("onethinker_hirest", f"{PROXY_BASE}/proxy_onethinker_hirest_140f50k"),
])

assert set(PROXY_ROOTS) == set(CORPORA), (
    "PROXY_ROOTS and CORPORA disagree: "
    f"{sorted(set(CORPORA) ^ set(PROXY_ROOTS))}"
)

LEGACY_SOURCE_ROOTS = [f"{SCRATCH}/videoreason/video_res/videos"]

PROXY_ROOT_LIST = list(PROXY_ROOTS.values())
SOURCE_ROOT_LIST = list(CORPORA.values()) + LEGACY_SOURCE_ROOTS

VIDEO_ROOTS = PROXY_ROOT_LIST + SOURCE_ROOT_LIST

ROOT_LABELS = {}
for _c, _d in PROXY_ROOTS.items():
    ROOT_LABELS[os.path.abspath(_d)] = ("proxy", _c)
for _c, _d in CORPORA.items():
    ROOT_LABELS.setdefault(os.path.abspath(_d), ("source", _c))
for _d in LEGACY_SOURCE_ROOTS:
    ROOT_LABELS.setdefault(os.path.abspath(_d), ("source", "video_res_legacy"))


def label_root(root):
    """-> (kind, corpus) for a known root, else ('unknown', '')."""
    return ROOT_LABELS.get(os.path.abspath(root), ("unknown", ""))


class OutputNameCollision(RuntimeError):
    """Two distinct media roots claimed ONE output path. Always fatal, never overwrite."""


def root_slug(root):
    kind, corpus = label_root(root)
    if kind != "unknown":
        slug = f"{kind}_{corpus}"
    else:
        aroot = os.path.abspath(root)
        base = os.path.basename(aroot) or "root"
        slug = f"{base}_{hashlib.sha1(aroot.encode()).hexdigest()[:8]}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", slug)


def plan_root_outputs(out_dir, roots, fname_fn):
    roots = sorted(roots)
    n = len(roots)
    entries, claimed = [], {}
    for root in roots:
        path = os.path.abspath(os.path.join(out_dir, fname_fn(root_slug(root), n)))
        if path in claimed:
            raise OutputNameCollision(
                f"two media roots map to one output file {path}:\n"
                f"    {claimed[path]}\n    {root}\n"
                "Writing both would destroy one corpus and emit a manifest with two "
                "entries pointing at one parquet. Add the root to video_corpora.py or "
                "fix root_slug()."
            )
        claimed[path] = root
        entries.append((path, root))
    return entries


def proxy_dir(corpus):
    return PROXY_ROOTS[corpus]


def missing_proxy_dirs():
    """Corpora whose proxy dir does not exist yet -> [(corpus, dir), ...]."""
    return [(c, d) for c, d in PROXY_ROOTS.items() if not os.path.isdir(d)]


if __name__ == "__main__":
    print(f"{'corpus':<24} {'source':>9} {'proxy':>9}  paths")
    for c in CORPORA:
        s, p = CORPORA[c], PROXY_ROOTS[c]
        ns = sum(1 for e in os.scandir(s) if e.name.endswith(".mp4")) if os.path.isdir(s) else None
        np_ = sum(1 for e in os.scandir(p) if e.name.endswith(".mp4")) if os.path.isdir(p) else None
        print(f"{c:<24} {('ABSENT' if ns is None else f'{ns:,}'):>9} "
              f"{('ABSENT' if np_ is None else f'{np_:,}'):>9}  {s}")
