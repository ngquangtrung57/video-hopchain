# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000

# Token budget for the two-wave common-prefix (divergence) measurement. The cost
# is O(n * cap) pure-Python comparisons per prompt inside the async dispatcher,
# so the cap bounds the event-loop stall a runaway-length rollout could cause.
_TW_LCP_CAP = 4096


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        processing_class = self.processor if self.processor is not None else self.tokenizer
        self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # <think> prefill (opt-in: VERL_THINK_PREFILL=1) — force the assistant turn to open a
            # think block so the graded format reward has variance to bootstrap from (base
            # Qwen3-VL-Instruct never emits <think> on its own -> uniform format=0 -> no gradient).
            # Guarded so a template that already emits <think> is not double-tagged. The reward side
            # (rewards/boxed_reward_wrapper.py) prepends the SAME string so the scored text
            # reconstructs faithfully. Processor (VL) path only; the tokenizer-only branch below is
            # never taken for Qwen3-VL.
            if os.environ.get("VERL_THINK_PREFILL", "0") == "1":
                _pf = os.environ.get("VERL_THINK_PREFILL_STR", "<think>")
                if not raw_prompt.rstrip().endswith(_pf):
                    raw_prompt = raw_prompt + _pf

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        # Exploration mode: the first k_explore of n_resp_per_prompt rollouts per prompt
        # carry a marker consumed by the EntropyDropoutLP logits processor. Auto-disabled
        # during validation.
        exploration_enabled = (
            not validate
            and getattr(config, "exploration", None) is not None
            and config.exploration.enable
            and config.exploration.k_explore > 0
        )
        # Per-prompt Bernoulli gate. Each unique sample_index in the batch is
        # independently selected with probability prompt_exploration_prob; at
        # 1.0 (the default) every prompt is eligible.
        prompt_explore_map: dict[int, bool] = {}
        if exploration_enabled:
            p_prompt = float(getattr(config.exploration, "prompt_exploration_prob", 1.0))
            if p_prompt >= 1.0:
                for ti in trajectory_info:
                    prompt_explore_map[int(ti["sample_index"])] = True
            else:
                # In async mode with gen_batch_size=1 each call handles one
                # prompt, so the set of unique "sample_index" values is always
                # {0} (the within-batch index, not the dataset row id) and a
                # seeded RNG keyed on it would collapse to a constant. A fresh
                # unseeded RNG per call gives each prompt an independent draw.
                unique_sids = {int(ti["sample_index"]) for ti in trajectory_info}
                if len(unique_sids) <= 1:
                    # gen_batch_size=1 fast path: single draw governs the whole batch.
                    selected = bool(np.random.random() < p_prompt)
                    for sid in unique_sids:
                        prompt_explore_map[sid] = selected
                else:
                    # gen_batch_size>1: per-prompt independent draws.
                    rng = np.random.default_rng()
                    for sid in unique_sids:
                        prompt_explore_map[sid] = bool(rng.random() < p_prompt)
        # Two-wave variance-targeted exploration: when enabled, route to a
        # per-prompt two-wave dispatcher that rolls wave-1 anchors normally,
        # measures wave-1 reward variance, then conditionally explores wave-2.
        # Off by default, in which case the else-branch below runs. Never fires
        # on validation or n < 2.
        two_wave = (
            exploration_enabled
            and getattr(config.exploration, "two_wave_enable", False)
            and not validate
            and len(batch) >= 2
        )
        # Rollout-group affinity key. All n rollouts of one prompt share a
        # byte-identical multimodal prefix, which the engine's prefix and
        # encoder caches can reuse only if the requests land on the same
        # replica; this key routes them there. On the fully-async path one
        # batch is one prompt's n rollouts, so rvrl_prompt_ordinal (stamped in
        # FullyAsyncRollouter._feed_samples) identifies the group. sample_index
        # cannot be used, because the curated parquets never write
        # extra_info.index and it is always 0. None => each loop keeps its own
        # uuid4.
        _ord = batch.meta_info.get("rvrl_prompt_ordinal")
        rvrl_group_key = None if _ord is None or int(_ord) < 0 else f"grp-{int(_ord)}"

        if two_wave:
            outputs = await self._dispatch_two_wave(
                batch,
                sampling_params,
                trajectory_info,
                traced_indices,
                per_sample_do_sample,
                config,
                validate,
                rvrl_group_key=rvrl_group_key,
            )
        else:
            tasks = []
            for i in range(len(batch)):
                trace_this_sample = i in traced_indices
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
                if rvrl_group_key is not None:
                    kwargs["rvrl_group_key"] = rvrl_group_key
                sample_sampling_params = dict(sampling_params)
                if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                    apply_greedy_sampling_params(sample_sampling_params)
                if (
                    exploration_enabled
                    and prompt_explore_map.get(int(trajectory_info[i]["sample_index"]), False)
                    and trajectory_info[i]["rollout_n"] < config.exploration.k_explore
                ):
                    existing = sample_sampling_params.get("extra_args") or {}
                    # Stamp a per-request UUID so the LP can bind dropped-position
                    # records to a stable key (independent of vLLM slot reshuffles).
                    rvrl_uuid = str(uuid4())
                    sample_sampling_params["extra_args"] = {
                        **existing,
                        "rvrl_exploration": True,
                        "rvrl_req_uuid": rvrl_uuid,
                    }
                tasks.append(
                    asyncio.create_task(
                        self._run_agent_loop(
                            sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs
                        )
                    )
                )
            outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _dispatch_two_wave(
        self,
        batch: DataProto,
        sampling_params: dict[str, Any],
        trajectory_info: list[dict[str, Any]],
        traced_indices: set,
        per_sample_do_sample,
        config,
        validate: bool,
        rvrl_group_key=None,
    ) -> list:
        """Two-wave variance-targeted exploration for one prompt's n rollouts.

        Wave 1 is ordinals [0, n_anchor): normal sampling, never explored. After
        awaiting wave 1 the reward is already populated inline
        (``_run_agent_loop`` -> ``_compute_score``), so wave-1 reward variance is
        measurable here. Wave 2 is ordinals [n_anchor, n): exploration is stamped
        on all of wave-2 iff the wave-1 group is degenerate (low variance).
        Order is preserved, since gather preserves order and
        outputs = wave1 + wave2 is contiguous in rollout_n, so the uid,
        drop-mask and loss-mask flows downstream are unchanged.
        """
        n = len(batch)
        anchor_fraction = float(getattr(config.exploration, "anchor_fraction", 0.5))
        # clamp so each wave is always non-empty (n=8, 0.5 => 4; never 0, never n).
        n_anchor = max(1, min(n - 1, int(round(n * anchor_fraction))))

        def make_task(i: int, allow_explore: bool):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            if rvrl_group_key is not None:
                kwargs["rvrl_group_key"] = rvrl_group_key
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                # Replicate apply_greedy_sampling_params (nested in generate_sequences).
                sample_sampling_params["top_p"] = 1.0
                sample_sampling_params["top_k"] = -1
                sample_sampling_params["temperature"] = 0
            if allow_explore:
                existing = sample_sampling_params.get("extra_args") or {}
                # Wave-2 owns the explore decision, so this is not combined with
                # the single-wave rollout_n < k_explore rule.
                rvrl_uuid = str(uuid4())
                sample_sampling_params["extra_args"] = {
                    **existing,
                    "rvrl_exploration": True,
                    "rvrl_req_uuid": rvrl_uuid,
                }
            elif getattr(config.exploration, "measure_entropy", False):
                # Mark the unperturbed rollouts for entropy measurement only.
                # `rvrl_measure` is a different key from `rvrl_exploration`
                # because the logits processor keys its drop decision solely on
                # the latter, so a measured anchor can never be perturbed.
                existing = sample_sampling_params.get("extra_args") or {}
                sample_sampling_params["extra_args"] = {
                    **existing,
                    "rvrl_measure": True,
                    "rvrl_req_uuid": str(uuid4()),
                }
            return asyncio.create_task(
                self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
            )

        # The toggle gates the STRATEGY, not the exploration. When the strategy
        # is off this prompt takes a plain-GRPO path; when it is on, the explore
        # decision below is still owned by wave-1 variance.
        prompt_ordinal = -1
        strategy_on = True
        if getattr(config.exploration, "two_wave_toggle_enable", False):
            _ord = batch.meta_info.get("rvrl_prompt_ordinal", -1)
            prompt_ordinal = -1 if _ord is None else int(_ord)
            if prompt_ordinal < 0:
                # Fail loud rather than silently running every prompt through
                # the strategy, which is what an unstamped ordinal would do.
                raise RuntimeError(
                    "exploration.two_wave_toggle_enable=true but "
                    "meta_info['rvrl_prompt_ordinal'] is missing or negative "
                    f"(got {_ord!r}). FullyAsyncRollouter._feed_samples must stamp it."
                )
            strategy_on = self._two_wave_strategy_on(prompt_ordinal, config)

        if strategy_on:
            wave1 = await asyncio.gather(*[make_task(i, False) for i in range(n_anchor)])
            explore, wave1_var = self._wave1_low_variance(wave1, config)
            wave2 = await asyncio.gather(*[make_task(i, explore) for i in range(n_anchor, n)])
        else:
            # Plain GRPO: one gather, no wave barrier, no exploration stamp anywhere.
            # Reusing make_task(i, False) is what guarantees the stamp is absent.
            _outs = await asyncio.gather(*[make_task(i, False) for i in range(n)])
            wave1, wave2 = list(_outs[:n_anchor]), list(_outs[n_anchor:])
            explore = False
            # Observation only: this split is nominal and gates nothing. These
            # rollouts are clean and independent, so measuring the first
            # n_anchor of them keeps the composition metrics defined over the
            # whole population.
            _, wave1_var = self._wave1_low_variance(wave1, config)

        outputs = list(wave1) + list(wave2)
        # Passive per-rollout metadata (constant within the prompt). Auto-flows
        # into non_tensor_batch via _postprocess's all-keys union; consumed by the
        # trainer's rvrl/two_wave* metrics. Baseline runs never see these keys.
        # INVARIANT: both branches above stamp an IDENTICAL key set. A key present
        # on some prompts and absent on others breaks the trainer's cross-prompt
        # concat / metric aggregation.
        stats = self._two_wave_group_stats(wave1, wave2, explore, config)
        for _i, o in enumerate(outputs):
            o.extra_fields["__two_wave_explored__"] = bool(explore)
            o.extra_fields["__wave1_var__"] = float(wave1_var)
            o.extra_fields["__tw_strategy_on__"] = bool(strategy_on)
            # Per-row wave labels. The rollout_n < k_explore bucket is wrong
            # under two-wave, because n_anchor derives from anchor_fraction
            # while k_explore is a separate knob. These two fields are stamped
            # by the code that made the decision, and `__tw_perturbed__` is the
            # only per-row flag that says a rollout carried the exploration
            # logits processor.
            o.extra_fields["__tw_wave__"] = 1 if _i < n_anchor else 2
            o.extra_fields["__tw_perturbed__"] = bool(explore and _i >= n_anchor)
            for _k, _v in stats.items():
                o.extra_fields[_k] = _v

        return outputs

    def _two_wave_strategy_on(self, prompt_ordinal: int, config) -> bool:
        """Whether the two-wave strategy runs for this prompt.

        Gates the strategy, not the exploration. A strategy-on prompt still
        explores only when its wave-1 group is degenerate; a strategy-off prompt
        takes a plain-GRPO path with no wave split and no exploration.

        The strategy runs on every prompt when the toggle is disabled or the
        period is 1.
        """
        if not getattr(config.exploration, "two_wave_toggle_enable", False):
            return True
        period = int(getattr(config.exploration, "two_wave_toggle_period", 2))
        if period <= 1:
            return True
        phase = int(getattr(config.exploration, "two_wave_toggle_phase", 1))
        return (int(prompt_ordinal) % period) == (phase % period)

    def _wave1_low_variance(self, wave1: list, config) -> tuple:
        """Return (explore: bool, variance: float) for the wave-1 group.

        Pulls a per-rollout scalar (``accuracy`` from reward_extra_info, or the
        blended ``reward_score``), computes population variance, and decides
        whether wave-2 should explore: variance <= variance_threshold AND
        mean <= explore_max_mean. Fail-safe: drop aborted / missing-score
        rollouts; if fewer than 2 valid scores remain, treat the group as
        high-variance (do NOT explore) — never crash, never hang.
        """
        metric = str(getattr(config.exploration, "variance_metric", "accuracy"))
        threshold = float(getattr(config.exploration, "variance_threshold", 0.0))
        explore_max_mean = float(getattr(config.exploration, "explore_max_mean", 1.0))

        scores = []
        for o in wave1:
            # partial_rollout aborts may carry no usable score; filter them.
            if getattr(o, "stop_reason", None) in ("aborted", "abort"):
                continue
            if metric == "reward":
                s = getattr(o, "reward_score", None)
            else:
                info = getattr(o, "extra_fields", None) or {}
                rei = info.get("reward_extra_info")
                s = rei.get("accuracy") if isinstance(rei, dict) else None
            if s is None:
                continue
            try:
                scores.append(float(s))
            except (TypeError, ValueError):
                continue

        if len(scores) < 2:
            return False, 0.0
        mean = sum(scores) / len(scores)
        var = sum((x - mean) ** 2 for x in scores) / len(scores)  # population variance
        explore = bool(var <= threshold and mean <= explore_max_mean)
        return explore, var

    def _two_wave_group_stats(self, wave1: list, wave2: list, explored: bool, config) -> dict:
        """Passive per-group two-wave diagnostics (pure Python, no numpy).

        Returns a flat dict of scalars and bools, constant within the prompt,
        that the dispatcher stamps onto every output and the trainer aggregates
        into ``rvrl/two_wave/*`` metrics: which groups the variance gate fires
        on (all-correct, all-wrong, mixed), and whether the wave-2 exploration
        manufactures variance in a degenerate group.

        A rollout counts as correct iff its scalar exceeds 0.5. Aborted and
        missing-score rollouts are dropped, mirroring ``_wave1_low_variance``.

        The remaining fields describe the STRUCTURE of that variance: an ANOVA
        split, a Bernoulli null, a behavioural-divergence measure and an ungated
        coverage statistic that gives the triggered rates a control arm.
        """
        metric = str(getattr(config.exploration, "variance_metric", "accuracy"))

        def _scalar(o):
            if getattr(o, "stop_reason", None) in ("aborted", "abort"):
                return None
            if metric == "reward":
                s = getattr(o, "reward_score", None)
            else:
                info = getattr(o, "extra_fields", None) or {}
                rei = info.get("reward_extra_info")
                s = rei.get("accuracy") if isinstance(rei, dict) else None
            if s is None:
                return None
            try:
                return float(s)
            except (TypeError, ValueError):
                return None

        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        def _var(xs):
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return sum((x - m) ** 2 for x in xs) / len(xs)  # population variance

        def _valid(ws):
            """[(scalar, output)] for the rollouts that carry a usable score."""
            out = []
            for o in ws:
                s = _scalar(o)
                if s is not None:
                    out.append((s, o))
            return out

        w1_pairs = _valid(wave1)
        w2_pairs = _valid(wave2)
        w1 = [s for s, _ in w1_pairs]
        w2 = [s for s, _ in w2_pairs]
        alls = w1 + w2

        w1_mean = _mean(w1)
        w1_var = _var(w1)
        # Classify the WHOLE population (not just triggered) by the wave-1 group.
        degenerate = len(w1) >= 2 and w1_var <= 1e-9
        all_correct = bool(degenerate and w1_mean >= 0.5)
        all_wrong = bool(degenerate and w1_mean < 0.5)
        mixed = bool(not degenerate)

        group_var_after = _var(alls)  # variance of the FULL n-group after both waves
        triggered = bool(explored)
        # Did wave-2 exploration actually break the degenerate group (revive grad)?
        var_manufactured = bool(triggered and group_var_after > 1e-9)
        # Directionality on triggered groups:
        cracked_all_wrong = bool(triggered and all_wrong and any(s > 0.5 for s in w2))
        broke_all_correct = bool(triggered and all_correct and any(s <= 0.5 for s in w2))

        # ---- variance structure ------------------------------------------------
        # `group_var_after` says how much variance the group ended up with, not
        # where it came from. The three quantities below separate the wave
        # contrast from resampling noise.
        #
        # 1. ANOVA split. With population variance the decomposition is exact:
        #       total = within + between
        #    where `within` is the size-weighted mean of each wave's own variance
        #    and `between` is the size-weighted variance of the two wave means.
        #    `between_share = between / total` is the fraction of the group's
        #    GRPO signal that the wave contrast itself carries.
        n1, n2 = len(w1), len(w2)
        n_all = n1 + n2
        between_share = 0.0
        if n_all > 0 and group_var_after > 1e-12:
            m_all = _mean(alls)
            between = (n1 * (_mean(w1) - m_all) ** 2 + n2 * (_mean(w2) - m_all) ** 2) / n_all
            between_share = min(max(between / group_var_after, 0.0), 1.0)

        # 2. Null variance. If wave-2 had not been perturbed it would be drawn
        #    from the same distribution as wave-1, so the group's expected
        #    population variance is that of n_all Bernoulli(p) draws with
        #    p = wave-1 accuracy: p(1-p)(n_all-1)/n_all. The trainer reports
        #    mean(group_var_after) - mean(var_null) as the excess variance. On a
        #    degenerate wave-1, p is exactly 0 or 1 and var_null is exactly 0;
        #    on mixed groups it is a noisy estimate.
        var_null = 0.0
        if n_all > 1:
            var_null = w1_mean * (1.0 - w1_mean) * (n_all - 1) / n_all

        # 3. Behavioural divergence. Reward variance is a 0/1 outcome measure,
        #    so two rollouts can score identically and still follow different
        #    trajectories. Every rollout in a group shares a byte-identical
        #    prompt, so the token index at which a rollout departs from the
        #    wave-1 reference measures when the group fans out. The reported
        #    signal is (lcp_w2 - lcp_w1), in tokens. -1.0 means not computable
        #    (no reference rollout); the trainer drops negatives rather than
        #    averaging them as zeros.
        lcp_w1, lcp_w2 = -1.0, -1.0

        def _as_ids(_obj):
            """Normalise response_ids to a plain list, or None if unusable.

            `response_ids` is declared as `list[int]` on AgentLoopOutput and as a
            `torch.Tensor` on the padded output, and this call site receives the
            tensor one, so it must never be tested for truth directly.
            Converting once also keeps the LCP loop below off per-element tensor
            indexing.
            """
            _ids = getattr(_obj, "response_ids", None)
            if _ids is None:
                return None
            if hasattr(_ids, "tolist"):
                _ids = _ids.tolist()
            return _ids if len(_ids) else None

        ref = _as_ids(w1_pairs[0][1]) if w1_pairs else None
        if ref is not None:
            def _lcp_mean(pairs, skip_first):
                lens = []
                for _s, _o in pairs[1:] if skip_first else pairs:
                    ids = _as_ids(_o)
                    if ids is None:
                        continue
                    lim = min(len(ids), len(ref), _TW_LCP_CAP)
                    j = 0
                    while j < lim and ids[j] == ref[j]:
                        j += 1
                    lens.append(j)
                return sum(lens) / len(lens) if lens else -1.0

            lcp_w1 = _lcp_mean(w1_pairs, True)   # reference excluded from its own mean
            lcp_w2 = _lcp_mean(w2_pairs, False)

        # 4. Coverage. pass@n over the full group against pass@n_anchor over
        #    wave-1 alone, computed on every group: triggered, untriggered and
        #    strategy-off alike, so the triggered rate has a control.
        pass_any_w1 = any(s > 0.5 for s in w1)
        pass_any_w2 = any(s > 0.5 for s in w2)
        pass_any = pass_any_w1 or pass_any_w2

        return {
            "__tw_all_correct__": all_correct,
            "__tw_all_wrong__": all_wrong,
            "__tw_mixed__": mixed,
            "__tw_triggered__": triggered,
            "__tw_var_manufactured__": var_manufactured,
            "__tw_group_var_after__": float(group_var_after),
            "__tw_cracked_all_wrong__": cracked_all_wrong,
            "__tw_broke_all_correct__": broke_all_correct,
            "__tw_w1_acc__": float(w1_mean),
            "__tw_w2_acc__": float(_mean(w2)),
            # variance-structure fields
            "__tw_between_share__": float(between_share),
            "__tw_var_null__": float(var_null),
            "__tw_lcp_w1__": float(lcp_w1),
            "__tw_lcp_w2__": float(lcp_w2),
            "__tw_pass_any__": bool(pass_any),
            "__tw_pass_any_w1__": bool(pass_any_w1),
            # Not gated on `triggered`: the untriggered and strategy-off groups
            # are the control arm for the triggered rate.
            "__tw_w2_solves_new__": bool(pass_any_w2 and not pass_any_w1),
        }

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            # Stamp rollout_n / sample_index so trainer can slice metrics by
            # explore (rollout_n < k_explore) vs anchor bucket. Cheap and
            # always-on; baseline runs simply ignore the keys.
            output.extra_fields["__rollout_n__"] = int(trajectory["rollout_n"])
            output.extra_fields["__sample_index__"] = int(trajectory["sample_index"])
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score([output], kwargs=kwargs)
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)

        # Densify the per-sample exploration_drop_positions lists into a dense
        # [bsz, response_length] BoolTensor. Only added when at least one sample
        # carries a non-empty drop list (so baseline runs see no extra key).
        has_drop = any(
            input.extra_fields.get("exploration_drop_positions") for input in inputs
        )
        if has_drop:
            response_length = response_mask.size(1)
            drop_mask = torch.zeros_like(response_mask, dtype=torch.bool)
            for i, input_item in enumerate(inputs):
                positions = input_item.extra_fields.get("exploration_drop_positions") or []
                for p in positions:
                    if 0 <= int(p) < response_length:
                        drop_mask[i, int(p)] = True
            optional_outputs["exploration_drop_mask"] = drop_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        # Union over all samples (not just sample 0) — if the first sample's
        # reward function errored and returned an empty dict, single-sample-key
        # extraction would drop format/accuracy keys for the whole batch.
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = sorted({k for info in reward_extra_infos for k in (info or {}).keys()})
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([(info or {}).get(key, 0.0) for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
