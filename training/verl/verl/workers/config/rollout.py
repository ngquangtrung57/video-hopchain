# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import warnings
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.disaggregation import DisaggregationConfig
from verl.workers.config.model import MtpConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "ExplorationConfig",
    "RolloutConfig",
    "CheckpointEngineConfig",
    "SkipConfig",
]


@dataclass
class SkipConfig(BaseConfig):
    """
    Configuration for rollout skip: load/dump previously generated rollout data
    instead of computing new rollouts (e.g. for debugging or reuse).
    """

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    max_dump_step: int = 1
    action: str = "cache"  # cache | repeat | repeat_last

    def get(self, key: str, default=None):
        """Dict-like get for compatibility with code that uses skip.get('enable', False)."""
        return getattr(self, key, default)


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class ExplorationConfig(BaseConfig):
    """Controlled-exploration config for entropy-aware token dropout at rollout.

    When enabled, the first ``k_explore`` of ``rollout.n`` responses per prompt
    use a logits processor that masks the top-1 token at trigger positions
    (selected per ``trigger_mode``) with probability ``perturb_prob``. The
    remaining rollouts are unperturbed anchors. When disabled (default),
    rollout behavior is identical to baseline.

    ``trigger_mode``:
      - ``"high"`` (default): perturb where ``max_prob > top_prob_threshold``.
      - ``"band"``: perturb where ``tau_low < max_prob < tau_high``.
    """

    enable: bool = False
    k_explore: int = 4
    # Trigger selector: "high" thresholds the top-1 probability, "band" uses a
    # two-sided window.
    trigger_mode: str = "high"
    top_prob_threshold: float = 0.9       # used when trigger_mode == "high"
    tau_low: float = 0.3                  # used when trigger_mode == "band" (lower edge)
    tau_high: float = 0.7                 # used when trigger_mode == "band" (upper edge)
    perturb_prob: float = 0.3
    drop_top_k: int = 1
    min_position: int = 8
    max_perturbations_per_seq: int = 32
    restrict_to_think_region: bool = True
    think_open_token_id: int = 151667  # Qwen3-VL <think>
    think_close_token_id: int = 151668  # Qwen3-VL </think>
    seed: Optional[int] = None
    prompt_exploration_prob: float = 1.0   # 1.0 = every prompt eligible
    deterministic: bool = False            # False = stochastic perturb_prob gate
    # Nominal switch for removing masked positions from the policy gradient.
    # It is not read anywhere: losses.py removes them whenever the drop mask is
    # present, so masking is unconditional.
    mask_from_loss: bool = False
    selection_seed: Optional[int] = None   # accepted but not read; the per-prompt draw is unseeded
    # Two-wave variance-targeted exploration. When enabled, each TRAINING
    # prompt's n rollouts split into wave-1 anchors [0, n_anchor), which are
    # never explored, and wave-2 [n_anchor, n); all of wave-2 is explored iff
    # the wave-1 reward variance is low, i.e. the group is degenerate and
    # carries no GRPO gradient. Validation is never affected.
    two_wave_enable: bool = False          # master gate
    anchor_fraction: float = 0.5           # wave-1 share of n (n=8 => 4 anchor / 4 conditional)
    variance_threshold: float = 0.0        # explore wave-2 iff wave1 var <= this (0 => all-identical)
    variance_metric: str = "accuracy"      # "accuracy" | "reward" — per-rollout scalar for variance
    explore_max_mean: float = 1.0          # explore only if wave1 mean <= this (1.0 => regardless of mean)
    # Two-wave duty cycle. When disabled, every prompt runs the two-wave
    # strategy. When enabled, the STRATEGY runs only on prompts whose ordinal
    # satisfies (ordinal % period == phase); every other prompt takes a
    # plain-GRPO path with no wave split and no exploration stamp. The toggle
    # gates the strategy, not the exploration: inside a strategy-on prompt the
    # explore decision is still driven by wave-1 variance.
    two_wave_toggle_enable: bool = False   # master gate
    two_wave_toggle_period: int = 2        # strategy runs on 1 of every N prompts; 1 => always
    two_wave_toggle_phase: int = 1         # which residue is ON
    # Aggregate drop statistics (mean log-prob of the dropped token, probability
    # mass removed, forced log-prob penalty) are always collected by the logits
    # processor and surface as rvrl/lp_*. These knobs control the expensive
    # per-drop record:
    #   record_details      per-drop rows (position, dropped token id, its
    #                       log-prob, the surviving runner-up, entropy) carried
    #                       back on each rollout and aggregated per group.
    #   record_entropy      full-vocab entropy at each dropped position.
    #   detail_topn         pre-drop top-N candidates kept per drop. vLLM returns
    #                       log-probs computed from the MASKED logits, so the
    #                       sampled token's ORIGINAL log-prob is otherwise
    #                       unrecoverable; this is what makes it recoverable.
    #   drop_dump_dir       when set, per-drop records are written as JSONL for
    #                       offline study. Empty => no dump.
    #   drop_dump_max_rows  cap on rows written per parameter version.
    # Whether the PROMPT already carries the opening <think> tag (VERL_THINK_PREFILL=1).
    # MUST be true whenever prefill is on and restrict_to_think_region is also on, or the
    # logits processor waits for a <think> the model will never emit and exploration is a
    # permanent no-op. The LP defaults this from the live env var, so leaving it False here
    # is safe; set it explicitly only to override.
    think_prefilled: bool = False
    record_details: bool = False
    record_entropy: bool = True
    detail_topn: int = 8
    detail_max_per_request: int = 512
    drop_dump_dir: str = ""
    drop_dump_max_rows: int = 2000
    # Worker-side drop trace. The _DROPPED_DETAILS side-channel cannot carry
    # it: the logits processor is built inside the vLLM worker while the drain
    # runs in the server, so their module globals never meet.
    drop_trace_dir: str = ""
    drop_trace_max_rows: int = 20000
    # ---- entropy instrumentation ----------------------------------------
    # Observation only: no gate, no gradient, no change to which tokens are
    # dropped.
    #
    #   measure_entropy       master switch for everything below.
    #   entropy_measure_stride
    #                         measure every Nth decode step. 1 = every step.
    #   entropy_post_drop_window
    #                         after each drop, keep measuring entropy for W
    #                         further tokens of that sequence. 0 = off.
    #   entropy_seq_trace_max_rows
    #                         cap on per-sequence entropy rows written to the
    #                         trace dir. Separate from drop_trace_max_rows so a
    #                         capped drop trace cannot starve the entropy rows.
    measure_entropy: bool = True
    entropy_measure_stride: int = 1
    entropy_post_drop_window: int = 16
    entropy_seq_trace_max_rows: int = 40000
    # Token-class filter: restrict drops to top-1 tokens that carry a real
    # lexical alternative rather than structure (whitespace, morpheme
    # fragments, digits).
    skip_special_tokens: bool = True
    skip_whitespace_punct: bool = False
    skip_digit_tokens: bool = False
    skip_subword_continuation: bool = False
    # How often (in apply() calls) to snapshot the LP counters into the drop
    # trace, so they reach disk independently of the Ray metric path.
    # 0 disables.
    counter_snapshot_every: int = 2000


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    function_tool_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    # Fully qualified class name for custom AgentLoopManager (e.g., "mypackage.module.MyManager").
    # Security: This class will be dynamically imported via importlib. Only use trusted class paths.
    agent_loop_manager_class: Optional[str] = None


@dataclass
class TraceConfig(BaseConfig):
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None
    backend: Optional[str] = None
    token2text: bool = False
    max_samples_per_step_per_worker: Optional[int] = None

    def __post_init__(self):
        if self.max_samples_per_step_per_worker is not None and self.max_samples_per_step_per_worker < 0:
            raise ValueError("`max_samples_per_step_per_worker` must be a non-negative integer or null.")


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class PrometheusConfig(BaseConfig):
    """
    Configuration for Prometheus server
    """

    # whether enable prometheus on server mode rollout
    enable: bool = False
    # Port number that Prometheus listens on, default is 9090
    port: int = 9090
    # Path to Prometheus configuration file
    file: str = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    # Specify served_model_name to avoid displaying overly long model paths in Grafana
    served_model_name: Optional[str] = None


@dataclass
class CheckpointEngineConfig(BaseConfig):
    """
    Configuration for checkpoint engine to update weights from trainer to rollout
    """

    # Backend for checkpoint engine: naive, nccl, nixl, hccl
    backend: Optional[str] = "naive"
    # Bucket size in MB to transfer multiple weights at one time
    update_weights_bucket_megabytes: int = 2048
    # Additional keyword arguments for checkpoint engine
    engine_kwargs: dict = field(default_factory=dict)
    # If set, this Python module is imported on every worker process before the
    # backend is instantiated, allowing custom backends to register themselves
    # in CheckpointEngineRegistry.
    custom_backend_module: Optional[str] = None


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {
        "max_model_len",
        "load_format",
        "engine_kwargs",
        "prompt_length",
        "response_length",
        "expert_parallel_size",
        "moe_tensor_parallel_size",
    }

    name: Optional[str] = MISSING
    mode: str = "async"
    nnodes: int = 0
    n_gpus_per_node: int = 8

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1
    repetition_penalty: float = 1.0

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    moe_tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Extension point for custom configurations
    custom: Optional[dict] = None

    # Fully qualified class name for a custom CheckpointEngineManager. When set, the trainer
    # loads this class instead of the built-in CheckpointEngineManager.
    checkpoint_manager_class: Optional[str] = None

    # Checkpoint Engine config for update weights from trainer to rollout
    checkpoint_engine: CheckpointEngineConfig = field(default_factory=CheckpointEngineConfig)

    # Rollout skip config (load/dump rollout data)
    skip: SkipConfig = field(default_factory=SkipConfig)

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    quantization: Optional[str] = None

    quantization_config_file: Optional[str] = None

    enable_rollout_routing_replay: bool = False

    enable_sleep_mode: bool = True

    mtp: MtpConfig = field(default_factory=MtpConfig)

    qat: Optional[dict] = None

    disaggregation: DisaggregationConfig = field(default_factory=DisaggregationConfig)

    def __post_init__(self):
        """Validate the rollout config"""
        # Deprecation warning for mode field - only async mode is supported
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )
        if self.mode != "async":
            warnings.warn(
                f"Unknown rollout mode '{self.mode}'. Only 'async' mode is supported. "
                "The 'mode' field is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.name != "trtllm" and self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size > 1:
            assert self.name == "trtllm", "moe_tensor_parallel_size is only supported for trtllm"

        if self.name == "trtllm":
            # If either expert_parallel_size or moe_tensor_parallel_size is at default 1,
            # convert to None so TensorRT-LLM treats it as unspecified.
            # When both unspecified: moe_ep_size=1, moe_tp_size=moe_world_size (no EP, all TP).
            # When only one set: the other is auto-derived from tensor_model_parallel_size.
            if self.expert_parallel_size is not None and self.expert_parallel_size == 1:
                self.expert_parallel_size = None
            if self.moe_tensor_parallel_size is not None and self.moe_tensor_parallel_size == 1:
                self.moe_tensor_parallel_size = None
            if self.expert_parallel_size is not None and self.moe_tensor_parallel_size is not None:
                assert self.moe_tensor_parallel_size * self.expert_parallel_size == self.tensor_model_parallel_size, (
                    "moe_tensor_parallel_size * expert_parallel_size must equal tensor_model_parallel_size "
                    f"(got {self.moe_tensor_parallel_size} * {self.expert_parallel_size} = "
                    f"{self.moe_tensor_parallel_size * self.expert_parallel_size}, "
                    f"tensor_model_parallel_size={self.tensor_model_parallel_size})"
                )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang" or self.name == "trtllm":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )

        # Hydra passes this as dict/DictConfig; coerce to dataclass so
        # downstream .enabled etc. work. BaseConfig is frozen, hence object.__setattr__.
        if isinstance(self.disaggregation, dict):
            object.__setattr__(self, "disaggregation", DisaggregationConfig(**self.disaggregation))
        elif not isinstance(self.disaggregation, DisaggregationConfig):
            from omegaconf import DictConfig, OmegaConf

            if not isinstance(self.disaggregation, DictConfig):
                raise TypeError(
                    f"rollout.disaggregation must be dict, DictConfig, or DisaggregationConfig; "
                    f"got {type(self.disaggregation).__name__}."
                )
            object.__setattr__(
                self,
                "disaggregation",
                DisaggregationConfig(**OmegaConf.to_container(self.disaggregation, resolve=True)),
            )

        if self.exploration.enable:
            if self.exploration.k_explore < 0:
                raise ValueError("exploration.k_explore must be >= 0")
            if self.exploration.k_explore > self.n:
                raise ValueError(
                    f"exploration.k_explore ({self.exploration.k_explore}) cannot exceed "
                    f"rollout.n ({self.n})."
                )
            if not 0.0 <= self.exploration.perturb_prob <= 1.0:
                raise ValueError("exploration.perturb_prob must be in [0, 1]")
            if self.exploration.trigger_mode not in ("high", "band"):
                raise ValueError(
                    f"exploration.trigger_mode must be 'high' or 'band', "
                    f"got {self.exploration.trigger_mode!r}"
                )
            if self.exploration.trigger_mode == "high":
                if not 0.0 < self.exploration.top_prob_threshold < 1.0:
                    raise ValueError(
                        "exploration.top_prob_threshold must be in (0, 1) when trigger_mode='high'"
                    )
            else:  # "band"
                if not 0.0 <= self.exploration.tau_low < self.exploration.tau_high <= 1.0:
                    raise ValueError(
                        "exploration.tau_low/tau_high must satisfy 0 <= tau_low < tau_high <= 1 "
                        f"(got tau_low={self.exploration.tau_low}, tau_high={self.exploration.tau_high})"
                    )
            if self.exploration.drop_top_k < 1:
                raise ValueError("exploration.drop_top_k must be >= 1")
            if self.exploration.min_position < 0:
                raise ValueError("exploration.min_position must be >= 0")
            if not 0.0 <= self.exploration.prompt_exploration_prob <= 1.0:
                raise ValueError(
                    "exploration.prompt_exploration_prob must be in [0, 1] "
                    f"(got {self.exploration.prompt_exploration_prob})"
                )
            # phase is taken modulo period at use, so no upper bound is needed
            # and period=1 stays valid with any phase.
            if self.exploration.two_wave_toggle_period < 1:
                raise ValueError(
                    "exploration.two_wave_toggle_period must be >= 1 "
                    f"(got {self.exploration.two_wave_toggle_period})"
                )
            if self.exploration.two_wave_toggle_phase < 0:
                raise ValueError(
                    "exploration.two_wave_toggle_phase must be >= 0 "
                    f"(got {self.exploration.two_wave_toggle_phase})"
                )
            if self.exploration.two_wave_toggle_enable and not self.exploration.two_wave_enable:
                raise ValueError(
                    "exploration.two_wave_toggle_enable=true requires two_wave_enable=true "
                    "(the toggle gates the two-wave strategy; there is no strategy to gate "
                    "when two-wave is off)"
                )

        if self.disaggregation.enabled and self.name != "sglang":
            raise ValueError(
                f"rollout.disaggregation.enabled=True is currently only supported with "
                f"rollout.name='sglang'; got {self.name!r}. (vLLM PD is a tracked follow-up.)"
            )
