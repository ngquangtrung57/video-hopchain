#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "$REPO"
source "${SCRIPT_DIR}/../common_env.sh"
source "${SCRIPT_DIR}/../dataset_groups.sh"
source .env 2>/dev/null || true

trainer_nnodes=${TRAINER_NNODES:-1}
n_gpus_training=8
rollout_nnodes=${ROLLOUT_NNODES:-3}
n_gpus_rollout=8
gen_tp=2

adv_estimator=grpo
loss_mode=vanilla

train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
require_batches=8
trigger_parameter_sync_step=4

max_prompt_length=5376
max_response_length=16384
max_pixels=50000

total_epochs=${TOTAL_EPOCHS:-4}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}

CKPT_SAVE_CONTENTS=${CKPT_SAVE_CONTENTS:-"[model,optimizer,extra,hf_model]"}
val_before_train=${VAL_BEFORE_TRAIN:-false}
total_rollout_steps=200000

lr=1e-6
lr_warmup_steps=25
weight_decay=0.1

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.3

n_resp_per_prompt=8
temperature=1.0
top_p=1.0
top_k=-1

param_offload=false
optimizer_offload=false
ref_offload=true
entropy_checkpointing=false
reshard_after_forward=true
fsdp_size=-1

sp_size=1
use_dynamic_bsz=true
max_model_len=$((max_prompt_length + max_response_length))

actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$max_model_len}
infer_ppo_max_token_len=$((max_model_len * 2))

gpu_memory_utilization=${GPU_MEM_UTIL:-0.80}
max_num_batched_tokens=$max_model_len
enable_chunked_prefill=true
enable_prefix_caching=true
enforce_eager=true

rollout_agent_workers=${ROLLOUT_AGENT_WORKERS:-32}

staleness_threshold=0.5
partial_rollout=True
use_trainer_do_validate=False
nccl_timeout=3600000

DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-64}
shuffle_dataset=true
return_raw_chat=True
MODEL_PATH="${MODEL_PATH_OVERRIDE:?set MODEL_PATH_OVERRIDE to the initial checkpoint}"

TRAIN_GROUP_NAME="GROUP_VIDEO_TRAIN_HOPCHAIN"
TRAIN_FILES_ARG=$(join_to_list "${GROUP_VIDEO_TRAIN_HOPCHAIN[@]}")
TRAIN_GROUP_REF=("${GROUP_VIDEO_TRAIN_HOPCHAIN[@]}")
VAL_FILES_ARG=$(join_to_list "${GROUP_VAL_HOPCHAIN[@]}")

for tf in "${TRAIN_GROUP_REF[@]}"; do
    if [ ! -f "$tf" ]; then
        echo "ERROR: train parquet $tf not found." >&2
        exit 1
    fi
done
for vf in "${GROUP_VAL_HOPCHAIN[@]}"; do
    if [ ! -f "$vf" ]; then
        echo "ERROR: val parquet $vf not found." >&2
        exit 1
    fi
done

experiment_name="${EXP_NAME_OVERRIDE:-hopchain_grpo}"
PROJECT_FOLDER="${OUTPUT_BASE}/${experiment_name}"

echo "============================================"
echo "Algo: GRPO — ${trainer_nnodes}T+${rollout_nnodes}R split, $(( (trainer_nnodes + rollout_nnodes) * 8 )) GPUs"
echo "Data: train=$TRAIN_FILES_ARG val=$VAL_FILES_ARG"
echo "Val:  val_before_train=$val_before_train  test_freq=$test_freq  max_samples=${VAL_MAX_SAMPLES:-500}"
echo "Run:  total_epochs=$total_epochs  save_freq=$save_freq  ckpt_contents=$CKPT_SAVE_CONTENTS"
echo "Init: $MODEL_PATH"
echo "Fmt:  VERL_THINK_PREFILL=${VERL_THINK_PREFILL:-0}  VERL_GRADED_THINK_FORMAT=${VERL_GRADED_THINK_FORMAT:-0}"
echo "Explore: enable=${EXPLORE_ENABLE:-false} cge=${CGE_ENABLE:-false} anchor_fraction=0.5 var_thr=0.0 metric=accuracy explore_max_mean=${EXPLORE_MAX_MEAN:-1.0} tau=${TOP_PROB_THRESHOLD:-0.95} mask_from_loss=true"
echo "Explore: trigger=${TRIGGER_MODE:-high} tau=${TOP_PROB_THRESHOLD:-0.95} band=[${TAU_LOW:-0.8},${TAU_HIGH:-0.95}] | filter special=${SKIP_SPECIAL:-true} ws_punct=${SKIP_WS_PUNCT:-true} digits=${SKIP_DIGITS:-true} subword_cont=${SKIP_SUBWORD:-false} | mask_ipc=${VERL_EXPLORATION_IPC_DIR:-/tmp/rvrl_drop_pos}"
echo "Seq:  prompt=$max_prompt_length  response=$max_response_length  model_len=$max_model_len  pixels=$max_pixels"
echo "Exp:  $experiment_name"
echo "Trainer: $trainer_nnodes × $n_gpus_training = $((trainer_nnodes * n_gpus_training)) GPU (FSDP dp=$((trainer_nnodes * n_gpus_training)))"
echo "Mini-batch: $train_prompt_mini_bsz prompts × $n_resp_per_prompt = $((train_prompt_mini_bsz * n_resp_per_prompt)) seq; dp=$((trainer_nnodes * n_gpus_training)) divides it: $(( (train_prompt_mini_bsz * n_resp_per_prompt) % (trainer_nnodes * n_gpus_training) == 0 ? 1 : 0 )) (must be 1)"
echo "Rollout: $rollout_nnodes × $n_gpus_rollout = $((rollout_nnodes * n_gpus_rollout)) GPU (TP=$gen_tp → $((rollout_nnodes * n_gpus_rollout / gen_tp)) vLLM replicas)"
echo "Global batch: $train_prompt_mini_bsz × $require_batches × $n_resp_per_prompt = $((train_prompt_mini_bsz * require_batches * $n_resp_per_prompt)) rollouts/step"
echo "Async: staleness=$staleness_threshold partial=$partial_rollout trigger=$trigger_parameter_sync_step"
echo "RAY_ADDRESS=$RAY_ADDRESS"
echo "============================================"

mkdir -p "$OUTPUT_BASE"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path="$REPO/verl/verl/experimental/fully_async_policy/config" \
    --config-name="fully_async_ppo_trainer" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.train_files="${TRAIN_FILES_ARG}" \
    data.val_files="${VAL_FILES_ARG}" \
    data.val_max_samples=${VAL_MAX_SAMPLES:-500} \
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
    data.shuffle=${shuffle_dataset} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.image_key=images \
    data.video_key=videos \
    +data.max_pixels=${max_pixels} \
    data.filter_overlong_prompts=false \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS:-false} \
    ++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL:-flash_attention_2} \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND:-torch} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers=${rollout_agent_workers} \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${reshard_after_forward} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=${FORWARD_PREFETCH:-false} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=${CALCULATE_ENTROPY:-true} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.enable_prefix_caching=${enable_prefix_caching} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path="${RLVR_ROOT}/rewards/vero_reward_wrapper.py" \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.rollout.exploration.enable=${EXPLORE_ENABLE:-false} \
    actor_rollout_ref.rollout.exploration.two_wave_enable=${CGE_ENABLE:-false} \
    actor_rollout_ref.rollout.exploration.anchor_fraction=0.5 \
    actor_rollout_ref.rollout.exploration.variance_threshold=0.0 \
    actor_rollout_ref.rollout.exploration.variance_metric=accuracy \
    actor_rollout_ref.rollout.exploration.explore_max_mean=${EXPLORE_MAX_MEAN:-1.0} \
    actor_rollout_ref.rollout.exploration.trigger_mode=${TRIGGER_MODE:-high} \
    actor_rollout_ref.rollout.exploration.tau_low=${TAU_LOW:-0.8} \
    actor_rollout_ref.rollout.exploration.tau_high=${TAU_HIGH:-0.95} \
    actor_rollout_ref.rollout.exploration.skip_special_tokens=${SKIP_SPECIAL:-true} \
    actor_rollout_ref.rollout.exploration.skip_whitespace_punct=${SKIP_WS_PUNCT:-false} \
    actor_rollout_ref.rollout.exploration.skip_digit_tokens=${SKIP_DIGITS:-false} \
    actor_rollout_ref.rollout.exploration.skip_subword_continuation=${SKIP_SUBWORD:-false} \
    actor_rollout_ref.rollout.exploration.top_prob_threshold=${TOP_PROB_THRESHOLD:-0.95} \
    actor_rollout_ref.rollout.exploration.drop_top_k=1 \
    actor_rollout_ref.rollout.exploration.perturb_prob=1.0 \
    actor_rollout_ref.rollout.exploration.deterministic=true \
    actor_rollout_ref.rollout.exploration.mask_from_loss=true \
    actor_rollout_ref.rollout.exploration.min_position=0 \
    actor_rollout_ref.rollout.exploration.max_perturbations_per_seq=16384 \
    actor_rollout_ref.rollout.exploration.restrict_to_think_region=true \
    actor_rollout_ref.rollout.exploration.selection_seed=42 \
    actor_rollout_ref.rollout.exploration.record_details=${DROP_RECORD_DETAILS:-true} \
    actor_rollout_ref.rollout.exploration.record_entropy=${DROP_RECORD_ENTROPY:-true} \
    actor_rollout_ref.rollout.exploration.detail_topn=${DROP_DETAIL_TOPN:-8} \
    actor_rollout_ref.rollout.exploration.detail_max_per_request=${DROP_DETAIL_MAX_PER_REQ:-512} \
    actor_rollout_ref.rollout.exploration.drop_dump_dir=${DROP_DUMP_DIR:-${OUTPUT_BASE}/${experiment_name}/exploration_drops} \
    actor_rollout_ref.rollout.exploration.drop_dump_max_rows=${DROP_DUMP_MAX_ROWS:-2000} \
    actor_rollout_ref.rollout.exploration.drop_trace_dir=${DROP_TRACE_DIR:-${OUTPUT_BASE}/${experiment_name}/drop_trace} \
    actor_rollout_ref.rollout.exploration.drop_trace_max_rows=${DROP_TRACE_MAX_ROWS:-20000} \
    actor_rollout_ref.rollout.exploration.measure_entropy=${MEASURE_ENTROPY:-true} \
    actor_rollout_ref.rollout.exploration.entropy_measure_stride=${ENTROPY_STRIDE:-1} \
    actor_rollout_ref.rollout.exploration.entropy_post_drop_window=${ENTROPY_POST_WINDOW:-16} \
    actor_rollout_ref.rollout.exploration.entropy_seq_trace_max_rows=${ENTROPY_SEQ_MAX_ROWS:-40000} \
    "trainer.logger=${TRAINER_LOGGER:-[\"console\"]}" \
    trainer.project_name="${WANDB_PROJECT:-video_hopchain}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.nnodes="${trainer_nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${rollout_nnodes}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${PROJECT_FOLDER}" \
    actor_rollout_ref.actor.checkpoint.async_save=true \
    "actor_rollout_ref.actor.checkpoint.save_contents=${CKPT_SAVE_CONTENTS}" \
    trainer.resume_mode=${RESUME_MODE:-auto} \
    trainer.log_val_generations=200 \
    trainer.rollout_data_dir="${PROJECT_FOLDER}/rollout/train" \
    trainer.validation_data_dir="${PROJECT_FOLDER}/rollout/val" \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    async_training.use_trainer_do_validate=False
