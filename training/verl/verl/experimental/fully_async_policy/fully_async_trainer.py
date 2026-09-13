# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

import ray
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None
        # Upstream's RayPPOTrainer.__init__ calls self._init_dump_executor() after
        # _create_dataloader; we don't chain super().__init__() so replicate it
        # explicitly. _dump_generations references self._dump_executor.
        self._init_dump_executor()

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break

        self.progress_bar.close()
        self._fit_save_checkpoint(force=True)
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        self._fit_save_checkpoint()
        await self._fit_validate()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def _fit_collect_metrics(self, batch):
        """Extend base metric collection with rvrl/* exploration metrics.

        Adds these key families (the exploration ones only when
        ``actor_rollout_ref.rollout.exploration.enable`` is True):
          * ``rvrl/lp_*``: logits-processor counters fetched from each vLLM
            server via ``get_exploration_stats.remote()``.
          * ``rvrl/score_*`` / ``rvrl/best_of_group_explore_share`` /
            ``rvrl/response_length_*`` / ``rvrl/format_pass_*``: trainer-side
            per-rollout-bucket slicing using ``__rollout_n__`` (explore:
            rollout_n < k_explore; anchor: >= k_explore). These buckets do not
            line up with the waves under two-wave, where n_anchor derives from
            anchor_fraction and k_explore is a separate knob; the
            ``rvrl/two_wave/*`` keys are labelled by wave instead.
          * ``rvrl/grpo/*``: GRPO gradient reachability
            (``zero_adv_group_frac`` and advantage-mass attribution). Emitted on
            every run including baselines, which are the control arm.
          * ``rvrl/two_wave/*``: group composition plus the variance-structure
            metrics (ANOVA between-wave share, excess over the Bernoulli null,
            behavioural divergence in tokens, pass@n coverage gain), each paired
            with its untriggered control arm.

        Failures in either path are logged and swallowed rather than crashing
        the trainer over a missing metric.
        """
        super()._fit_collect_metrics(batch)
        # OUTSIDE the exploration gate: a baseline run is the control arm for
        # every rvrl/grpo/* number, so the block has to emit there too.
        try:
            self._collect_grpo_variance_metrics(batch, self.metrics)
        except Exception as gv_e:
            if not getattr(self, "_rvrl_grpo_warned", False):
                logger.warning("grpo variance metrics unavailable (non-fatal): %s", gv_e)
                self._rvrl_grpo_warned = True
        try:
            exploration_cfg = getattr(self.config.actor_rollout_ref.rollout, "exploration", None)
            # Hydra may surface 'true'/'True' as string for some structured-config
            # paths, or the field may have been stored as a OmegaConf node that
            # bool() doesn't resolve to Python True. Coerce defensively.
            enable_raw = getattr(exploration_cfg, "enable", None) if exploration_cfg is not None else None
            if isinstance(enable_raw, str):
                enable = enable_raw.lower() in ("true", "1", "yes")
            else:
                enable = bool(enable_raw) if enable_raw is not None else False
            if exploration_cfg is None or not enable:
                return
            # Each helper is isolated: the LP side can fail (for example when
            # async_rollout_manager is not initialized in this pipeline) without
            # blocking the trainer-side rollout-bucket metrics.
            try:
                self._collect_rvrl_lp_metrics(self.metrics)
            except Exception as lp_e:
                if not getattr(self, "_rvrl_lp_warned", False):
                    logger.warning(
                        "rvrl LP metrics unavailable (rollout-bucket metrics still flow): %s",
                        lp_e,
                    )
                    self._rvrl_lp_warned = True
            try:
                self._dump_exploration_drops(batch, self.metrics)
            except Exception as dd_e:
                logger.warning("exploration drop dump failed (non-fatal): %s", dd_e)
            try:
                self._collect_rvrl_rollout_metrics(batch, self.metrics, int(exploration_cfg.k_explore))
            except Exception as ro_e:
                logger.warning("rvrl rollout metrics failed (non-fatal): %s", ro_e)
        except Exception as e:
            logger.warning("rvrl metric collection failed (non-fatal): %s", e)

    def _collect_rvrl_lp_metrics(self, metrics: dict) -> None:
        """Pull LP counters from each vLLM HTTP server and aggregate.

        Two paths to fetch the per-server counters depending on which
        verl trainer pipeline this is:
          1. Trainer-side validation path (``use_trainer_do_validate=true``):
             ``self.async_rollout_manager.server_handles`` exists; iterate.
          2. Fully-async path (this run): the AgentLoopManager lives inside
             the rollouter Ray actor, not on the trainer. Delegate to
             ``self.rollouter.get_exploration_stats.remote()`` which sums
             counters internally and returns the aggregated dict.
        """
        agg = {}
        async_mgr = getattr(self, "async_rollout_manager", None)
        if async_mgr is not None:
            servers = getattr(async_mgr, "server_handles", None) or []
            if servers:
                try:
                    stats_list = ray.get(
                        [s.get_exploration_stats.remote() for s in servers],
                        timeout=30.0,
                    )
                    from verl.workers.rollout.exploration import merge_lp_counters
                    for snap in stats_list:
                        merge_lp_counters(agg, snap)
                except Exception as e:
                    logger.warning("rvrl lp stats fetch (direct) failed: %s", e)
        # Fallback: ask the rollouter to aggregate for us. This path covers
        # the fully-async pipeline where the trainer doesn't see the rollout
        # servers directly.
        if not agg:
            rollouter = getattr(self, "rollouter", None)
            if rollouter is not None:
                try:
                    agg = ray.get(rollouter.get_exploration_stats.remote(), timeout=30.0)
                    if not isinstance(agg, dict):
                        agg = {}
                except Exception as e:
                    logger.warning("rvrl lp stats fetch (via rollouter) failed: %s", e)
                    agg = {}
        # Last-resort fallback: read the counter snapshots the logits processor
        # writes into its trace dir. Both Ray paths above ask a server actor for
        # counters that live in module globals of the vLLM worker process, and
        # those globals never meet, so the trace file is the remaining channel
        # out of the worker.
        if not agg:
            agg = self._rvrl_counters_from_trace()
        if not agg:
            return
        active = max(int(agg.get("active_positions", 0)), 1)
        triggered = int(agg.get("triggered", 0))
        perturbed = int(agg.get("perturbed", 0))
        finished = int(agg.get("finished_seqs", 0))
        active_steps = max(int(agg.get("active_seq_count_steps", 0)), 1)

        rvrl = {
            "rvrl/lp_trigger_rate": triggered / active,
            "rvrl/lp_perturb_rate": (perturbed / triggered) if triggered > 0 else 0.0,
            "rvrl/lp_perturb_per_seq_mean": (
                int(agg.get("perturb_count_sum", 0)) / finished
            ) if finished > 0 else 0.0,
            "rvrl/lp_cap_hit_rate": (
                int(agg.get("cap_hit_seqs", 0)) / finished
            ) if finished > 0 else 0.0,
            "rvrl/lp_skipped_outside_think": int(agg.get("skipped_outside_think", 0)) / active,
            "rvrl/lp_skipped_min_pos": int(agg.get("skipped_min_pos", 0)) / active,
            "rvrl/lp_active_seq_mean": int(agg.get("active_seq_count_sum", 0)) / active_steps,
        }

        # ---- drop statistics -----------------------------------------------
        # Every mean below divides by ``perturbed`` (the number of masked
        # positions), so they are per-drop averages, not per-token.
        if perturbed > 0:
            _d = float(perturbed)
            dropped_logp_mean = float(agg.get("dropped_logp_sum", 0.0)) / _d
            runner_up_logp_mean = float(agg.get("runner_up_logp_sum", 0.0)) / _d
            rvrl.update({
                # Mean log-prob the model assigned to the removed token: near 0
                # means the trigger fires on near-certain tokens.
                "rvrl/drop/logp_mean": dropped_logp_mean,
                "rvrl/drop/prob_mean": float(agg.get("dropped_prob_sum", 0.0)) / _d,
                # Total probability mass removed per drop (equal to prob_mean
                # when drop_top_k=1).
                "rvrl/drop/mass_mean": float(agg.get("dropped_mass_sum", 0.0)) / _d,
                # Log-prob of the best surviving token: where the sampler lands.
                "rvrl/drop/runner_up_logp_mean": runner_up_logp_mean,
                # mean [log p(dropped) - log p(survivor)]: the log-prob penalty
                # the intervention forces on the policy.
                "rvrl/drop/forced_logp_delta_mean": (
                    float(agg.get("forced_logp_delta_sum", 0.0)) / _d
                ),
            })
            _ent_n = int(agg.get("dropped_entropy_n", 0))
            if _ent_n > 0:
                rvrl["rvrl/drop/entropy_mean"] = (
                    float(agg.get("dropped_entropy_sum", 0.0)) / _ent_n
                )
            # Fraction of drops landing in each decile of top-1 confidence.
            _hist = agg.get("drop_top1_hist")
            if isinstance(_hist, list) and sum(_hist) > 0:
                _tot = float(sum(_hist))
                for _b, _c in enumerate(_hist):
                    rvrl[f"rvrl/drop/top1_hist_{_b}"] = _c / _tot
        if triggered > 0:
            rvrl["rvrl/drop/trigger_top1_prob_mean"] = (
                float(agg.get("trigger_top1_prob_sum", 0.0)) / float(triggered)
            )

        # ---- entropy ---------------------------------------------------------
        # The trigger is log-prob based; entropy is a property of the whole
        # next-token distribution, so these keys report the shape of the
        # residual mass that p_top1 alone does not determine. All of them are
        # observation only: no gate, no gradient, no change to which tokens are
        # dropped.
        _hn = int(agg.get("drop_H_post_n", 0))
        if _hn > 0:
            _hp_mean = float(agg.get("drop_H_post_sum", 0.0)) / _hn
            _hp_var = max(
                float(agg.get("drop_H_post_sq_sum", 0.0)) / _hn - _hp_mean ** 2, 0.0
            )
            rvrl.update({
                # Entropy of the distribution the sampler draws from after the
                # mask.
                "rvrl/entropy/drop_post_mask_mean": _hp_mean,
                "rvrl/entropy/drop_post_mask_std": _hp_var ** 0.5,
                # H_post - H_orig. Negative means the mask closed the position:
                # deleting the top of a two-way split leaves a forced remainder.
                "rvrl/entropy/drop_delta_mean": (
                    float(agg.get("drop_dH_sum", 0.0)) / _hn
                ),
                # p2/(1-p1): ~1 is a binary fork, ~0 a diffuse tail.
                "rvrl/entropy/drop_residual_concentration": (
                    float(agg.get("drop_resid_conc_sum", 0.0)) / _hn
                ),
                # Population split of the above.
                "rvrl/entropy/drop_near_forced_frac": (
                    int(agg.get("drop_H_post_near_forced", 0)) / _hn
                ),
                "rvrl/entropy/drop_open_fork_frac": (
                    int(agg.get("drop_H_post_open", 0)) / _hn
                ),
            })
        # Per-token policy entropy, split by wave. This is the paired
        # comparison: the same prompts produce both arms in the same batch, so
        # the difference is not confounded by prompt difficulty.
        _ne = int(agg.get("ent_tok_n_explore", 0))
        _na = int(agg.get("ent_tok_n_anchor", 0))
        if _ne > 0:
            _me = float(agg.get("ent_tok_sum_explore", 0.0)) / _ne
            rvrl["rvrl/entropy/seq_explore_mean"] = _me
            rvrl["rvrl/entropy/seq_explore_std"] = max(
                float(agg.get("ent_tok_sq_explore", 0.0)) / _ne - _me ** 2, 0.0
            ) ** 0.5
        if _na > 0:
            _ma = float(agg.get("ent_tok_sum_anchor", 0.0)) / _na
            rvrl["rvrl/entropy/seq_anchor_mean"] = _ma
            rvrl["rvrl/entropy/seq_anchor_std"] = max(
                float(agg.get("ent_tok_sq_anchor", 0.0)) / _na - _ma ** 2, 0.0
            ) ** 0.5
        if _ne > 0 and _na > 0:
            # Mean per-token entropy of the perturbed wave minus that of its
            # clean control on the same prompts.
            rvrl["rvrl/entropy/seq_explore_minus_anchor"] = (
                float(agg.get("ent_tok_sum_explore", 0.0)) / _ne
                - float(agg.get("ent_tok_sum_anchor", 0.0)) / _na
            )
        _pn = int(agg.get("post_drop_ent_n", 0))
        if _pn > 0 and _ne > 0:
            _pm = float(agg.get("post_drop_ent_sum", 0.0)) / _pn
            rvrl["rvrl/entropy/post_drop_window_mean"] = _pm
            # Entropy in the W tokens after a drop, relative to the explore
            # wave's own mean.
            rvrl["rvrl/entropy/post_drop_lift"] = (
                _pm - float(agg.get("ent_tok_sum_explore", 0.0)) / _ne
            )
        # Entropy histograms, which resolve the right tail that a mean cannot.
        for _tag in ("explore", "anchor"):
            _h = agg.get(f"ent_hist_{_tag}")
            if isinstance(_h, list) and sum(_h) > 0:
                _t = float(sum(_h))
                for _b, _c in enumerate(_h):
                    rvrl[f"rvrl/entropy/hist_{_tag}_{_b}"] = _c / _t
                # Share of tokens above 1 nat.
                _hi = sum(_h[8:])
                rvrl[f"rvrl/entropy/forking_frac_{_tag}"] = _hi / _t
        metrics.update(rvrl)

    def _dump_exploration_drops(self, batch, metrics: dict) -> None:
        """Persist per-drop exploration records as JSONL.

        Requires ``exploration.record_details=true``, so the logits processor
        fills the side-channel, and a non-empty ``drop_dump_dir``.

        Each row is one dropped token joined to the token the sampler chose in
        its place: the logits processor knows the pre-drop distribution but not
        what got sampled, and the response knows what got sampled but not what
        was removed, so the join can only be made here.

        vLLM computes returned log-probs from the masked logits, so the value
        the rollout carries is renormalised over the surviving vocabulary rather
        than the original policy's log-prob. ``topn_*`` from the LP is the
        pre-drop distribution, so when the sampled token is inside the top-N its
        original log-prob is recoverable; ``replacement_logp_orig`` is that
        value.
        """
        import json
        import os

        exp_cfg = getattr(self.config.actor_rollout_ref.rollout, "exploration", None)
        if exp_cfg is None:
            return
        dump_dir = str(getattr(exp_cfg, "drop_dump_dir", "") or "")
        if not dump_dir:
            return
        ntb = getattr(batch, "non_tensor_batch", None) or {}
        details = ntb.get("exploration_drop_details")
        if details is None:
            return

        responses = batch.batch["responses"]
        resp_len = responses.size(1)
        max_rows = int(getattr(exp_cfg, "drop_dump_max_rows", 2000))

        def _col(name):
            v = ntb.get(name)
            return v if v is not None else [None] * len(details)

        uids = _col("uid")
        explored = _col("__two_wave_explored__")
        strat_on = _col("__tw_strategy_on__")
        rollout_n = _col("__rollout_n__")

        pv = int(getattr(self, "current_param_version", 0))
        exp_name = str(self.config.trainer.experiment_name)
        out_dir = os.path.join(dump_dir, exp_name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"drops_{pv}.jsonl")

        n_rows = 0
        n_drops_total = 0
        recovered = 0
        logp_gap_sum = 0.0
        with open(path, "w") as fh:
            for i, det in enumerate(details):
                if not det or not det.get("pos"):
                    continue
                positions = det["pos"]
                n_drops_total += len(positions)
                topn_tok = det.get("topn_tok") or []
                topn_logp = det.get("topn_logp") or []
                for j, pos in enumerate(positions):
                    if n_rows >= max_rows:
                        break
                    pos = int(pos)
                    if not (0 <= pos < resp_len):
                        continue
                    repl = int(responses[i, pos].item())
                    dropped_tok = int(det["tok"][j])
                    # Original log-prob of what was actually sampled, when the
                    # LP's pre-drop top-N happens to contain it.
                    repl_logp_orig = None
                    if j < len(topn_tok):
                        cand = topn_tok[j]
                        if repl in cand:
                            repl_logp_orig = float(topn_logp[j][cand.index(repl)])
                            recovered += 1
                            logp_gap_sum += float(det["logp"][j]) - repl_logp_orig
                    rec = {
                        "pv": pv,
                        "rollout_idx": i,
                        "uid": str(uids[i]) if uids[i] is not None else None,
                        "rollout_n": rollout_n[i],
                        "explored": explored[i],
                        "strategy_on": strat_on[i],
                        "pos": pos,
                        "dropped_token_id": dropped_tok,
                        "dropped_token": self._safe_decode(dropped_tok),
                        "dropped_logp": float(det["logp"][j]),
                        "runner_up_logp": float(det["ru_logp"][j]),
                        "mass_removed": float(det["mass"][j]),
                        "entropy": float(det["ent"][j]),
                        "replacement_token_id": repl,
                        "replacement_token": self._safe_decode(repl),
                        "replacement_logp_orig": repl_logp_orig,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rows += 1
                if n_rows >= max_rows:
                    break

        # Coverage of this dump, so a truncated file is not read as a complete
        # one.
        metrics["rvrl/drop/dump_rows"] = n_rows
        metrics["rvrl/drop/dump_truncated"] = float(n_rows >= max_rows)
        if n_drops_total > 0:
            metrics["rvrl/drop/dump_coverage"] = n_rows / float(n_drops_total)
        if recovered > 0:
            # Mean [log p_orig(dropped) - log p_orig(sampled)], measured on the
            # original policy rather than the renormalised one.
            metrics["rvrl/drop/substitution_logp_cost_mean"] = logp_gap_sum / recovered
            metrics["rvrl/drop/replacement_in_topn_frac"] = recovered / float(max(n_rows, 1))

    def _rvrl_counters_from_trace(self) -> dict:
        """Recover LP counters from the worker-written trace files.

        The logits processor periodically appends a ``kind="counters"`` snapshot
        to ``drop_trace_dir``, exempt from the drop-row cap precisely so a capped
        trace cannot also blind the counters. Those snapshots are CUMULATIVE per
        worker process, so the per-interval value is a diff against the previous
        read; the first call therefore contributes nothing and returns ``{}``
        rather than reporting a whole run's totals as one step.

        Only the tail of each file is read: they grow large and are appended to
        by a live writer.
        """
        import glob as _glob
        import json as _json

        try:
            exp_cfg = self.config.actor_rollout_ref.rollout.exploration
        except Exception:
            return {}
        trace_dir = str(getattr(exp_cfg, "drop_trace_dir", "") or "")
        if not trace_dir or not os.path.isdir(trace_dir):
            return {}

        from verl.workers.rollout.exploration import merge_lp_counters

        cur: dict = {}
        for path in sorted(_glob.glob(os.path.join(trace_dir, "drops_pid*.jsonl"))):
            try:
                with open(path, "rb") as fh:
                    fh.seek(0, os.SEEK_END)
                    size = fh.tell()
                    fh.seek(max(0, size - 512 * 1024))
                    tail = fh.read().decode("utf-8", "replace")
                snap = None
                for line in reversed(tail.splitlines()):
                    if '"kind": "counters"' not in line and '"kind":"counters"' not in line:
                        continue
                    try:
                        cand = _json.loads(line)
                    except Exception:
                        continue        # a torn last line while the writer is mid-append
                    snap = cand
                    break
                if snap:
                    snap.pop("kind", None)
                    snap.pop("pid", None)
                    snap.pop("apply_n", None)
                    merge_lp_counters(cur, snap)
            except Exception:
                continue
        if not cur:
            return {}

        prev = getattr(self, "_rvrl_prev_counters", None)
        self._rvrl_prev_counters = {
            k: (list(v) if isinstance(v, list) else v) for k, v in cur.items()
        }
        if prev is None:
            return {}

        delta: dict = {}
        for k, v in cur.items():
            p = prev.get(k)
            if isinstance(v, list):
                p = p if isinstance(p, list) else []
                delta[k] = [
                    v[i] - (p[i] if i < len(p) else 0) for i in range(len(v))
                ]
            else:
                try:
                    d = v - (p or 0)
                except TypeError:
                    continue
                # A worker restart resets its counters, which would make the
                # diff negative; report the raw value instead of a negative one.
                delta[k] = d if d >= 0 else v
        return delta

    def _safe_decode(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([int(token_id)])
        except Exception:
            return ""

    def _collect_grpo_variance_metrics(self, batch, metrics: dict) -> None:
        """How much of the batch produces a GRPO gradient.

        The optimiser consumes ``adv_i = (r_i - mean(r_group)) / std(r_group)``,
        so a group whose rollouts all score the same contributes nothing.
        ``rvrl/grpo/zero_adv_group_frac`` measures that share of groups directly.

        This block runs outside the ``exploration.enable`` gate, so a baseline
        run supplies the control value of every metric here. It needs only
        ``advantages``, ``response_mask`` and ``uid``, which a plain-GRPO run
        already has.

        Advantage per row is the mean over that row's live tokens. Under GRPO
        the advantage is constant across a sequence, so this is exact; under a
        token-level estimator it is a per-row summary.

        Attribution uses ``__tw_perturbed__``, the per-row flag stamped by the
        two-wave dispatcher, and falls back to the ``rollout_n < k_explore``
        bucket only on the single-wave explore path where that bucket is
        correct. Two shares are reported: ``adv_mass_perturbed_share`` is per
        rollout, while ``adv_token_mass_perturbed_share`` weights by response
        length, which is what the optimiser integrates over. Either can be
        compared against ``perturbed_row_frac``.
        """
        import numpy as np

        if not hasattr(batch, "batch") or batch.batch is None:
            return
        keys = set(batch.batch.keys())
        if "advantages" not in keys or "response_mask" not in keys:
            return

        rm = batch.batch["response_mask"]
        tok = rm.sum(dim=-1).float()
        row_adv = (
            (batch.batch["advantages"] * rm).sum(dim=-1) / tok.clamp(min=1.0)
        ).detach().float().cpu().numpy()
        tok = tok.detach().float().cpu().numpy()
        if row_adv.size == 0:
            return

        eps = 1e-6
        abs_adv = np.abs(row_adv)
        nz = abs_adv > eps
        out = {
            # Share of rollouts that carry any gradient at all.
            "rvrl/grpo/eff_sample_frac": float(nz.mean()),
            "rvrl/grpo/adv_abs_mean": float(abs_adv.mean()),
        }

        ntb = getattr(batch, "non_tensor_batch", None) or {}
        uids = ntb.get("uid")
        if uids is not None and len(uids) == row_adv.size:
            _, inv = np.unique(np.asarray([str(x) for x in uids], dtype=object), return_inverse=True)
            n_groups = int(inv.max()) + 1
            alive = np.bincount(inv, weights=nz.astype(float), minlength=n_groups) > 0
            # Share of prompt groups that were rolled out and scored and then
            # contributed nothing to the update.
            out["rvrl/grpo/zero_adv_group_frac"] = float(1.0 - alive.mean())
            out["rvrl/grpo/n_groups"] = n_groups

        pmask = None
        if "__tw_perturbed__" in ntb:
            pmask = np.asarray([bool(x) for x in ntb["__tw_perturbed__"]], dtype=bool)
        elif "__rollout_n__" in ntb:
            # Single-wave explore path only; under two-wave the key above exists
            # and takes precedence.
            try:
                k_explore = int(self.config.actor_rollout_ref.rollout.exploration.k_explore)
                rn = np.asarray([int(x) if x is not None else -1 for x in ntb["__rollout_n__"]])
                pmask = rn < k_explore
            except Exception:
                pmask = None

        if pmask is not None and pmask.size == row_adv.size and bool(pmask.any()):
            out["rvrl/grpo/perturbed_row_frac"] = float(pmask.mean())
            total = float(abs_adv.sum())
            if total > 0.0:
                out["rvrl/grpo/adv_mass_perturbed_share"] = float(abs_adv[pmask].sum() / total)
            weighted = abs_adv * tok
            total_w = float(weighted.sum())
            if total_w > 0.0:
                out["rvrl/grpo/adv_token_mass_perturbed_share"] = float(
                    weighted[pmask].sum() / total_w
                )

        metrics.update(out)

    def _collect_rvrl_rollout_metrics(self, batch, metrics: dict, k_explore: int) -> None:
        """Slice scores/lengths by explore vs anchor rollout buckets."""
        import numpy as np

        if not hasattr(batch, "non_tensor_batch") or batch.non_tensor_batch is None:
            return
        ntb = batch.non_tensor_batch
        if "__rollout_n__" not in ntb:
            return
        try:
            rollout_n = np.asarray([int(x) if x is not None else -1 for x in ntb["__rollout_n__"]])
        except Exception:
            return
        if rollout_n.size == 0:
            return
        explore_mask = rollout_n < k_explore
        anchor_mask = rollout_n >= k_explore
        n_explore = int(explore_mask.sum())
        n_anchor = int(anchor_mask.sum())

        # Scores: prefer summed token_level_scores (per-seq scalar reward).
        # Fallback chain handles different verl trainer pipelines that store
        # the per-sample scalar reward under different keys.
        score_arr = None
        try:
            if "token_level_scores" in batch.batch.keys():
                score_arr = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
            elif "rm_scores" in batch.batch.keys():
                score_arr = batch.batch["rm_scores"].sum(dim=-1).detach().cpu().numpy()
            elif "scores" in batch.batch.keys():
                _t = batch.batch["scores"]
                if _t.dim() > 1:
                    _t = _t.sum(dim=-1)
                score_arr = _t.detach().cpu().numpy()
            elif "score" in ntb:
                score_arr = np.asarray([float(x) if x is not None else 0.0 for x in ntb["score"]])
            elif "acc" in ntb:
                score_arr = np.asarray([float(x) if x is not None else 0.0 for x in ntb["acc"]])
        except Exception:
            score_arr = None

        rvrl = {
            "rvrl/n_explore_in_batch": n_explore,
            "rvrl/n_anchor_in_batch": n_anchor,
        }
        if score_arr is not None and score_arr.size == rollout_n.size:
            if n_explore > 0:
                rvrl["rvrl/score_explore_mean"] = float(score_arr[explore_mask].mean())
                rvrl["rvrl/score_explore_std"] = float(score_arr[explore_mask].std())
            if n_anchor > 0:
                rvrl["rvrl/score_anchor_mean"] = float(score_arr[anchor_mask].mean())
                rvrl["rvrl/score_anchor_std"] = float(score_arr[anchor_mask].std())
            if n_explore > 0 and n_anchor > 0:
                rvrl["rvrl/score_diff"] = (
                    rvrl["rvrl/score_explore_mean"] - rvrl["rvrl/score_anchor_mean"]
                )
            # best_of_group_*_share: within each per-prompt group, was the
            # top-reward rollout a perturbed one?
            #
            # The group key is `uid`, stamped per prompt in
            # FullyAsyncRollouter._process_single_sample_streaming, not
            # `__sample_index__`: the curated parquets never write
            # extra_info.index, so sample_index is a constant 0 for every row
            # and would fold the whole training batch into one group.
            group_ids = None
            if "uid" in ntb:
                group_ids = np.asarray([str(x) for x in ntb["uid"]], dtype=object)
            elif "__sample_index__" in ntb:
                try:
                    group_ids = np.asarray(
                        [int(x) if x is not None else -1 for x in ntb["__sample_index__"]]
                    )
                except Exception:
                    group_ids = None
            if group_ids is not None and group_ids.size == score_arr.size:
                # The rollout_n bucket, plus the exact per-row perturbation flag
                # under its own key whenever the dispatcher stamped it.
                buckets = [("rvrl/best_of_group_explore_share", explore_mask)]
                if "__tw_perturbed__" in ntb:
                    try:
                        pm = np.asarray([bool(x) for x in ntb["__tw_perturbed__"]], dtype=bool)
                        if pm.size == score_arr.size:
                            buckets.append(("rvrl/two_wave/best_of_group_perturbed_share", pm))
                    except Exception:
                        pass
                unique_groups = np.unique(group_ids)
                for _key, _mask in buckets:
                    hits = 0
                    counted = 0
                    for g in unique_groups:
                        gmask = group_ids == g
                        if gmask.sum() < 2:
                            continue
                        gscore = score_arr[gmask]
                        # any tie? pick all argmaxes
                        argmax_mask = gscore == gscore.max()
                        # if any bucket rollout is among the argmaxes, count as hit
                        if (argmax_mask & _mask[gmask]).any():
                            hits += 1
                        counted += 1
                    if counted > 0:
                        rvrl[_key] = hits / counted

        # Response length per bucket from response_mask (excludes padding/tool tokens).
        try:
            if "response_mask" in batch.batch.keys():
                resp_len = batch.batch["response_mask"].sum(dim=-1).detach().cpu().numpy()
                if resp_len.size == rollout_n.size:
                    if n_explore > 0:
                        rvrl["rvrl/response_length_explore_mean"] = float(resp_len[explore_mask].mean())
                    if n_anchor > 0:
                        rvrl["rvrl/response_length_anchor_mean"] = float(resp_len[anchor_mask].mean())
        except Exception:
            pass

        # format pass rate (vero reward wrapper field). reward_extra_info is
        # merged into non_tensor_batch by _fit_compute_advantage; the key is
        # 'format' (a 0/1 float per sample).
        if "format" in ntb:
            try:
                fmt = np.asarray([float(x) if x is not None else 0.0 for x in ntb["format"]])
                if fmt.size == rollout_n.size:
                    if n_explore > 0:
                        rvrl["rvrl/format_pass_explore"] = float(fmt[explore_mask].mean())
                    if n_anchor > 0:
                        rvrl["rvrl/format_pass_anchor"] = float(fmt[anchor_mask].mean())
            except Exception:
                pass

        # Two-wave variance-targeted exploration metrics. The keys are present
        # only when two_wave_enable fired in agent_loop; other runs skip this
        # block. The explore flag is constant within a prompt and every prompt
        # contributes the same n rollouts, so a mean over rollouts equals the
        # per-prompt fraction.
        if "__two_wave_explored__" in ntb:
            try:
                flags = [bool(x) for x in ntb["__two_wave_explored__"] if x is not None]
                if flags:
                    rvrl["rvrl/two_wave_explore_frac"] = float(np.mean(flags))
            except Exception:
                pass
        if "__wave1_var__" in ntb:
            try:
                wave1_vars = np.asarray(
                    [float(x) for x in ntb["__wave1_var__"] if x is not None], dtype=float
                )
                if wave1_vars.size > 0:
                    rvrl["rvrl/wave1_var_mean"] = float(wave1_vars.mean())
                    rvrl["rvrl/wave1_var_p50"] = float(np.median(wave1_vars))
            except Exception:
                pass

        # Two-wave group-composition and exploration-effect diagnostics (the
        # ``__tw_*`` arrays from _two_wave_group_stats). Values are constant
        # within a prompt and every prompt contributes equal n rollouts, so an
        # output-level mean equals a per-group mean and masked output-level
        # ratios equal per-group conditional shares. These keys are labelled by
        # wave: wave-1 is the normal anchor half, wave-2 the conditionally
        # explored half.
        if "__tw_all_wrong__" in ntb:
            try:
                def _tw_bool(key):
                    return np.asarray([bool(x) for x in ntb[key]], dtype=bool) if key in ntb else None

                def _tw_float(key):
                    return np.asarray([float(x) for x in ntb[key]], dtype=float) if key in ntb else None

                ac = _tw_bool("__tw_all_correct__")
                aw = _tw_bool("__tw_all_wrong__")
                mx = _tw_bool("__tw_mixed__")
                trig = _tw_bool("__tw_triggered__")
                if ac is not None and ac.size > 0:
                    # Group composition over ALL prompts (answers "how much do
                    # all-correct vs all-wrong vs mixed groups account for").
                    rvrl["rvrl/two_wave/all_correct_frac"] = float(ac.mean())
                    rvrl["rvrl/two_wave/all_wrong_frac"] = float(aw.mean())
                    rvrl["rvrl/two_wave/mixed_frac"] = float(mx.mean())
                    # Duty-cycle telemetry, present only when the dispatcher
                    # stamped the strategy flag.
                    on = _tw_bool("__tw_strategy_on__")
                    if on is not None and on.size > 0:
                        # Sits at ~1/two_wave_toggle_period when the toggle is
                        # on, and at 1 when it is off.
                        rvrl["rvrl/two_wave/strategy_on_frac"] = float(on.mean())
                        n_on = int(on.sum())
                        if n_on > 0 and trig is not None:
                            # Degeneracy rate among strategy-on prompts.
                            rvrl["rvrl/two_wave/explored_frac_when_on"] = float((trig & on).sum() / n_on)
                        # Degenerate prompts the duty cycle skipped: 0.0 when
                        # the toggle is off.
                        deg = ac | aw
                        n_deg = int(deg.sum())
                        if n_deg > 0:
                            rvrl["rvrl/two_wave/suppressed_by_toggle_share"] = float((deg & ~on).sum() / n_deg)
                    if trig is not None:
                        n_trig = int(trig.sum())
                        if n_trig > 0:
                            # Composition of the TRIGGERED (explored) subset.
                            rvrl["rvrl/two_wave/triggered_all_wrong_share"] = float((aw & trig).sum() / n_trig)
                            rvrl["rvrl/two_wave/triggered_all_correct_share"] = float((ac & trig).sum() / n_trig)
                            vm = _tw_bool("__tw_var_manufactured__")
                            if vm is not None:
                                # Of the triggered groups, how often exploration
                                # actually broke degeneracy (revived the gradient).
                                rvrl["rvrl/two_wave/variance_manufactured_frac"] = float((vm & trig).sum() / n_trig)
                        # Conditional crack/break shares.
                        naw_trig = int((aw & trig).sum())
                        if naw_trig > 0:
                            cracked = _tw_bool("__tw_cracked_all_wrong__")
                            if cracked is not None:
                                rvrl["rvrl/two_wave/cracked_all_wrong_share"] = float(
                                    (cracked & aw & trig).sum() / naw_trig
                                )
                        nac_trig = int((ac & trig).sum())
                        if nac_trig > 0:
                            broke = _tw_bool("__tw_broke_all_correct__")
                            if broke is not None:
                                rvrl["rvrl/two_wave/broke_all_correct_share"] = float(
                                    (broke & ac & trig).sum() / nac_trig
                                )
                # Accuracy means + correctly-labeled exploration effect.
                w1a = _tw_float("__tw_w1_acc__")
                w2a = _tw_float("__tw_w2_acc__")
                gva = _tw_float("__tw_group_var_after__")
                if w1a is not None and w1a.size > 0:
                    rvrl["rvrl/two_wave/wave1_acc_mean"] = float(w1a.mean())
                if w2a is not None and w2a.size > 0:
                    rvrl["rvrl/two_wave/wave2_acc_mean"] = float(w2a.mean())
                if (
                    trig is not None and bool(trig.any())
                    and w1a is not None and w2a is not None
                    and w1a.size == trig.size and w2a.size == trig.size
                ):
                    # wave-2 (explored) minus wave-1 (normal) accuracy on
                    # triggered groups.
                    rvrl["rvrl/two_wave/explore_acc_delta"] = float(w2a[trig].mean() - w1a[trig].mean())
                if gva is not None and trig is not None and bool(trig.any()) and gva.size == trig.size:
                    rvrl["rvrl/two_wave/group_var_after_triggered_mean"] = float(gva[trig].mean())

                # ---- variance structure ------------------------------------
                # Every metric below is emitted twice: once on the triggered
                # groups and once on the untriggered ones. Triggered and
                # untriggered groups differ in composition, since triggered ones
                # are degenerate by construction, so the pair is a
                # difference-in-population rather than a randomised control; the
                # randomised comparison is the strategy-off arm of a duty-cycled
                # run, or a baseline run.
                if trig is not None and trig.size > 0:
                    ctl = ~trig
                    n_trig_rows, n_ctl_rows = int(trig.sum()), int(ctl.sum())

                    def _split(arr, stem, valid=None):
                        """Emit mean(arr) on the triggered rows and on the control rows."""
                        if arr is None or arr.size != trig.size:
                            return
                        ok = np.ones(arr.size, dtype=bool) if valid is None else valid
                        for _m, _sfx in ((trig & ok, "triggered"), (ctl & ok, "untriggered")):
                            if int(_m.sum()) > 0:
                                rvrl[f"rvrl/two_wave/{stem}_{_sfx}"] = float(arr[_m].mean())

                    # 1. ANOVA between-wave share: the fraction of the group's
                    #    variance carried by the contrast between the two wave
                    #    means rather than by variance within a wave.
                    _split(_tw_float("__tw_between_share__"), "between_share")

                    # 2. Excess variance over the "wave-2 is distributed like
                    #    wave-1" null. On the untriggered arm wave-2 really is
                    #    unperturbed, so var_gain_untriggered is the calibration
                    #    check for the null.
                    vnul = _tw_float("__tw_var_null__")
                    if gva is not None and vnul is not None and gva.size == trig.size == vnul.size:
                        _gain = gva - vnul
                        _split(_gain, "var_gain")

                    # 3. Behavioural divergence, in tokens. lcp_w1_mean is the
                    #    scale: how far two unperturbed rollouts of the same
                    #    prompt agree before sampling noise separates them.
                    #    lcp_delta = lcp_w2 - lcp_w1, so a negative value means
                    #    wave-2 leaves the anchor earlier than sampling alone
                    #    would.
                    l1, l2 = _tw_float("__tw_lcp_w1__"), _tw_float("__tw_lcp_w2__")
                    if l1 is not None and l2 is not None and l1.size == l2.size == trig.size:
                        okl = (l1 >= 0.0) & (l2 >= 0.0)  # -1.0 = not computable
                        if bool(okl.any()):
                            rvrl["rvrl/two_wave/lcp_w1_mean"] = float(l1[okl].mean())
                        _split(l2 - l1, "lcp_delta", valid=okl)

                    # 4. Coverage. pass@n over the whole group against
                    #    pass@n_anchor over wave-1 alone; coverage_gain is the
                    #    difference.
                    pa, pa1 = _tw_bool("__tw_pass_any__"), _tw_bool("__tw_pass_any_w1__")
                    if pa is not None and pa1 is not None and pa.size == pa1.size == trig.size:
                        rvrl["rvrl/two_wave/pass_at_n_mean"] = float(pa.mean())
                        rvrl["rvrl/two_wave/pass_at_n_anchor_mean"] = float(pa1.mean())
                        rvrl["rvrl/two_wave/coverage_gain"] = float(pa.mean() - pa1.mean())

                    # 5. Solution discovery, computed on every group, so the
                    #    untriggered rate is the rate at which an ordinary
                    #    unperturbed second wave finds something the first wave
                    #    missed.
                    sn = _tw_bool("__tw_w2_solves_new__")
                    _split(None if sn is None else sn.astype(float), "solves_new_frac")
                    rvrl["rvrl/two_wave/n_triggered_rows"] = n_trig_rows
                    rvrl["rvrl/two_wave/n_untriggered_rows"] = n_ctl_rows
            except Exception:
                pass

        metrics.update(rvrl)
