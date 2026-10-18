# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Single-node FSDP OPD entry point; config is supplied by the shared launcher."""

import sys

import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role


def create_rl_dataset(data_paths, data_config, tokenizer, processor=None, **kwargs):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    return RLHFDataset(
        data_paths,
        tokenizer,
        data_config,
        processor=processor,
        max_samples=data_config.get("train_max_samples", -1),
    )


def create_rl_sampler(data_config, dataset):
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        generator = torch.Generator().manual_seed(data_config.seed)
        return RandomSampler(dataset, generator=generator)
    return SequentialSampler(dataset)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from transformers import AutoTokenizer

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils.config import validate_config
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker

        tokenizer = AutoTokenizer.from_pretrained(
            config.actor_rollout_ref.model.path,
            trust_remote_code=config.data.trust_remote_code,
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.reward_model.model.path,
            trust_remote_code=config.data.trust_remote_code,
        )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("OPD requires matching student/teacher token IDs")
        del teacher_tokenizer
        roles = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
        mapping = {Role.ActorRollout: "opd"}
        if config.reward_model.enable:
            roles[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = "opd"
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            roles[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = "opd"
        validate_config(
            config, use_reference_policy=Role.RefPolicy in roles, use_critic=False
        )
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=roles,
            resource_pool_manager=ResourcePoolManager(
                {"opd": [config.trainer.n_gpus_per_node]}, mapping
            ),
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    full_config = OmegaConf.load(sys.argv[1])
    config = OmegaConf.create(
        OmegaConf.to_container(full_config.training, resolve=True)
    )
    if full_config.run.teacher_chat_template:
        from pathlib import Path
        from modules.rethinking.model_view import apply_teacher_metadata

        view = Path(config.ray_kwargs.ray_init._temp_dir) / "student"
        apply_teacher_metadata(
            config.actor_rollout_ref.model.path,
            config.reward_model.model.path,
            view,
            link_weights=True,
        )
        config.actor_rollout_ref.model.path = str(view)
    kwargs = OmegaConf.to_container(config.ray_kwargs.ray_init, resolve=True)
    kwargs["runtime_env"] = OmegaConf.to_container(
        OmegaConf.merge(get_ppo_ray_runtime_env(), kwargs.pop("runtime_env", {})),
        resolve=True,
    )
    # Never connect to or stop a shared Ray cluster.
    ray.init(address="local", **kwargs)
    try:
        ray.get(TaskRunner.remote().run.remote(config))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
