# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from datetime import datetime

def add_clean_rl_args(parser: argparse.ArgumentParser):
    arg_group = parser.add_argument_group(
        "clean_rl", description="Arguments for CleanRL agent."
    )
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    training_mode = arg_group.add_mutually_exclusive_group()
    training_mode.add_argument(
        "--resume", action="store_true", default=False, help="从指定检查点继续训练。"
    )
    training_mode.add_argument(
        "--finetune", action="store_true", default=False,
        help="完整加载智能体，使用新优化器、独立迭代预算和成熟约束课程微调。",
    )
    arg_group.add_argument("--allow_partial_resume", action="store_true", help="允许旧模型缺少优化器等状态的近似续训。")
    arg_group.add_argument("--resume_constraint_progress", type=float, default=1.0, help="旧模型约束课程进度，范围 0 到 1。")
    arg_group.add_argument("--stop_after_iterations", type=int, default=0, help="本进程完成指定轮数后停止；不改变总训练预算和学习率调度。")
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    arg_group.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices={"wandb", "tensorboard"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=f"clr-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        help="Name of the logging project when using wandb",
    )


def parse_clean_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    cleanrl_cfg = load_cfg_from_registry(task_name, "clean_rl_cfg_entry_point")
    cleanrl_cfg = update_clean_rl_cfg(cleanrl_cfg, args_cli)
    return cleanrl_cfg


def update_clean_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    agent_cfg.finetune = getattr(args_cli, "finetune", False)
    if agent_cfg.resume and agent_cfg.finetune:
        raise ValueError("--resume 与 --finetune 不能同时启用。")
    agent_cfg.allow_partial_resume = args_cli.allow_partial_resume
    agent_cfg.resume_constraint_progress = args_cli.resume_constraint_progress
    agent_cfg.stop_after_iterations = args_cli.stop_after_iterations
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb
    if agent_cfg.logger in {"wandb"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name

    return agent_cfg
