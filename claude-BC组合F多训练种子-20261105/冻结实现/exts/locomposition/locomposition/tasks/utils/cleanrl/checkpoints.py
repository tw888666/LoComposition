"""保存和加载优化器状态；恢复时物理仿真仍从新回合开始。"""
from pathlib import Path
import hashlib
import os
import re
import torch


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_finetune_checkpoint(path, agent, *, device):
    """完整加载智能体权重与归一化统计，不恢复优化器或环境状态。"""
    path = Path(path).expanduser().resolve()
    state = torch.load(path, map_location=device, weights_only=True)
    full_state = isinstance(state, dict) and state.get("format_version") == 1
    model_state = state["model_state_dict"] if full_state else state
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("微调检查点必须包含完整智能体状态字典。")
    if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
               for value in model_state.values()):
        raise ValueError("微调检查点包含非张量或非有限数值。")
    agent.load_state_dict(model_state, strict=True)
    loaded_state = agent.state_dict()
    verified_keys = []
    for name, source in model_state.items():
        loaded = loaded_state[name]
        if source.dtype != loaded.dtype or not torch.equal(source, loaded):
            raise ValueError(f"微调检查点逐键一致性校验失败：{name}")
        verified_keys.append(name)
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    source_iteration = int(state["iteration"]) if full_state else (int(match.group(1)) if match else None)
    return {
        "来源检查点": str(path),
        "来源文件SHA256": digest.hexdigest(),
        "来源迭代": source_iteration,
        "来源格式": "完整训练状态中提取智能体" if full_state else "完整智能体状态字典",
        "逐键完全一致": True,
        "已核验状态键": verified_keys,
        "优化器已恢复": False,
        "地形与约束归一化已恢复": False,
    }


def initialize_finetune_environment(env):
    """新环境从第零步开始，CaT（约束终止）课程直接使用成熟阶段。"""
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.curriculum_manager.constraints_curriculum = 1.0


def load_training_checkpoint(path, agent, optimizer, *, device, total_iterations,
                             allow_partial=False, constraint_progress=1.0):
    path = Path(path).expanduser().resolve()
    if not 0 <= constraint_progress <= 1:
        raise ValueError("约束课程进度必须位于 [0, 1]。")
    state = torch.load(path, map_location=device, weights_only=True)
    full_state = isinstance(state, dict) and state.get("format_version") == 1
    if full_state:
        if state["total_iterations"] != total_iterations:
            raise ValueError("恢复时总迭代预算必须与检查点一致，以保持学习率调度。")
        model_state = state["model_state_dict"]
        iteration = int(state["iteration"])
    else:
        if not allow_partial:
            raise ValueError("旧文件没有优化器及环境课程状态；需明确指定 --allow_partial_resume。")
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match is None:
            raise ValueError("旧模型文件名必须为 model_迭代数.pt，以确定课程和学习率进度。")
        iteration = int(match.group(1))
        model_state = state
        state = {"iteration": iteration, "constraints_curriculum": constraint_progress}
    if not 0 <= iteration < total_iterations:
        raise ValueError("检查点迭代数必须小于总训练预算。")
    if not all(torch.isfinite(value).all() for value in model_state.values()):
        raise ValueError("模型包含非有限数值。")
    agent.load_state_dict(model_state)
    if full_state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    state["source_path"] = str(path)
    state["partial_resume"] = not full_state
    return state


def restore_environment_progress(env, state, num_steps):
    env.common_step_counter = int(state.get("common_step_counter", state["iteration"] * num_steps))
    env._sim_step_counter = env.common_step_counter * env.cfg.decimation
    env.curriculum_manager.constraints_curriculum = float(state["constraints_curriculum"])
    if "constraint_running_maxes" in state:
        env.constraint_manager.cat.running_maxes = {
            name: value.to(env.device) for name, value in state["constraint_running_maxes"].items()
        }
    if "terrain_levels" in state:
        terrain = env.scene.terrain
        levels = state["terrain_levels"].to(env.device)
        types = state["terrain_types"].to(env.device)
        if levels.shape != terrain.terrain_levels.shape or types.shape != terrain.terrain_types.shape:
            raise ValueError("恢复地形分配需要相同的并行环境数量。")
        terrain.terrain_levels.copy_(levels)
        terrain.terrain_types.copy_(types)
        terrain.env_origins.copy_(terrain.terrain_origins[levels, types])


def save_training_checkpoint(path, agent, optimizer, env, iteration, total_iterations):
    state = {
        "format_version": 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "total_iterations": total_iterations,
        "common_step_counter": env.common_step_counter,
        "constraints_curriculum": float(env.curriculum_manager.constraints_curriculum),
        "constraint_running_maxes": env.constraint_manager.cat.running_maxes,
        "resume_semantics": "恢复模型、优化器、课程和地形分配；物理仿真及随机数状态重新初始化。",
    }
    terrain = env.scene.terrain
    if getattr(terrain, "terrain_levels", None) is not None:
        state["terrain_levels"] = terrain.terrain_levels
        state["terrain_types"] = terrain.terrain_types
    atomic_save(state, path)
