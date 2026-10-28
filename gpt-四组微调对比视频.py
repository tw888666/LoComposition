"""回放五组正式评估的真实轨迹，生成平地与楼梯同步视频，不重新运行物理或策略。

示例：.venv-v100/bin/python gpt-四组微调对比视频.py --root gpt-四组微调实验-20261026
全部五组、两个场景均完成后才能生成正式视频；--preview 也执行完整性核验。
--internal-preview-run 仅允许历史轨迹单帧验证，图片显著标注“非四组实验结果”。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "gpt-四组微调实验-20261026"
DEFAULT_MODEL = ROOT / "gpt-复现记录/gpt-关闭高程图对照-20261022/机器人模型"
GROUPS = ("A0", "A", "B", "C", "D")
SCENES = ("平地1.0", "楼梯8.5厘米")
LABELS = {
    "A0": ("三万次基准", "原始训练模型"),
    "A": ("原权重微调", "功率权重 0.008 · 混合系数 0"),
    "B": ("增大功率惩罚", "功率权重 0.012 · 混合系数 0"),
    "C": ("混合能耗微调", "总权重 0.008 · 混合系数 0.1"),
    "D": ("混合能耗微调", "总权重 0.008 · 混合系数 0.3"),
}
WIDTH, HEIGHT, FPS, SECONDS = 640, 360, 25, 10
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Clip:
    group: str
    scene: str
    run: Path
    source: Path
    sim_source: Path
    cfg: dict
    positions: np.ndarray
    angles: np.ndarray
    joints: np.ndarray
    feet: np.ndarray
    velocities: np.ndarray
    joint_names: np.ndarray
    dt: float
    terminated: bool
    internal: bool = False

    @property
    def duration(self):
        return len(self.positions) * self.dt


def choose_run(root, group, scene, checkpoint):
    candidates = sorted((root / "评估" / group).glob(f"检查点*/种子46/{scene}"))
    if checkpoint is not None:
        candidates = [path for path in candidates if path.parent.parent.name == f"检查点{checkpoint}"]
    if len(candidates) != 1:
        raise ValueError(f"{group}/{scene} 需要唯一的检查点目录，找到 {len(candidates)} 个：{candidates}")
    return candidates[0]


def load_clip(run, group, scene, *, internal=False):
    cfg = read_json(run / "gpt-评估条件.json")
    seed = cfg.get("种子", cfg.get("评估种子"))
    if seed != 46:
        raise ValueError(f"{run} 的评估种子不是 46：{seed}")
    dt = float(cfg["控制周期秒"])
    if not np.isclose(dt, 0.02):
        raise ValueError(f"{run} 控制周期不是 0.02 秒，无法按既定帧率同步")
    sim_paths = sorted(run.glob("eval_*/plots/sim_data.npz"))
    if len(sim_paths) != 1:
        raise ValueError(f"{run} 必须有且仅有一份原始仿真轨迹")
    sim_source = sim_paths[0]
    with np.load(sim_source, allow_pickle=False) as sim:
        names = sim["joint_names"].copy()
        if internal:
            result = read_json(run / "gpt-场景结果.json")
            end = int(result["首次重置前记录步数"])
            positions = sim["base_position_array"][:end].copy()
            angles = sim["base_orientation_array"][:end].copy()
            joints = sim["joint_positions_array"][:end].copy()
            feet = sim["foot_positions_world_frame_array"][:end].copy()
            velocities = sim["base_linear_velocity_body_array"][:end].copy()
            source, terminated = sim_source, bool(result["提前终止"])
    if not internal:
        completed = read_json(run / "gpt-评估完成.json")
        if not (completed.get("状态") == "完成" and completed.get("正式评估") is True
                and completed.get("归一化冻结核验") is True):
            raise ValueError(f"{run} 尚未通过正式评估与归一化冻结核验")
        source = run / "gpt-控制步真实数据.npz"
        with np.load(source, allow_pickle=False) as data:
            stop = np.asarray(data["终止"], dtype=bool).reshape(-1) | np.asarray(data["超时"], dtype=bool).reshape(-1)
            stops = np.flatnonzero(stop)
            end = int(stops[0]) + 1 if len(stops) else len(stop)
            # 这里的每一帧均在自动重置前采集，故首个终态可以安全保留。
            positions = data["机身位置"][:end].copy()
            angles = data["机身姿态偏航俯仰横滚"][:end].copy()
            joints = data["关节位置"][:end].copy()
            feet = data["足端世界坐标"][:end].copy()
            velocities = data["机身线速度机体系"][:end].copy()
            terminated = bool(data["终止"][end - 1])
    if end < 1 or len(names) != 12:
        raise ValueError(f"{run} 没有有效首回合轨迹或关节名称数量错误")
    for key, values, shape in (("机身位置", positions, (end, 3)), ("姿态", angles, (end, 3)),
                               ("关节位置", joints, (end, 12)), ("足端", feet, (end, 4, 3)),
                               ("机身速度", velocities, (end, 3))):
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{run} 的 {key} 形状或数值无效：{values.shape}")
    if not (run / "gpt-实际地形.npz").is_file():
        raise ValueError(f"{run} 缺少实际地形，禁止使用替代楼梯")
    return Clip(group, scene, run, source, sim_source, cfg, positions, angles, joints,
                feet, velocities, names, dt, terminated, internal)


def load_all(root, finetune_checkpoint, baseline_checkpoint):
    clips, failures = {}, []
    for scene in SCENES:
        for group in GROUPS:
            try:
                checkpoint = baseline_checkpoint if group == "A0" else finetune_checkpoint
                run = choose_run(root, group, scene, checkpoint)
                clips[group, scene] = load_clip(run, group, scene)
            except (ValueError, KeyError, FileNotFoundError) as error:
                failures.append(f"{group}/{scene}：{error}")
    if failures:
        raise ValueError("五组正式数据尚未齐全，拒绝输出完整对比结果：\n" + "\n".join(failures))
    return clips


def require_completed_summary(root):
    summary = read_json(root / "评估" / "gpt-统一评估汇总.json")
    if summary.get("状态") != "90场全部完成且配对核验通过":
        raise ValueError("九十场正式评估尚未通过完整汇总，拒绝生成正式对比视频。")
    checks = read_json(root / "评估" / "gpt-配对核验.json")
    if not (isinstance(checks, list) and len(checks) == 18
            and all(isinstance(item, dict) and item.get("环境地形电机动力学首观测配对") == "通过"
                    for item in checks)):
        raise ValueError("必须恰有十八项全部通过的配对核验，才能生成正式对比视频。")


def load_rendering_dependencies():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("本地渲染依赖需要 Python 3.11，请使用项目 .venv-v100/bin/python 或 .venv/bin/python。")
    sys.path.insert(0, str(ROOT / ".gpt-video-deps"))
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless",
                      LIBGL_ALWAYS_SOFTWARE="1", MESA_SHADER_CACHE_DIR="/tmp/gpt-mesa-finetune",
                      LP_NUM_THREADS="4")
    global mujoco, trimesh, Rotation, Image, ImageDraw, ImageFont
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    from PIL import Image, ImageDraw, ImageFont
    if not FONT.is_file():
        raise FileNotFoundError(f"缺少中文字体：{FONT}")


class Replay:
    def __init__(self, clip, model_dir, cache):
        self.clip = clip
        cache.mkdir(parents=True, exist_ok=True)
        self.origin = clip.positions[0].copy()
        self.origin[2] = 0
        self.rotation = Rotation.from_euler("z", -clip.angles[0, 0])
        with np.load(clip.run / "gpt-实际地形.npz", allow_pickle=False) as terrain:
            vertices, faces = terrain["vertices"].copy(), terrain["faces"].copy()
        triangles = vertices[faces]
        lower = clip.positions[:, :2].min(axis=0) - 3
        upper = clip.positions[:, :2].max(axis=0) + 3
        if "楼梯" in clip.scene:
            lower, upper = np.minimum(lower, [-14, 12]), np.maximum(upper, [-6, 20])
        keep = (triangles[:, :, :2].max(axis=1) >= lower).all(axis=1) & (triangles[:, :, :2].min(axis=1) <= upper).all(axis=1)
        if not keep.any():
            raise ValueError("实际地形裁剪为空")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces[keep], process=False)
        mesh.remove_unreferenced_vertices()
        mesh.vertices = self.rotation.apply(mesh.vertices - self.origin)
        terrain_path = cache / "gpt-真实地形.obj"
        mesh.export(terrain_path)
        robot_tree = ET.parse(model_dir / "go2.xml")
        robot_tree.getroot().find("compiler").set("meshdir", str((model_dir / "assets").resolve()))
        robot_path = cache / "gpt-机器人回放.xml"
        robot_tree.write(robot_path, encoding="unicode")
        tree = ET.parse(model_dir / "scene.xml")
        root = tree.getroot()
        root.find("include").set("file", str(robot_path))
        visual = root.find("visual")
        visual.find("global").set("offwidth", str(WIDTH))
        visual.find("global").set("offheight", str(HEIGHT))
        visual.find("headlight").set("ambient", "0.4 0.4 0.4")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        quality.set("shadowsize", "1024")
        quality.set("offsamples", "4")
        for texture in root.find("asset").findall("texture"):
            if texture.get("type") == "skybox":
                texture.set("rgb1", "0.72 0.80 0.88")
                texture.set("rgb2", "0.90 0.94 0.97")
        ET.SubElement(root.find("asset"), "mesh", name="actual_terrain", file=str(terrain_path), inertia="shell")
        world = root.find("worldbody")
        world.remove(world.find("geom[@name='floor']"))
        world.find("light").set("pos", "-3 -4 8")
        world.find("light").set("dir", "0.3 0.4 -1")
        ET.SubElement(world, "geom", name="actual_terrain", type="mesh", mesh="actual_terrain",
                      rgba="0.49 0.55 0.60 1", contype="0", conaffinity="0")
        scene_path = cache / "gpt-真实轨迹场景.xml"
        tree.write(scene_path, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.state = mujoco.MjData(self.model)
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(name)) for name in clip.joint_names]
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FL", "FR", "RL", "RR")]
        if min(joint_ids + self.foot_ids) < 0:
            raise ValueError("机器人模型缺少原始轨迹对应的关节或足端")
        self.addresses = self.model.jnt_qposadr[joint_ids]
        if len(set(self.addresses.tolist())) != 12 or min(self.addresses) < 7:
            raise ValueError("关节映射错误")
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.65, 75, -20
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3] = 0
        self.fonts = {size: ImageFont.truetype(str(FONT), size) for size in (14, 17, 20, 23)}
        self.errors = []
        self.last_index, self.last_frame = None, None

    def frame(self, output_index):
        # 每个输出帧显示对应40毫秒区间末端，最后一帧可包含第500步真实终态。
        requested = output_index * 2 + 1
        index = min(requested, len(self.clip.positions) - 1)
        frozen = requested > index
        if index != self.last_index:
            self.state.qpos[:3] = self.rotation.apply(self.clip.positions[index] - self.origin)
            yaw, pitch, roll = self.clip.angles[index]
            quaternion = (self.rotation * Rotation.from_euler("xyz", [roll, pitch, yaw])).as_quat()
            self.state.qpos[3:7] = quaternion[[3, 0, 1, 2]]
            self.state.qpos[self.addresses] = self.clip.joints[index]
            mujoco.mj_forward(self.model, self.state)
            reference = self.rotation.apply(self.clip.feet[index] - self.origin)
            errors = np.linalg.norm(self.state.geom_xpos[self.foot_ids] - reference, axis=1)
            self.errors.extend(errors.tolist())
            if errors.max() > 0.015:
                raise RuntimeError(f"{self.clip.group}/{self.clip.scene} 第 {index + 1} 步足端最大误差 {errors.max():.6f} 米")
            self.camera.lookat[:] = self.state.qpos[:3] + [0.15, 0, -0.04]
            self.renderer.update_scene(self.state, camera=self.camera, scene_option=self.options)
            self.last_frame = Image.fromarray(self.renderer.render()).copy()
            self.last_index = index
        frame = self.last_frame.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, WIDTH, 64), fill=(19, 29, 43))
        title, detail = LABELS.get(self.clip.group, ("历史三万次轨迹", "内部渲染验证"))
        draw.text((16, 8), f"{self.clip.group} · {title}", font=self.fonts[23], fill="white")
        draw.text((16, 39), detail, font=self.fonts[14], fill=(194, 213, 230))
        draw.rectangle((0, HEIGHT - 59, WIDTH, HEIGHT), fill=(19, 29, 43))
        clock = (index + 1) * self.clip.dt
        speed = self.clip.velocities[index, 0]
        draw.text((16, HEIGHT - 55), f"时间 {clock:4.2f} 秒    实际前向速度 {speed:.2f} 米/秒",
                  font=self.fonts[17], fill="white")
        note = "首次终止后定格" if frozen and self.clip.terminated else "首回合结束后定格" if frozen else "真实轨迹与真实地形 · 正常速度"
        if self.clip.internal:
            note = "内部验证 · 历史轨迹 · 非四组实验结果"
        draw.text((16, HEIGHT - 28), note, font=self.fonts[14], fill=(255, 174, 145) if frozen or self.clip.internal else (194, 213, 230))
        return frame

    def evidence(self):
        return {
            "组别": self.clip.group, "场景": self.clip.scene, "种子": 46,
            "目录": str(self.clip.run), "来源轨迹": str(self.clip.source), "来源SHA256": sha256(self.clip.source),
            "原始轨迹": str(self.clip.sim_source), "原始轨迹SHA256": sha256(self.clip.sim_source),
            "地形SHA256": sha256(self.clip.run / "gpt-实际地形.npz"),
            "首回合真实控制步": len(self.clip.positions), "首回合真实时长秒": self.clip.duration,
            "提前终止": self.clip.terminated, "足端最大误差米": max(self.errors),
            "足端平均误差米": float(np.mean(self.errors)), "仅运动学回放": True,
            "关节映射": {str(name): int(address) for name, address in zip(self.clip.joint_names, self.addresses)},
        }

    def close(self):
        self.renderer.close()


def info_panel(scene, output_index, finetune_iteration):
    frame = Image.new("RGB", (WIDTH, HEIGHT), (19, 29, 43))
    draw = ImageDraw.Draw(frame)
    font = lambda size: ImageFont.truetype(str(FONT), size)
    title = "平地 · 指令 1.0 米/秒" if scene == "平地1.0" else "楼梯 · 每级 8.5 厘米"
    draw.text((24, 22), title, font=font(26), fill="white")
    lines = ["五组同种子 46 · 同一仿真时间轴", "左上起：A0、A、B；下排：C、D",
             f"A0：三万次基准；A—D：另微调 {finetune_iteration} 次", "仅回放原评估轨迹，未重新仿真",
             "首次终止后保留真实终态并标注定格", "数值结论以多种子汇总为准"]
    for index, line in enumerate(lines):
        draw.text((24, 80 + index * 34), line, font=font(18), fill=(194, 213, 230))
    draw.text((24, 307), f"同步时刻 {(output_index + 1) / FPS:4.2f} 秒", font=font(21), fill=(116, 229, 187))
    return frame


def verify_video(path, seconds):
    metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)]))
    stream = metadata["streams"][0]
    if int(stream["nb_read_frames"]) != FPS * seconds or not np.isclose(float(metadata["format"]["duration"]), seconds):
        raise RuntimeError(f"{path} 帧数或时长不正确")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"视频": str(path), "SHA256": sha256(path), "时长秒": seconds,
            "帧数": int(stream["nb_read_frames"]), "宽": stream["width"], "高": stream["height"], "完整解码": "通过"}


def render_scene(clips, scene, out, model_dir, preview):
    replays, evidence = [], []
    encoder = None
    temporary = out / f"gpt-五组同步对比-{scene}.临时.mp4"
    target = out / f"gpt-五组同步对比-{scene}.mp4"
    started = time.monotonic()
    finetune_iteration = clips["A", scene].run.parent.parent.name.removeprefix("检查点")
    try:
        for group in GROUPS:
            replays.append(Replay(clips[group, scene], model_dir, out / "回放缓存" / scene / group))
        with (out / f"gpt-{scene}-编码.log").open("w", encoding="utf-8") as log:
            if not preview:
                encoder = subprocess.Popen(["ffmpeg", "-y", "-v", "warning", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{WIDTH * 3}x{HEIGHT * 2}", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
                    "-threads", "4", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)],
                    stdin=subprocess.PIPE, stderr=log)
            for index in ([100] if preview else range(FPS * SECONDS)):
                canvas = Image.new("RGB", (WIDTH * 3, HEIGHT * 2))
                for panel, replay in enumerate(replays):
                    canvas.paste(replay.frame(index), ((panel % 3) * WIDTH, (panel // 3) * HEIGHT))
                canvas.paste(info_panel(scene, index, finetune_iteration), (WIDTH * 2, HEIGHT))
                if preview or index == 100:
                    canvas.save(out / f"gpt-同步对比预览-{scene}.png")
                if encoder:
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                if not preview and (index + 1) % 50 == 0:
                    print(f"{scene}：已渲染 {index + 1}/{FPS * SECONDS} 帧，耗时 {time.monotonic() - started:.1f} 秒", flush=True)
            if encoder:
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError(f"{scene} 编码失败，请检查日志")
                temporary.replace(target)
        evidence = [replay.evidence() for replay in replays]
    finally:
        for replay in replays:
            replay.close()
        if encoder and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=30)
        if temporary.exists():
            temporary.unlink()
    return {"场景": scene, "轨迹核验": evidence,
            "视频核验": None if preview else verify_video(target, SECONDS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--finetune-checkpoint", type=int, default=5000)
    parser.add_argument("--baseline-checkpoint", type=int, default=30000)
    parser.add_argument("--preview", action="store_true", help="完整数据通过核验后只生成两张同步预览")
    parser.add_argument("--validate-only", action="store_true", help="只检查正式轨迹完整性，不生成媒体")
    parser.add_argument("--internal-preview-run", type=Path, help="历史数据单帧验证，不能生成正式对比结果")
    parser.add_argument("--internal-preview-output", type=Path, default=Path("/tmp/gpt-四组视频内部验证"))
    args = parser.parse_args()
    if args.baseline_checkpoint != 30000:
        parser.error("本实验 A0 必须是原始三万次模型，baseline-checkpoint 必须为 30000。")
    model_dir = args.model_dir.resolve()
    if args.internal_preview_run:
        run = args.internal_preview_run.resolve()
        clip = load_clip(run, "内部验证", "楼梯8.5厘米" if "楼梯" in str(run) else "平地1.0", internal=True)
        load_rendering_dependencies()
        out = args.internal_preview_output.resolve()
        replay = Replay(clip, model_dir, out / "缓存")
        try:
            replay.frame(min(100, (len(clip.positions) - 1) // 2)).save(out / "gpt-历史轨迹内部验证-非实验结果.png")
            evidence = replay.evidence()
            evidence["用途"] = "历史数据内部渲染验证，非四组微调实验结果"
            write_json(out / "gpt-内部验证.json", evidence)
            print(json.dumps(evidence, ensure_ascii=False), flush=True)
        finally:
            replay.close()
        return
    root = args.root.resolve()
    try:
        require_completed_summary(root)
        clips = load_all(root, args.finetune_checkpoint, args.baseline_checkpoint)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")
    if args.validate_only:
        print("完整性核验通过：五组、两个场景、种子46，共10份正式轨迹。", flush=True)
        return
    load_rendering_dependencies()
    out = root / "对比视频"
    out.mkdir(parents=True, exist_ok=True)
    records = [render_scene(clips, scene, out, model_dir, args.preview) for scene in SCENES]
    result = {"状态": "同步预览已完成" if args.preview else "五组同步视频已完成", "正式完整输入": True,
              "场景": records, "每秒帧数": FPS, "渲染器版本": mujoco.__version__,
              "说明": "仅调用运动学前向计算与软件渲染；无策略推理和物理步进。首个终止/超时之前的真实数据包括终态，之后定格。",
              "机器人模型来源": str(model_dir), "机器人模型SHA256": sha256(model_dir / "go2.xml")}
    if not args.preview:
        target = out / "gpt-五组微调-平地与楼梯同步对比.mp4"
        videos = [record["视频核验"]["视频"] for record in records]
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-filter_complex_threads", "2", "-i", videos[0], "-i", videos[1],
                        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-an", "-c:v", "libx264",
                        "-threads", "4", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True)
        result["合成视频"] = verify_video(target, SECONDS * len(SCENES))
    marker = "gpt-同步预览核验.json" if args.preview else "gpt-同步视频完成.json"
    write_json(out / marker, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
