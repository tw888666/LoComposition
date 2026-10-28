"""把真实Isaac Sim评估轨迹回放成视频；只做运动学与渲染，不重新运行物理或策略。"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".gpt-video-deps"))
os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless",
                  LIBGL_ALWAYS_SOFTWARE="1", MESA_SHADER_CACHE_DIR="/tmp/gpt-mesa-video", LP_NUM_THREADS="8")
import argparse
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

OUT = ROOT / "gpt-复现记录/gpt-30000保留质心随机偏移录像-20261026/平地视频"
SOURCE = next((ROOT / "gpt-复现记录/gpt-30000保留质心随机偏移录像-20261026/录像评估").glob("eval_*/plots/sim_data.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    data = np.load(SOURCE, allow_pickle=False)
    summary = json.loads((SOURCE.parent.parent / "metrics_summary.json").read_text())
    model_dir = OUT.parent / "机器人模型"
    tree = ET.parse(model_dir / "scene.xml")
    visual = tree.getroot().find("visual")
    visual.find("global").set("offwidth", "960")
    visual.find("global").set("offheight", "540")
    visual.find("headlight").set("ambient", "0.4 0.4 0.4")
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    for tx in tree.getroot().find("asset").findall("texture"):
        if tx.get("name") == "groundplane":
            tx.set("rgb1", "0.55 0.59 0.63")
            tx.set("rgb2", "0.43 0.48 0.53")
        if tx.get("type") == "skybox":
            tx.set("rgb1", "0.72 0.80 0.88")
            tx.set("rgb2", "0.90 0.94 0.97")
    scene = model_dir / "gpt-回放场景.xml"
    tree.write(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(scene))
    state = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(n))]
                 for n in data["joint_names"]]
    assert len(set(addresses)) == 12 and min(addresses) >= 7
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in ("FL", "FR", "RL", "RR")]

    scenarios = {row[0]: i for i, row in enumerate(summary["fixed_command_scenarios"])}
    clips = [
        ("平地行走 · 指令 1.0 米/秒", "walk_x_flat_terrain_1.0mps", np.arange(0, 500, 2), 1.0),
        ("平地快走 · 指令 1.6 米/秒", "walk_x_flat_terrain_1.6mps", np.arange(0, 500, 2), 1.0),
        ("步态细看 · 指令 1.0 米/秒", "walk_x_flat_terrain_1.0mps", np.arange(100, 300), 0.5),
    ]
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (17, 20, 25)}
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 1.95, 70, -18
    opts = mujoco.MjvOption()
    opts.geomgroup[3] = 0
    target = OUT / "gpt-自训30000保留质心偏移平地行走.mp4"
    frames, errors, records = 0, [], []
    encoder = None
    if not args.preview:
        log = (OUT / "gpt-视频编码.log").open("w")
        encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "warning", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-s", "960x540", "-r", "25", "-i", "pipe:0",
                                    "-an", "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "20",
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                                   stdin=subprocess.PIPE, stderr=log)
    started = time.monotonic()
    try:
        for title, scenario, offsets, speed in clips:
            assert summary["fixed_command_scenario_completion"][scenario], "场景提前失败，应保留失败并截断录像"
            start = summary["random_sim_steps"] + scenarios[scenario] * summary["fixed_command_sim_steps"]
            origin = data["base_position_array"][start].copy()
            origin[2] = 0.0
            world_rot = Rotation.from_euler("z", -data["base_orientation_array"][start, 0])
            records.append({"场景": scenario, "标题": title, "首控制步": int(start + offsets[0]),
                            "末控制步": int(start + offsets[-1]), "原轨迹控制步间隔": int(offsets[1]-offsets[0]),
                            "播放速度": speed, "视频起始秒": frames/25, "视频时长秒": len(offsets)/25})
            for offset in ([150] if args.preview else offsets):
                idx = start + offset
                state.qpos[:3] = world_rot.apply(data["base_position_array"][idx] - origin)
                yaw, pitch, roll = data["base_orientation_array"][idx]
                q = (world_rot * Rotation.from_euler("xyz", [roll, pitch, yaw])).as_quat()
                state.qpos[3:7] = q[[3, 0, 1, 2]]
                state.qpos[addresses] = data["joint_positions_array"][idx]
                mujoco.mj_forward(model, state)
                reference = world_rot.apply(data["foot_positions_world_frame_array"][idx] - origin)
                error = np.linalg.norm(state.geom_xpos[foot_ids] - reference, axis=1)
                errors.extend(error.tolist())
                if error.max() > .015:
                    raise RuntimeError(f"足端运动学与真实轨迹不符：步{idx}，最大误差{error.max()}米")
                camera.lookat[:] = state.qpos[:3] + [0.08, 0, -.02]
                renderer.update_scene(state, camera=camera, scene_option=opts)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 99), fill=(19, 29, 43))
                draw.text((24, 12), f"自训 30000  |  {title}", font=fonts[25], fill="white")
                draw.text((24, 52), "真实仿真轨迹回放  ·  保留质心随机偏移  ·  评估种子 46", font=fonts[17], fill=(194, 213, 230))
                draw.rectangle((0, 477, 960, 540), fill=(19, 29, 43))
                velocity = data["base_linear_velocity_body_array"][idx, 0]
                draw.text((24, 490), f"时间 {offset*.02:4.2f} 秒    实际前向速度 {velocity:.2f} 米/秒", font=fonts[20], fill="white")
                draw.text((24, 517), "机身与关节运动来自原评估数据；独立渲染，未重新进行物理仿真", font=fonts[17], fill=(194, 213, 230))
                draw.text((784, 12), "正常速度" if speed == 1 else "0.5 倍慢放", font=fonts[20], fill=(116, 229, 187))
                for j, label in enumerate(("左前", "右前", "左后", "右后")):
                    x = 635 + j * 78
                    color = (85, 220, 161) if data["contact_state_array"][idx, j] else (83, 97, 116)
                    draw.ellipse((x, 491, x + 11, 502), fill=color)
                    draw.text((x + 16, 484), label, font=fonts[17], fill="white")
                if args.preview:
                    frame.save(OUT / "gpt-视频预览.png")
                    print("预览已保存；足端最大误差米", max(errors), flush=True)
                    return
                encoder.stdin.write(np.asarray(frame).tobytes())
                frames += 1
                if frames % 100 == 0:
                    print(f"已渲染{frames}/700帧，用时{time.monotonic()-started:.1f}秒", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("视频编码失败，请检查日志")
    finally:
        renderer.close()
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait()
    metadata = {"模型": "model_30000.pt", "条件": "保留质心随机偏移",
                "来源轨迹": str(SOURCE), "来源SHA256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "说明": "原始仿真轨迹的运动学回放，MuJoCo仅用于独立渲染，没有执行mj_step或重新进行策略推理。地面为原场景平面；显示坐标整体平移旋转以便观看。",
                "模型几何来源": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2",
                "足端几何中心与记录足端位置最大误差米": max(errors), "平均误差米": float(np.mean(errors)),
                "片段": records, "帧数": frames, "每秒帧数": 25, "视频时长秒": frames/25,
                "渲染器版本": mujoco.__version__, "视频": str(target),
                "关节映射": {str(name): int(addr) for name, addr in zip(data["joint_names"], addresses)}}
    (OUT / "gpt-视频来源与核验.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("完成", target, flush=True)


if __name__ == "__main__":
    main()
