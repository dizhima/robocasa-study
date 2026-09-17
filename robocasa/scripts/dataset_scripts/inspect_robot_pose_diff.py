import argparse
import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


MOBILE_JOINTS = [
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "mobilebase0_joint_torso_height",
]


def _read_xml(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def _find_body(root, name):
    for elem in root.iter("body"):
        if elem.get("name") == name:
            return elem
    return None


def _joint_qpos_width(joint):
    joint_type = joint.get("type", "hinge")
    if joint_type == "free":
        return 7
    if joint_type == "ball":
        return 4
    return 1


def _joint_qpos_addresses(root):
    joints = {}
    qpos_addr = 0
    for elem in root.iter("joint"):
        name = elem.get("name")
        if name:
            width = _joint_qpos_width(elem)
            joints[name] = (qpos_addr, width)
            qpos_addr += width
    return joints


def _key_qpos(root):
    keyframe = root.find("keyframe")
    if keyframe is None:
        return {}
    result = {}
    for key in keyframe.findall("key"):
        qpos = key.get("qpos")
        if qpos:
            result[key.get("name", "<unnamed>")] = np.fromstring(qpos, sep=" ")
    return result


def _state_qpos(states_npz, qpos_width):
    data = np.load(states_npz)["states"]
    first = data[0]
    if len(first) >= 1 + qpos_width:
        return first[1 : 1 + qpos_width]
    return first[:qpos_width]


def _load_demo_model(dataset, model_xml):
    import copy
    import mujoco
    import robosuite
    import robocasa.utils.lerobot_utils as LU

    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env = robosuite.make(**env_kwargs)
    try:
        fixed_xml = env.edit_model_xml(model_xml)
    finally:
        env.close()
    return mujoco.MjModel.from_xml_string(fixed_xml)


def _load_custom_model(xml_path):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(xml_path))


def _model_joint_info(model, joint_name):
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None
    return (
        int(model.jnt_qposadr[joint_id]),
        int(model.jnt_type[joint_id]),
        model.jnt_axis[joint_id].copy(),
        model.jnt_pos[joint_id].copy(),
    )


def _body_world_pose(model, qpos, body_name):
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _joint_qpos_slices(model):
    import mujoco

    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
    }
    result = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        start = int(model.jnt_qposadr[joint_id])
        width = widths.get(int(model.jnt_type[joint_id]), 1)
        result[name] = slice(start, start + width)
    return result


def _copy_body_pose(demo_model, scene_model, body_name):
    import mujoco

    demo_id = mujoco.mj_name2id(demo_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    scene_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if demo_id >= 0 and scene_id >= 0:
        scene_model.body_pos[scene_id] = demo_model.body_pos[demo_id]
        scene_model.body_quat[scene_id] = demo_model.body_quat[demo_id]


def _simulate_replay_mapping(demo_model, scene_model, demo_qpos):
    scene_qpos = np.zeros(scene_model.nq)
    demo_slices = _joint_qpos_slices(demo_model)
    scene_slices = _joint_qpos_slices(scene_model)
    for name, demo_slice in demo_slices.items():
        scene_slice = scene_slices.get(name)
        if scene_slice is None:
            continue
        if demo_slice.stop - demo_slice.start != scene_slice.stop - scene_slice.start:
            continue
        scene_qpos[scene_slice] = demo_qpos[demo_slice]
    _copy_body_pose(demo_model, scene_model, "robot0_base")
    return _body_world_pose(scene_model, scene_qpos, "mobilebase0_base")


def _summarize(label, xml_path, states_npz=None, dataset=None):
    xml = _read_xml(xml_path)
    root = ET.fromstring(xml)
    joints = _joint_qpos_addresses(root)
    robot_base = _find_body(root, "robot0_base")
    mobile_base = _find_body(root, "mobilebase0_base")
    model = _load_demo_model(dataset, xml) if dataset is not None else _load_custom_model(xml_path)

    print(f"\n== {label} ==")
    print(f"xml: {xml_path}")
    if robot_base is None:
        print("robot0_base: <missing>")
    else:
        print(f"robot0_base pos: {robot_base.get('pos')}")
        print(f"robot0_base quat: {robot_base.get('quat')}")
    if mobile_base is None:
        print("mobilebase0_base: <missing>")
    else:
        print(f"mobilebase0_base pos: {mobile_base.get('pos')}")
        print(f"mobilebase0_base quat: {mobile_base.get('quat')}")

    for joint in MOBILE_JOINTS:
        model_info = _model_joint_info(model, joint)
        if model_info is not None:
            addr, joint_type, axis, pos = model_info
            print(
                f"compiled joint qpos {joint}: "
                f"addr={addr} type={joint_type} axis={axis} pos={pos}"
            )
        elif joint in joints:
            addr, width = joints[joint]
            print(f"xml joint qpos {joint}: addr={addr} width={width}")
        else:
            print(f"joint qpos {joint}: <missing>")

    key_qpos = _key_qpos(root)
    if key_qpos:
        for key_name, qpos in key_qpos.items():
            print(f"key {key_name} qpos len: {len(qpos)}")
            for joint in MOBILE_JOINTS:
                model_info = _model_joint_info(model, joint)
                if model_info is not None:
                    addr, _, _, _ = model_info
                    if addr < len(qpos):
                        print(f"  key {joint}: {qpos[addr]}")
            if len(qpos) == model.nq:
                pose = _body_world_pose(model, qpos, "mobilebase0_base")
                if pose is not None:
                    pos, quat = pose
                    print(f"  key mobilebase0_base world pos: {pos}")
                    print(f"  key mobilebase0_base world quat: {quat}")
    else:
        print("keyframe qpos: <none>")

    qpos = None
    if states_npz is not None:
        qpos = _state_qpos(states_npz, model.nq)
        print(f"states first qpos len: {len(qpos)}")
        for joint in MOBILE_JOINTS:
            model_info = _model_joint_info(model, joint)
            if model_info is not None:
                addr, _, _, _ = model_info
                if addr < len(qpos):
                    print(f"  state {joint}: {qpos[addr]}")
        pose = _body_world_pose(model, qpos, "mobilebase0_base")
        if pose is not None:
            pos, quat = pose
            print(f"  state mobilebase0_base world pos: {pos}")
            print(f"  state mobilebase0_base world quat: {quat}")

    return model, qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--custom-xml", type=Path, required=True)
    args = parser.parse_args()

    ep_meta = json.loads((args.episode_dir / "ep_meta.json").read_text(encoding="utf-8"))
    print("Episode metadata:")
    for key in ("layout_id", "style_id", "init_robot_base_pos", "init_robot_base_ori"):
        print(f"  {key}: {ep_meta.get(key)}")

    episode_model, episode_qpos = _summarize(
        "episode",
        args.episode_dir / "model.xml.gz",
        states_npz=args.episode_dir / "states.npz",
        dataset=args.episode_dir.parent.parent,
    )
    custom_model, _ = _summarize("custom", args.custom_xml)
    if episode_qpos is not None:
        pose = _simulate_replay_mapping(episode_model, custom_model, episode_qpos)
        if pose is not None:
            pos, quat = pose
            print("\n== simulated mapped replay on custom ==")
            print(f"mobilebase0_base world pos: {pos}")
            print(f"mobilebase0_base world quat: {quat}")


if __name__ == "__main__":
    main()
