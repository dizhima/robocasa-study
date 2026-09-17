r"""Replay one RoboCasa atomic demo on a custom MJCF scene.

Example usage:
uv run python -m robocasa.scripts.dataset_scripts.replay_atomic_on_scene `
  --scene-xml <playground>/frontend/public/assets/robocasa/layout012_study.xml `
  --layout 12 `
  --task OpenDrawer `
  --episode-index 0 `
  --robot-index 1 `
  --trajectory-json <playground>/frontend/public/trajectories/layout012_study/robot1/open_drawer.json `
  --no-viewer
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from robocasa.scripts.dataset_scripts import atomic_replay_core as ReplayCore


def _default_atomic_root():
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "datasets" / "v1.0" / "pretrain" / "atomic"


def _default_summary_json():
    return _default_atomic_root() / "atomic_layout_summary.json"


def _load_summary(summary_json):
    if not summary_json.exists():
        raise FileNotFoundError(
            f"Summary JSON not found: {summary_json}\n"
            "Generate it with: uv run python -m robocasa.scripts.dataset_scripts.summarize_atomic_layouts"
        )
    return json.loads(summary_json.read_text(encoding="utf-8"))


def _select_episode(summary, task, layout, episode_index):
    layout_key = str(layout)
    try:
        episode_ids = summary["by_task"][task]["layouts"][layout_key]["episode_ids"]
    except KeyError as exc:
        available = sorted(summary.get("by_task", {}).get(task, {}).get("layouts", {}).keys())
        raise KeyError(
            f"No episodes found for task={task!r}, layout={layout_key!r}. "
            f"Available layouts for this task: {available}"
        ) from exc

    if not episode_ids:
        raise ValueError(f"No episode ids found for task={task}, layout={layout}")
    if episode_index < 0 or episode_index >= len(episode_ids):
        raise IndexError(
            f"episode-index {episode_index} is out of range for {len(episode_ids)} episodes"
        )

    episode_id = episode_ids[episode_index]
    return episode_id, summary["episodes"][episode_id]


def _episode_number(episode_name):
    if not episode_name.startswith("episode_"):
        raise ValueError(f"Unexpected episode name: {episode_name}")
    return int(episode_name.removeprefix("episode_"))


def _dataset_dir(atomic_root, episode):
    return atomic_root / episode["task"] / episode["dataset_date"] / "lerobot"


def _make_demo_env(dataset):
    import copy
    import robosuite
    import robocasa.utils.lerobot_utils as LU

    env_meta = LU.get_env_metadata(dataset)
    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def _load_demo_model_from_episode_xml(dataset, model_xml):
    import mujoco

    env = _make_demo_env(dataset)
    try:
        fixed_xml = env.edit_model_xml(model_xml)
    finally:
        env.close()
    return mujoco.MjModel.from_xml_string(fixed_xml)


def _joint_qpos_width(model, joint_id):
    import mujoco

    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_qpos_slices(model):
    import mujoco

    slices = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            continue
        start = int(model.jnt_qposadr[joint_id])
        stop = start + _joint_qpos_width(model, joint_id)
        slices[name] = slice(start, stop)
    return slices


MOBILE_BASE_POSE_JOINTS = {
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
}

ROBOT_JOINT_PREFIXES = (
    "robot0_",
    "mobilebase0_",
    "gripper0_",
)


def _robot_joint_prefixes(robot_index):
    return (
        f"robot{robot_index}_",
        f"mobilebase{robot_index}_",
        f"gripper{robot_index}_",
    )


def _mobile_base_pose_joints(robot_index):
    return {
        f"mobilebase{robot_index}_joint_mobile_forward",
        f"mobilebase{robot_index}_joint_mobile_side",
        f"mobilebase{robot_index}_joint_mobile_yaw",
    }


def _map_demo_robot_name(name, robot_index):
    for demo_prefix, scene_prefix in zip(
        ROBOT_JOINT_PREFIXES, _robot_joint_prefixes(robot_index)
    ):
        if name.startswith(demo_prefix):
            return f"{scene_prefix}{name.removeprefix(demo_prefix)}"
    return name


def _match_joints(
    demo_model,
    scene_model,
    include_freejoints,
    skip_mobile_base_pose,
    robot_index=0,
):
    demo_slices = _joint_qpos_slices(demo_model)
    scene_slices = _joint_qpos_slices(scene_model)

    matches = []
    skipped_width = []
    skipped_free = []
    skipped_mobile_base = []

    matched_scene_names = set()
    demo_only = []
    for demo_name in sorted(demo_slices):
        scene_name = _map_demo_robot_name(demo_name, robot_index)
        if scene_name not in scene_slices:
            demo_only.append(demo_name)
            continue
        demo_slice = demo_slices[demo_name]
        scene_slice = scene_slices[scene_name]
        width = demo_slice.stop - demo_slice.start
        if width != scene_slice.stop - scene_slice.start:
            skipped_width.append((demo_name, scene_name))
            continue
        if width == 7 and not include_freejoints:
            skipped_free.append((demo_name, scene_name))
            continue
        if skip_mobile_base_pose and demo_name in MOBILE_BASE_POSE_JOINTS:
            skipped_mobile_base.append((demo_name, scene_name))
            continue
        matches.append((demo_name, scene_name, demo_slice, scene_slice))
        matched_scene_names.add(scene_name)

    return {
        "matches": matches,
        "demo_only": demo_only,
        "scene_only": sorted(set(scene_slices) - matched_scene_names),
        "skipped_width": skipped_width,
        "skipped_free": skipped_free,
        "skipped_mobile_base": skipped_mobile_base,
    }


def _state_qpos(state, model):
    nq_nv_na = model.nq + model.nv + model.na
    if len(state) >= 1 + nq_nv_na:
        return state[1 : 1 + model.nq]
    if len(state) >= nq_nv_na:
        return state[: model.nq]
    raise ValueError(
        f"State length {len(state)} is too short for model nq={model.nq}, nv={model.nv}, na={model.na}"
    )


def _copy_matched_qpos(demo_qpos, scene_data, matches):
    for _, _, demo_slice, scene_slice in matches:
        scene_data.qpos[scene_slice] = demo_qpos[demo_slice]


def _joint_names_for_body_subtree(model, root_body_ids):
    import mujoco

    root_body_ids = set(root_body_ids)
    if not root_body_ids:
        return set()

    body_ids = set(root_body_ids)
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            parent_id = int(model.body_parentid[body_id])
            if parent_id in body_ids and body_id not in body_ids:
                body_ids.add(body_id)
                changed = True

    joint_names = set()
    for body_id in body_ids:
        joint_start = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        for offset in range(joint_count):
            joint_id = joint_start + offset
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                joint_names.add(name)
    return joint_names


def _body_ids_matching_ref(model, ref):
    import mujoco

    body_ids = []
    if not ref:
        return body_ids
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name == ref or (name and name.startswith(f"{ref}_")):
            body_ids.append(body_id)
    return body_ids


def _object_refs_from_meta(ep_meta):
    refs = []
    object_refs = ep_meta.get("object_refs", {}) if isinstance(ep_meta, dict) else {}
    if isinstance(object_refs, dict):
        refs.extend(value for value in object_refs.values() if isinstance(value, str))
    object_cfgs = ep_meta.get("object_cfgs", []) if isinstance(ep_meta, dict) else []
    for cfg in object_cfgs:
        if not isinstance(cfg, dict):
            continue
        name = cfg.get("name")
        if isinstance(name, str):
            refs.append(name)
    return refs


def _fixture_refs_from_meta(ep_meta, explicit_ref=None):
    refs = []
    if explicit_ref is not None:
        refs.append(explicit_ref)
    fixture_refs = ep_meta.get("fixture_refs", {}) if isinstance(ep_meta, dict) else {}
    if isinstance(fixture_refs, dict):
        refs.extend(value for value in fixture_refs.values() if isinstance(value, str))
    return refs


def _joint_names_for_refs(model, refs):
    names = set()
    for ref in refs:
        names.update(_joint_names_for_body_subtree(model, _body_ids_matching_ref(model, ref)))
    return names


def _is_robot_joint(name):
    return name in MOBILE_BASE_POSE_JOINTS or name.startswith(ROBOT_JOINT_PREFIXES)


def _local_patch_matches(match_info, demo_model, ep_meta, target_fixture_ref, include_objects):
    target_fixture_joints = _joint_names_for_refs(
        demo_model,
        _fixture_refs_from_meta(ep_meta, explicit_ref=target_fixture_ref),
    )
    target_object_joints = set()
    if include_objects:
        target_object_joints = _joint_names_for_refs(demo_model, _object_refs_from_meta(ep_meta))

    filtered = []
    skipped_local_state = []
    for demo_name, scene_name, demo_slice, scene_slice in match_info["matches"]:
        if (
            _is_robot_joint(demo_name)
            or demo_name in target_fixture_joints
            or demo_name in target_object_joints
        ):
            filtered.append((demo_name, scene_name, demo_slice, scene_slice))
        else:
            skipped_local_state.append(demo_name)

    patch_info = dict(match_info)
    patch_info["matches"] = filtered
    patch_info["skipped_local_state"] = skipped_local_state
    patch_info["target_fixture_joints"] = sorted(target_fixture_joints)
    patch_info["target_object_joints"] = sorted(target_object_joints)
    return patch_info


def _body_pose_from_qpos(model, qpos, body_name):
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body {body_name!r} was not found")
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _body_pose_from_data(model, data, body_name):
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _quat_conj(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_to_mat(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _transform_pose_between_frames(
    source_ref_pos,
    source_ref_quat,
    source_pose_pos,
    source_pose_quat,
    target_ref_pos,
    target_ref_quat,
):
    source_ref_rot = _quat_to_mat(source_ref_quat)
    target_ref_rot = _quat_to_mat(target_ref_quat)
    rel_pos = source_ref_rot.T @ (source_pose_pos - source_ref_pos)
    rel_quat = _quat_mul(_quat_conj(source_ref_quat), source_pose_quat)
    target_pos = target_ref_pos + target_ref_rot @ rel_pos
    target_quat = _quat_mul(target_ref_quat, rel_quat)
    target_quat = target_quat / np.linalg.norm(target_quat)
    return target_pos, target_quat


def _set_body_pose(model, body_name, pos, quat):
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body {body_name!r} was not found")
    model.body_pos[body_id] = pos
    model.body_quat[body_id] = quat


def _set_scalar_joint_qpos(model, data, joint_name, value):
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return False
    addr = int(model.jnt_qposadr[joint_id])
    if _joint_qpos_width(model, joint_id) != 1:
        return False
    data.qpos[addr] = value
    return True


def _zero_mobile_base_pose_qpos(scene_model, scene_data, robot_index=0):
    for joint_name in _mobile_base_pose_joints(robot_index):
        _set_scalar_joint_qpos(scene_model, scene_data, joint_name, 0.0)


def _align_scene_robot_to_demo_start(
    demo_model,
    scene_model,
    scene_data,
    demo_qpos,
    robot_root_body,
    mobile_base_body,
    target_body=None,
    robot_index=0,
):
    demo_mobile_pos, demo_mobile_quat = _body_pose_from_qpos(demo_model, demo_qpos, mobile_base_body)
    pos, quat = demo_mobile_pos, demo_mobile_quat
    if target_body is not None:
        demo_target_pos, demo_target_quat = _body_pose_from_qpos(
            demo_model, demo_qpos, target_body
        )
        scene_target_pose = _body_pose_from_data(scene_model, scene_data, target_body)
        if scene_target_pose is not None:
            scene_target_pos, scene_target_quat = scene_target_pose
            pos, quat = _transform_pose_between_frames(
                source_ref_pos=demo_target_pos,
                source_ref_quat=demo_target_quat,
                source_pose_pos=demo_mobile_pos,
                source_pose_quat=demo_mobile_quat,
                target_ref_pos=scene_target_pos,
                target_ref_quat=scene_target_quat,
            )
    _set_body_pose(scene_model, robot_root_body, pos, quat)
    _zero_mobile_base_pose_qpos(scene_model, scene_data, robot_index)
    return pos, quat


def _body_exists(model, body_name):
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0


def _target_body_candidates(ep_meta, explicit_ref=None):
    fixture_refs = ep_meta.get("fixture_refs", {}) if isinstance(ep_meta, dict) else {}
    refs = []
    if explicit_ref is not None:
        refs.append(explicit_ref)
    refs.extend(value for value in fixture_refs.values() if isinstance(value, str))

    candidates = []
    for ref in refs:
        candidates.extend(
            [
                f"{ref}_door_handle_main",
                f"{ref}_right_door_handle_main",
                f"{ref}_left_door_handle_main",
                f"{ref}_handle_main",
                f"{ref}_door_main",
                f"{ref}_main",
            ]
        )
    return candidates


def _choose_target_body(demo_model, scene_model, ep_meta, explicit_ref=None):
    for candidate in _target_body_candidates(ep_meta, explicit_ref):
        if _body_exists(demo_model, candidate) and _body_exists(scene_model, candidate):
            return candidate
    return None


def _copy_body_pose(demo_model, scene_model, demo_body_name, scene_body_name):
    import mujoco

    demo_body_id = mujoco.mj_name2id(
        demo_model, mujoco.mjtObj.mjOBJ_BODY, demo_body_name
    )
    scene_body_id = mujoco.mj_name2id(
        scene_model, mujoco.mjtObj.mjOBJ_BODY, scene_body_name
    )
    if demo_body_id < 0:
        raise KeyError(
            f"Body {demo_body_name!r} was not found in the recorded episode model"
        )
    if scene_body_id < 0:
        raise KeyError(
            f"Body {scene_body_name!r} was not found in the custom scene model"
        )

    scene_model.body_pos[scene_body_id] = demo_model.body_pos[demo_body_id]
    scene_model.body_quat[scene_body_id] = demo_model.body_quat[demo_body_id]


def _print_match_summary(match_info, max_names):
    matches = match_info["matches"]
    print(f"Matched joints: {len(matches)}")
    print(f"Demo-only joints: {len(match_info['demo_only'])}")
    print(f"Scene-only joints: {len(match_info['scene_only'])}")
    print(f"Skipped width mismatch: {len(match_info['skipped_width'])}")
    print(f"Skipped free joints: {len(match_info['skipped_free'])}")
    print(f"Skipped mobile base pose joints: {len(match_info['skipped_mobile_base'])}")
    if "skipped_local_state" in match_info:
        print(f"Skipped non-local state joints: {len(match_info['skipped_local_state'])}")
    if "target_fixture_joints" in match_info:
        print(f"Target fixture joints: {len(match_info['target_fixture_joints'])}")
    if "target_object_joints" in match_info:
        print(f"Target object joints: {len(match_info['target_object_joints'])}")
    if max_names > 0:
        print("First matched joints:")
        for demo_name, scene_name, _, _ in matches[:max_names]:
            if demo_name == scene_name:
                print(f"  {demo_name}")
            else:
                print(f"  {demo_name} -> {scene_name}")
        for label in (
            "demo_only",
            "scene_only",
            "skipped_width",
            "skipped_free",
            "skipped_mobile_base",
            "skipped_local_state",
            "target_fixture_joints",
            "target_object_joints",
        ):
            values = match_info.get(label, [])
            if values:
                print(f"First {label}:")
                for item in values[:max_names]:
                    if isinstance(item, tuple):
                        print(f"  {item[0]} -> {item[1]}")
                    else:
                        print(f"  {item}")


def replay_on_scene(args):
    import mujoco
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_utils import (
        resolve_instruction_from_ep_meta,
    )

    robot_index = args.robot_index
    scene_robot_root_body = args.robot_root_body or f"robot{robot_index}_base"
    summary_json = args.summary_json
    summary = _load_summary(summary_json)
    atomic_root = Path(summary["summary"]["atomic_root"])
    episode_id, episode = _select_episode(
        summary=summary,
        task=args.task,
        layout=args.layout,
        episode_index=args.episode_index,
    )
    dataset = _dataset_dir(atomic_root, episode)
    episode_num = _episode_number(episode["episode"])

    print(f"Scene XML: {args.scene_xml}")
    print(f"Task: {args.task}")
    print(f"Layout: {args.layout}")
    print(f"Episode index: {args.episode_index}")
    print(f"Scene robot index: {robot_index}")
    print(f"Scene robot root body: {scene_robot_root_body}")
    print(f"Episode id: {episode_id}")
    print(f"Dataset: {dataset}")

    states = LU.get_episode_states(dataset, episode_num)
    model_xml = LU.get_episode_model_xml(dataset, episode_num)
    ep_meta = LU.get_episode_meta(dataset, episode_num)
    instruction = resolve_instruction_from_ep_meta(ep_meta)
    if instruction:
        print(f"Instruction: {instruction}")
    print(f"Frames: {len(states)}")

    print("Loading demo episode model...")
    demo_model = _load_demo_model_from_episode_xml(dataset, model_xml)
    print(
        f"Demo model nq/nv/na: {demo_model.nq}/{demo_model.nv}/{demo_model.na}; "
        f"state width: {states.shape[1]}"
    )

    scene_model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    if not _body_exists(scene_model, scene_robot_root_body):
        raise KeyError(
            f"Selected scene robot body {scene_robot_root_body!r} was not found. "
            f"Check --robot-index or --robot-root-body."
        )
    if args.use_recorded_robot_root_pose:
        _copy_body_pose(
            demo_model=demo_model,
            scene_model=scene_model,
            demo_body_name="robot0_base",
            scene_body_name=scene_robot_root_body,
        )
        print(f"Copied recorded robot0_base pose to: {scene_robot_root_body}")

    clip, replay_match_info = ReplayCore.build_atomic_clip(
        demo_model=demo_model,
        scene_model=scene_model,
        states=states,
        ep_meta=ep_meta,
        robot_index=robot_index,
        source_fps=args.fps,
        phase=args.phase or args.task,
        include_freejoints=args.include_freejoints,
        local_state_replay=args.local_state_replay,
        include_objects=args.local_state_include_objects,
        target_fixture_ref=args.target_fixture_ref,
        replay_adapter=args.replay_adapter,
    )
    _print_match_summary(replay_match_info, args.print_names)
    print(f"Replay adapter target body: {clip.target_body or '<none>'}")
    print(f"Affected scene joints: {len(clip.affected_joints)}")

    if args.trajectory_json is not None:
        trajectory = ReplayCore.write_trajectory_json(
            args.trajectory_json,
            clip,
            output_fps=args.trajectory_fps,
        )
        print(
            f"Wrote {args.trajectory_json} ({len(trajectory)} frames, "
            f"nq={clip.scene_nq}, fps={args.trajectory_fps:g})"
        )

    if args.dry_run or args.no_viewer:
        return

    import mujoco.viewer

    scene_data = mujoco.MjData(scene_model)
    _, replay_qpos = ReplayCore.materialize_full_qpos(clip, output_fps=args.fps)
    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    frame_indices = list(range(0, len(replay_qpos), args.frame_stride))
    if args.extend_last > 0:
        frame_indices.extend([len(replay_qpos) - 1] * args.extend_last)

    with mujoco.viewer.launch_passive(scene_model, scene_data) as viewer:
        for frame_idx in frame_indices:
            if not viewer.is_running():
                break
            start = time.time()
            scene_data.qpos[:] = replay_qpos[frame_idx]
            mujoco.mj_forward(scene_model, scene_data)
            viewer.sync()
            if frame_dt > 0:
                elapsed = time.time() - start
                time.sleep(max(0.0, frame_dt - elapsed))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a RoboCasa atomic recorded-state episode on a custom MJCF scene by matching joint names."
    )
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--robot-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="scene robot index that receives the recorded robot0 state",
    )
    parser.add_argument("--summary-json", type=Path, default=_default_summary_json())
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--trajectory-json",
        type=Path,
        default=None,
        help="write a full-qpos JSON trajectory compatible with mujoco-react",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=float,
        default=30.0,
        help="output JSON frame rate; defaults to the current frontend's 30 FPS",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        help="readable phase stored in every JSON frame; defaults to the task name",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="build and optionally export the replay without opening MuJoCo viewer",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--extend-last", type=int, default=50)
    parser.add_argument(
        "--include-freejoints",
        action="store_true",
        help="also map 7-DoF free joints when names match",
    )
    parser.add_argument(
        "--use-recorded-robot-root-pose",
        action="store_true",
        help=(
            "copy the robot root body pos/quat from the recorded episode model before "
            "replaying qpos; useful when the custom scene is close to the original layout"
        ),
    )
    parser.add_argument(
        "--robot-root-body",
        type=str,
        default=None,
        help=(
            "optional scene robot root body override; defaults to robot{robot-index}_base"
        ),
    )
    parser.add_argument(
        "--mobile-base-body",
        type=str,
        default="mobilebase0_base",
        help="recorded demo mobile-base body used by the replay adapter",
    )
    parser.add_argument(
        "--target-fixture-ref",
        type=str,
        default=None,
        help=(
            "fixture ref/name used for target-relative replay alignment; "
            "defaults to the first fixture_refs entry from the episode metadata"
        ),
    )
    parser.add_argument(
        "--no-replay-adapter",
        dest="replay_adapter",
        action="store_false",
        help=(
            "disable the default replay adapter and directly copy all matched qpos "
            "except skipped free joints"
        ),
    )
    parser.set_defaults(replay_adapter=True)
    parser.add_argument(
        "--full-state-replay",
        dest="local_state_replay",
        action="store_false",
        help=(
            "copy every matched qpos from the recorded state; by default this script "
            "uses local-state replay and only patches robot joints plus the target "
            "fixture joints so chained actions keep earlier scene changes"
        ),
    )
    parser.set_defaults(local_state_replay=True)
    parser.add_argument(
        "--local-state-include-objects",
        action="store_true",
        help=(
            "also allow object joints from episode metadata in local-state replay; "
            "useful for pick/place tasks, but off by default to avoid moving distractor objects"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load both models and print joint matching info without opening a viewer",
    )
    parser.add_argument(
        "--print-names",
        type=int,
        default=20,
        help="number of joint names to print in each diagnostic group",
    )
    args = parser.parse_args()
    if not args.scene_xml.exists():
        parser.error(f"scene XML does not exist: {args.scene_xml}")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.extend_last < 0:
        parser.error("--extend-last must be >= 0")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.trajectory_fps <= 0:
        parser.error("--trajectory-fps must be > 0")
    return args


def main():
    replay_on_scene(parse_args())


if __name__ == "__main__":
    main()
