r"""Reusable atomic replay construction and MuJoCo React trajectory export.

Example usage is provided by the CLI wrapper:
uv run python -m robocasa.scripts.dataset_scripts.replay_atomic_on_scene `
  --scene-xml C:\path\to\scene.xml `
  --layout 12 `
  --task OpenDrawer `
  --robot-index 1 `
  --trajectory-json C:\path\to\public\trajectories\robot1_open_drawer.json `
  --no-viewer
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


DEMO_MOBILE_BASE_POSE_JOINTS = {
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
}
DEMO_ROBOT_JOINT_PREFIXES = ("robot0_", "mobilebase0_", "gripper0_")


@dataclass
class AtomicReplayClip:
    scene_nq: int
    source_fps: float
    phase: str
    robot_index: int
    affected_joints: tuple[str, ...]
    qpos_indices: np.ndarray
    patches: np.ndarray
    initial_qpos: np.ndarray
    target_body: str | None

    @property
    def frame_count(self):
        return int(self.patches.shape[0])

    @property
    def duration(self):
        if self.frame_count <= 1:
            return 0.0
        return (self.frame_count - 1) / self.source_fps


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
        if name:
            start = int(model.jnt_qposadr[joint_id])
            slices[name] = slice(start, start + _joint_qpos_width(model, joint_id))
    return slices


def state_qpos(state, model):
    nq_nv_na = model.nq + model.nv + model.na
    if len(state) >= 1 + nq_nv_na:
        return np.asarray(state[1 : 1 + model.nq], dtype=float)
    if len(state) >= nq_nv_na:
        return np.asarray(state[: model.nq], dtype=float)
    raise ValueError(
        f"State length {len(state)} is too short for model "
        f"nq={model.nq}, nv={model.nv}, na={model.na}"
    )


def _robot_joint_prefixes(robot_index):
    return (
        f"robot{robot_index}_",
        f"mobilebase{robot_index}_",
        f"gripper{robot_index}_",
    )


def mobile_base_pose_joints(robot_index):
    return (
        f"mobilebase{robot_index}_joint_mobile_forward",
        f"mobilebase{robot_index}_joint_mobile_side",
        f"mobilebase{robot_index}_joint_mobile_yaw",
    )


def map_demo_robot_name(name, robot_index):
    for demo_prefix, scene_prefix in zip(
        DEMO_ROBOT_JOINT_PREFIXES, _robot_joint_prefixes(robot_index)
    ):
        if name.startswith(demo_prefix):
            return f"{scene_prefix}{name.removeprefix(demo_prefix)}"
    return name


def match_joints(
    demo_model,
    scene_model,
    include_freejoints=False,
    skip_mobile_base_pose=True,
    robot_index=0,
):
    demo_slices = _joint_qpos_slices(demo_model)
    scene_slices = _joint_qpos_slices(scene_model)
    matches = []
    matched_scene_names = set()
    info = {
        "matches": matches,
        "demo_only": [],
        "scene_only": [],
        "skipped_width": [],
        "skipped_free": [],
        "skipped_mobile_base": [],
    }

    for demo_name in sorted(demo_slices):
        scene_name = map_demo_robot_name(demo_name, robot_index)
        if scene_name not in scene_slices:
            info["demo_only"].append(demo_name)
            continue
        demo_slice = demo_slices[demo_name]
        scene_slice = scene_slices[scene_name]
        demo_width = demo_slice.stop - demo_slice.start
        if demo_width != scene_slice.stop - scene_slice.start:
            info["skipped_width"].append((demo_name, scene_name))
            continue
        if demo_width == 7 and not include_freejoints:
            info["skipped_free"].append((demo_name, scene_name))
            continue
        if skip_mobile_base_pose and demo_name in DEMO_MOBILE_BASE_POSE_JOINTS:
            info["skipped_mobile_base"].append((demo_name, scene_name))
            continue
        matches.append((demo_name, scene_name, demo_slice, scene_slice))
        matched_scene_names.add(scene_name)

    info["scene_only"] = sorted(set(scene_slices) - matched_scene_names)
    return info


def copy_matched_qpos(demo_qpos, scene_qpos, matches):
    for _, _, demo_slice, scene_slice in matches:
        scene_qpos[scene_slice] = demo_qpos[demo_slice]


def _joint_names_for_body_subtree(model, root_body_ids):
    import mujoco

    body_ids = set(root_body_ids)
    if not body_ids:
        return set()
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            parent_id = int(model.body_parentid[body_id])
            if parent_id in body_ids and body_id not in body_ids:
                body_ids.add(body_id)
                changed = True

    names = set()
    for body_id in body_ids:
        start = int(model.body_jntadr[body_id])
        for offset in range(int(model.body_jntnum[body_id])):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, start + offset
            )
            if name:
                names.add(name)
    return names


def _body_ids_matching_ref(model, ref):
    import mujoco

    if not ref:
        return []
    result = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name == ref or (name and name.startswith(f"{ref}_")):
            result.append(body_id)
    return result


def _joint_names_for_refs(model, refs):
    names = set()
    for ref in refs:
        names.update(
            _joint_names_for_body_subtree(model, _body_ids_matching_ref(model, ref))
        )
    return names


def _fixture_refs(ep_meta, explicit_ref=None):
    refs = [explicit_ref] if explicit_ref else []
    fixture_refs = ep_meta.get("fixture_refs", {}) if isinstance(ep_meta, dict) else {}
    if isinstance(fixture_refs, dict):
        refs.extend(value for value in fixture_refs.values() if isinstance(value, str))
    return refs


def _object_refs(ep_meta):
    refs = []
    object_refs = ep_meta.get("object_refs", {}) if isinstance(ep_meta, dict) else {}
    if isinstance(object_refs, dict):
        refs.extend(value for value in object_refs.values() if isinstance(value, str))
    object_cfgs = ep_meta.get("object_cfgs", []) if isinstance(ep_meta, dict) else []
    for cfg in object_cfgs:
        if isinstance(cfg, dict) and isinstance(cfg.get("name"), str):
            refs.append(cfg["name"])
    return refs


def _is_demo_robot_joint(name):
    return name in DEMO_MOBILE_BASE_POSE_JOINTS or name.startswith(
        DEMO_ROBOT_JOINT_PREFIXES
    )


def local_patch_matches(
    match_info,
    demo_model,
    ep_meta,
    target_fixture_ref=None,
    include_objects=False,
):
    fixture_joints = _joint_names_for_refs(
        demo_model, _fixture_refs(ep_meta, target_fixture_ref)
    )
    object_joints = (
        _joint_names_for_refs(demo_model, _object_refs(ep_meta))
        if include_objects
        else set()
    )
    filtered = []
    skipped = []
    for match in match_info["matches"]:
        demo_name = match[0]
        if (
            _is_demo_robot_joint(demo_name)
            or demo_name in fixture_joints
            or demo_name in object_joints
        ):
            filtered.append(match)
        else:
            skipped.append(demo_name)

    result = dict(match_info)
    result["matches"] = filtered
    result["skipped_local_state"] = skipped
    result["target_fixture_joints"] = sorted(fixture_joints)
    result["target_object_joints"] = sorted(object_joints)
    return result


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
    return target_pos, target_quat / np.linalg.norm(target_quat)


def _body_exists(model, body_name):
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0


def _target_body_candidates(ep_meta, explicit_ref=None):
    candidates = []
    for ref in _fixture_refs(ep_meta, explicit_ref):
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


def choose_target_body(demo_model, scene_model, ep_meta, explicit_ref=None):
    for candidate in _target_body_candidates(ep_meta, explicit_ref):
        if _body_exists(demo_model, candidate) and _body_exists(scene_model, candidate):
            return candidate
    return None


def desired_mobile_base_pose(
    demo_model,
    scene_model,
    scene_data,
    demo_qpos,
    target_body,
    demo_mobile_base_body="mobilebase0_base",
):
    demo_pos, demo_quat = _body_pose_from_qpos(
        demo_model, demo_qpos, demo_mobile_base_body
    )
    if target_body is None:
        return demo_pos, demo_quat
    demo_target_pos, demo_target_quat = _body_pose_from_qpos(
        demo_model, demo_qpos, target_body
    )
    scene_target_pose = _body_pose_from_data(scene_model, scene_data, target_body)
    if scene_target_pose is None:
        return demo_pos, demo_quat
    return _transform_pose_between_frames(
        demo_target_pos,
        demo_target_quat,
        demo_pos,
        demo_quat,
        scene_target_pose[0],
        scene_target_pose[1],
    )


def _quat_yaw(quat):
    rot = _quat_to_mat(quat)
    return math.atan2(rot[1, 0], rot[0, 0])


def _wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def solve_mobile_base_qpos(
    scene_model,
    baseline_qpos,
    robot_index,
    desired_pos,
    desired_quat,
    tolerance=1e-8,
    max_iterations=12,
):
    import mujoco

    joint_slices = _joint_qpos_slices(scene_model)
    joint_names = mobile_base_pose_joints(robot_index)
    missing = [name for name in joint_names if name not in joint_slices]
    if missing:
        raise KeyError(f"Scene is missing mobile-base joints: {missing}")
    addresses = np.array([joint_slices[name].start for name in joint_names], dtype=int)
    body_name = f"mobilebase{robot_index}_base"
    body_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"Body {body_name!r} was not found")

    data = mujoco.MjData(scene_model)
    data.qpos[:] = baseline_qpos
    values = data.qpos[addresses].copy()
    desired = np.array([desired_pos[0], desired_pos[1], _quat_yaw(desired_quat)])

    def evaluate(q_values):
        data.qpos[:] = baseline_qpos
        data.qpos[addresses] = q_values
        mujoco.mj_forward(scene_model, data)
        current = np.array(
            [data.xpos[body_id, 0], data.xpos[body_id, 1], _quat_yaw(data.xquat[body_id])]
        )
        residual = current - desired
        residual[2] = _wrap_angle(residual[2])
        return residual

    for _ in range(max_iterations):
        residual = evaluate(values)
        if np.linalg.norm(residual) <= tolerance:
            break
        jacobian = np.zeros((3, 3), dtype=float)
        epsilon = 1e-6
        for column in range(3):
            perturbed = values.copy()
            perturbed[column] += epsilon
            delta = evaluate(perturbed) - residual
            delta[2] = _wrap_angle(delta[2])
            jacobian[:, column] = delta / epsilon
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        values += step
    final_residual = evaluate(values)
    if np.linalg.norm(final_residual) > 1e-5:
        raise RuntimeError(
            "Could not encode replay-adapter placement in mobile-base qpos; "
            f"residual={final_residual.tolist()}"
        )
    return dict(zip(joint_names, values)), addresses


def _affected_qpos_indices(matches, extra_joint_names, scene_model):
    scene_slices = _joint_qpos_slices(scene_model)
    indices = set()
    names = []
    for _, scene_name, _, scene_slice in matches:
        names.append(scene_name)
        indices.update(range(scene_slice.start, scene_slice.stop))
    for name in extra_joint_names:
        scene_slice = scene_slices[name]
        names.append(name)
        indices.update(range(scene_slice.start, scene_slice.stop))
    return tuple(dict.fromkeys(names)), np.array(sorted(indices), dtype=int)


def build_atomic_clip(
    demo_model,
    scene_model,
    states,
    ep_meta,
    robot_index=0,
    source_fps=20.0,
    phase="atomic",
    include_freejoints=False,
    local_state_replay=True,
    include_objects=False,
    target_fixture_ref=None,
    replay_adapter=True,
):
    import mujoco

    if source_fps <= 0:
        raise ValueError("source_fps must be greater than zero")
    states = np.asarray(states)
    if len(states) == 0:
        raise ValueError("states cannot be empty")
    scene_root = f"robot{robot_index}_base"
    if not _body_exists(scene_model, scene_root):
        raise KeyError(f"Selected scene robot body {scene_root!r} was not found")

    initial_data = mujoco.MjData(scene_model)
    initial_qpos = initial_data.qpos.copy()
    first_demo_qpos = state_qpos(states[0], demo_model)
    match_info = match_joints(
        demo_model,
        scene_model,
        include_freejoints=include_freejoints,
        skip_mobile_base_pose=replay_adapter,
        robot_index=robot_index,
    )
    if local_state_replay:
        match_info = local_patch_matches(
            match_info,
            demo_model,
            ep_meta,
            target_fixture_ref=target_fixture_ref,
            include_objects=include_objects,
        )
    if not any(_is_demo_robot_joint(match[0]) for match in match_info["matches"]):
        raise RuntimeError(f"No robot joints matched scene robot index {robot_index}")

    alignment_qpos = initial_qpos.copy()
    copy_matched_qpos(first_demo_qpos, alignment_qpos, match_info["matches"])
    alignment_data = mujoco.MjData(scene_model)
    alignment_data.qpos[:] = alignment_qpos
    mujoco.mj_forward(scene_model, alignment_data)

    target_body = choose_target_body(
        demo_model, scene_model, ep_meta, explicit_ref=target_fixture_ref
    )
    base_values = {}
    if replay_adapter:
        desired_pos, desired_quat = desired_mobile_base_pose(
            demo_model,
            scene_model,
            alignment_data,
            first_demo_qpos,
            target_body,
        )
        base_values, _ = solve_mobile_base_qpos(
            scene_model,
            alignment_qpos,
            robot_index,
            desired_pos,
            desired_quat,
        )

    affected_names, qpos_indices = _affected_qpos_indices(
        match_info["matches"], base_values.keys(), scene_model
    )
    scene_slices = _joint_qpos_slices(scene_model)
    patches = np.empty((len(states), len(qpos_indices)), dtype=float)
    scratch_data = mujoco.MjData(scene_model)
    for frame_index, state in enumerate(states):
        frame_qpos = initial_qpos.copy()
        copy_matched_qpos(
            state_qpos(state, demo_model), frame_qpos, match_info["matches"]
        )
        for joint_name, value in base_values.items():
            frame_qpos[scene_slices[joint_name]] = value
        scratch_data.qpos[:] = frame_qpos
        mujoco.mj_normalizeQuat(scene_model, scratch_data.qpos)
        if not np.all(np.isfinite(scratch_data.qpos)):
            raise ValueError(f"Non-finite qpos generated at frame {frame_index}")
        patches[frame_index] = scratch_data.qpos[qpos_indices]

    return AtomicReplayClip(
        scene_nq=int(scene_model.nq),
        source_fps=float(source_fps),
        phase=str(phase),
        robot_index=int(robot_index),
        affected_joints=affected_names,
        qpos_indices=qpos_indices,
        patches=patches,
        initial_qpos=initial_qpos,
        target_body=target_body,
    ), match_info


def materialize_full_qpos(clip, output_fps=None, baseline_qpos=None):
    output_fps = float(output_fps or clip.source_fps)
    if output_fps <= 0:
        raise ValueError("output_fps must be greater than zero")
    baseline = np.asarray(
        clip.initial_qpos if baseline_qpos is None else baseline_qpos, dtype=float
    )
    if baseline.shape != (clip.scene_nq,):
        raise ValueError(
            f"baseline qpos shape {baseline.shape} does not match nq={clip.scene_nq}"
        )

    output_count = max(1, int(round(clip.duration * output_fps)) + 1)
    times = np.arange(output_count, dtype=float) / output_fps
    source_indices = np.minimum(
        np.floor(times * clip.source_fps + 1e-9).astype(int),
        clip.frame_count - 1,
    )
    frames = np.repeat(baseline[None, :], output_count, axis=0)
    frames[:, clip.qpos_indices] = clip.patches[source_indices]
    return times, frames


def write_trajectory_json(path, clip, output_fps=30.0, baseline_qpos=None):
    path = Path(path)
    times, frames = materialize_full_qpos(clip, output_fps, baseline_qpos)
    if frames.shape[1] != clip.scene_nq:
        raise ValueError(f"Trajectory qpos width does not match nq={clip.scene_nq}")
    if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(times)):
        raise ValueError("Trajectory contains non-finite values")
    trajectory = [
        {
            "time": float(time_value),
            "qpos": qpos.tolist(),
            "phase": clip.phase,
        }
        for time_value, qpos in zip(times, frames)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(trajectory, file, ensure_ascii=True, allow_nan=False)
    return trajectory
