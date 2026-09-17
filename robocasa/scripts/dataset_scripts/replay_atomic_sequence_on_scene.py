r"""Replay multiple RoboCasa atomic demos on one custom MJCF scene.

Example usage:
uv run python -m robocasa.scripts.dataset_scripts.replay_atomic_sequence_on_scene `
  --scene-xml <playground>/frontend/public/assets/robocasa/layout012_study.xml `
  --layout 12 `
  --tasks OpenDrawer OpenDrawer `
  --episode-indices 0 1
"""

import argparse
import time
from pathlib import Path


def _episode_index_for_action(args, action_idx):
    if not args.episode_indices:
        return 0
    if len(args.episode_indices) == 1:
        return args.episode_indices[0]
    return args.episode_indices[action_idx]


def _target_fixture_ref_for_action(args, action_idx):
    if not args.target_fixture_refs:
        return None
    value = args.target_fixture_refs[action_idx]
    if value in ("-", "none", "None", "null"):
        return None
    return value


def _load_action(args, action_idx, task, scene_model, summary, atomic_root):
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_utils import (
        resolve_instruction_from_ep_meta,
    )
    from robocasa.scripts.dataset_scripts.replay_atomic_on_scene import (
        _choose_target_body,
        _dataset_dir,
        _episode_number,
        _load_demo_model_from_episode_xml,
        _local_patch_matches,
        _match_joints,
        _print_match_summary,
        _select_episode,
        _state_qpos,
    )

    episode_index = _episode_index_for_action(args, action_idx)
    target_fixture_ref = _target_fixture_ref_for_action(args, action_idx)
    episode_id, episode = _select_episode(
        summary=summary,
        task=task,
        layout=args.layout,
        episode_index=episode_index,
    )
    dataset = _dataset_dir(atomic_root, episode)
    episode_num = _episode_number(episode["episode"])

    print("")
    print(f"[{action_idx + 1}/{len(args.tasks)}] Task: {task}")
    print(f"Episode index: {episode_index}")
    print(f"Episode id: {episode_id}")
    print(f"Dataset: {dataset}")
    if target_fixture_ref is not None:
        print(f"Explicit target fixture ref: {target_fixture_ref}")

    states = LU.get_episode_states(dataset, episode_num)
    model_xml = LU.get_episode_model_xml(dataset, episode_num)
    ep_meta = LU.get_episode_meta(dataset, episode_num)
    instruction = resolve_instruction_from_ep_meta(ep_meta)
    if instruction:
        print(f"Instruction: {instruction}")
    print(f"Frames: {len(states)}")

    print("Loading demo episode model...")
    demo_model = _load_demo_model_from_episode_xml(dataset, model_xml)
    first_demo_qpos = _state_qpos(states[0], demo_model)

    match_info = _match_joints(
        demo_model,
        scene_model,
        include_freejoints=args.include_freejoints,
        skip_mobile_base_pose=args.replay_adapter,
    )
    if args.local_state_replay:
        replay_match_info = _local_patch_matches(
            match_info=match_info,
            demo_model=demo_model,
            ep_meta=ep_meta,
            target_fixture_ref=target_fixture_ref,
            include_objects=args.local_state_include_objects,
        )
    else:
        replay_match_info = match_info

    _print_match_summary(replay_match_info, args.print_names)
    if not replay_match_info["matches"]:
        raise RuntimeError(f"No matched joints found for action {action_idx + 1}: {task}")

    target_body = None
    if args.replay_adapter:
        target_body = _choose_target_body(
            demo_model=demo_model,
            scene_model=scene_model,
            ep_meta=ep_meta,
            explicit_ref=target_fixture_ref,
        )
        if target_body is None:
            print("Replay adapter target body: <none>; will use recorded absolute robot pose")
        else:
            print(f"Replay adapter target body: {target_body}")

    return {
        "task": task,
        "episode_id": episode_id,
        "demo_model": demo_model,
        "first_demo_qpos": first_demo_qpos,
        "states": states,
        "matches": replay_match_info["matches"],
        "target_body": target_body,
        "state_qpos": _state_qpos,
    }


def _start_action(args, action, scene_model, scene_data):
    import mujoco
    from robocasa.scripts.dataset_scripts.replay_atomic_on_scene import (
        _align_scene_robot_to_demo_start,
        _copy_matched_qpos,
        _zero_mobile_base_pose_qpos,
    )

    _copy_matched_qpos(action["first_demo_qpos"], scene_data, action["matches"])
    if args.replay_adapter:
        _zero_mobile_base_pose_qpos(scene_model, scene_data)
    mujoco.mj_forward(scene_model, scene_data)

    if args.replay_adapter:
        pos, quat = _align_scene_robot_to_demo_start(
            demo_model=action["demo_model"],
            scene_model=scene_model,
            scene_data=scene_data,
            demo_qpos=action["first_demo_qpos"],
            robot_root_body=args.robot_root_body,
            mobile_base_body=args.mobile_base_body,
            target_body=action["target_body"],
        )
        print(
            "Replay adapter aligned custom robot start: "
            f"{args.mobile_base_body} world pos={pos}, quat={quat}"
        )
    mujoco.mj_forward(scene_model, scene_data)


def replay_sequence_on_scene(args):
    import mujoco
    from robocasa.scripts.dataset_scripts.replay_atomic_on_scene import _load_summary

    summary = _load_summary(args.summary_json)
    atomic_root = Path(summary["summary"]["atomic_root"])

    print(f"Scene XML: {args.scene_xml}")
    print(f"Layout: {args.layout}")
    print(f"Tasks: {args.tasks}")
    print(f"Atomic root: {atomic_root}")
    print(f"Local-state replay: {args.local_state_replay}")

    scene_model = mujoco.MjModel.from_xml_path(str(args.scene_xml))
    scene_data = mujoco.MjData(scene_model)
    mujoco.mj_forward(scene_model, scene_data)

    print("")
    print("Preloading actions...")
    actions = [
        _load_action(
            args=args,
            action_idx=action_idx,
            task=task,
            scene_model=scene_model,
            summary=summary,
            atomic_root=atomic_root,
        )
        for action_idx, task in enumerate(args.tasks)
    ]
    print(f"Preloaded actions: {len(actions)}")

    if args.dry_run:
        for action in actions:
            _start_action(args, action, scene_model, scene_data)
        return

    import mujoco.viewer
    from robocasa.scripts.dataset_scripts.replay_atomic_on_scene import (
        _copy_matched_qpos,
        _zero_mobile_base_pose_qpos,
    )

    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    with mujoco.viewer.launch_passive(scene_model, scene_data) as viewer:
        for action_idx, action in enumerate(actions):
            print("")
            print(f"Playing [{action_idx + 1}/{len(actions)}]: {action['episode_id']}")
            _start_action(args, action, scene_model, scene_data)
            frame_indices = list(range(0, len(action["states"]), args.frame_stride))
            if args.extend_last_per_action > 0:
                frame_indices.extend(
                    [len(action["states"]) - 1] * args.extend_last_per_action
                )

            for frame_idx in frame_indices:
                if not viewer.is_running():
                    return
                start = time.time()
                demo_qpos = action["state_qpos"](
                    action["states"][frame_idx], action["demo_model"]
                )
                _copy_matched_qpos(demo_qpos, scene_data, action["matches"])
                if args.replay_adapter:
                    _zero_mobile_base_pose_qpos(scene_model, scene_data)
                mujoco.mj_forward(scene_model, scene_data)
                viewer.sync()
                if frame_dt > 0:
                    elapsed = time.time() - start
                    time.sleep(max(0.0, frame_dt - elapsed))


def parse_args():
    from robocasa.scripts.dataset_scripts.replay_atomic_on_scene import (
        _default_summary_json,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Replay a sequence of RoboCasa atomic demos on one custom scene while "
            "preserving scene state between actions."
        )
    )
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--layout", type=int, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument(
        "--episode-indices",
        nargs="*",
        type=int,
        default=None,
        help=(
            "episode-index for each task. If omitted, every task uses 0. If one "
            "value is provided, it is reused for every task."
        ),
    )
    parser.add_argument(
        "--target-fixture-refs",
        nargs="*",
        default=None,
        help=(
            "optional explicit target fixture ref for each task. Use '-' to let "
            "that action fall back to ep_meta.fixture_refs."
        ),
    )
    parser.add_argument("--summary-json", type=Path, default=_default_summary_json())
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--extend-last-per-action", type=int, default=20)
    parser.add_argument(
        "--include-freejoints",
        action="store_true",
        help="also map 7-DoF free joints when names match",
    )
    parser.add_argument(
        "--robot-root-body",
        type=str,
        default="robot0_base",
        help="robot root body moved by the replay adapter",
    )
    parser.add_argument(
        "--mobile-base-body",
        type=str,
        default="mobilebase0_base",
        help="mobile base body whose recorded pose is used by the replay adapter",
    )
    parser.add_argument(
        "--no-replay-adapter",
        dest="replay_adapter",
        action="store_false",
        help="disable target-relative robot alignment and direct mobile-base qpos skipping",
    )
    parser.set_defaults(replay_adapter=True)
    parser.add_argument(
        "--full-state-replay",
        dest="local_state_replay",
        action="store_false",
        help="copy every matched qpos instead of only robot plus target fixture joints",
    )
    parser.set_defaults(local_state_replay=True)
    parser.add_argument(
        "--local-state-include-objects",
        action="store_true",
        help="also allow object joints from episode metadata in local-state replay",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load all actions and print matching diagnostics without opening a viewer",
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
    if args.extend_last_per_action < 0:
        parser.error("--extend-last-per-action must be >= 0")
    if args.episode_indices and len(args.episode_indices) not in (1, len(args.tasks)):
        parser.error("--episode-indices must contain either 1 value or one value per task")
    if args.target_fixture_refs and len(args.target_fixture_refs) != len(args.tasks):
        parser.error("--target-fixture-refs must contain one value per task")
    return args


def main():
    replay_sequence_on_scene(parse_args())


if __name__ == "__main__":
    main()
