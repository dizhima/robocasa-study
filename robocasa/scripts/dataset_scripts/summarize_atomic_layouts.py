"""Summarize available RoboCasa atomic episodes by task, layout, and style.

Example usage:
uv run python -m robocasa.scripts.dataset_scripts.summarize_atomic_layouts
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm


def _default_atomic_root():
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "datasets" / "v1.0" / "pretrain" / "atomic"


def _counter_sort_key(item):
    key, _ = item
    if key == "unknown":
        return (1, 0)
    try:
        return (0, int(key))
    except ValueError:
        return (0, key)


def _sort_counter(counter):
    return {key: value for key, value in sorted(counter.items(), key=_counter_sort_key)}


def _sort_nested_counter(counter_by_key):
    return {
        key: _sort_counter(counter)
        for key, counter in sorted(counter_by_key.items(), key=_counter_sort_key)
    }


def _sort_bucket_map(bucket_by_key):
    return {
        key: {
            "count": bucket["count"],
            "episode_ids": sorted(bucket["episode_ids"]),
        }
        for key, bucket in sorted(bucket_by_key.items(), key=_counter_sort_key)
    }


def _sort_nested_bucket_map(bucket_by_key):
    return {
        key: _sort_bucket_map(bucket)
        for key, bucket in sorted(bucket_by_key.items(), key=_counter_sort_key)
    }


def _sort_task_counts(counter):
    return {key: value for key, value in sorted(counter.items())}


def _episode_meta_paths(atomic_root):
    paths = set(atomic_root.glob("*/*/lerobot/extras/episode_*/ep_meta.json"))
    paths.update(atomic_root.glob("*/lerobot/extras/episode_*/ep_meta.json"))
    return sorted(paths)


def _task_and_dataset_date(atomic_root, meta_path):
    rel_parts = meta_path.relative_to(atomic_root).parts
    lerobot_idx = rel_parts.index("lerobot")
    if lerobot_idx == 2:
        return rel_parts[0], rel_parts[1]
    if lerobot_idx == 1:
        return atomic_root.name, rel_parts[0]
    raise ValueError(f"Unexpected ep_meta path under {atomic_root}: {meta_path}")


def _parse_scalar(value):
    value = value.strip()
    if value == "null":
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return int(value)


def _read_layout_style(meta_path):
    with meta_path.open("r", encoding="utf-8") as f:
        head = f.read(4096)

    layout_match = re.search(r'"layout_id"\s*:\s*(null|-?\d+|"[^"]+")', head)
    style_match = re.search(r'"style_id"\s*:\s*(null|-?\d+|"[^"]+")', head)
    if layout_match and style_match:
        return _parse_scalar(layout_match.group(1)), _parse_scalar(style_match.group(1))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("layout_id"), meta.get("style_id")


def _relative_path(path, root):
    return path.relative_to(root).as_posix()


def _episode_record(atomic_root, meta_path, task_name, dataset_date, layout, style):
    episode_name = meta_path.parent.name
    episode_id = f"{task_name}/{dataset_date}/{episode_name}"
    lerobot_root = meta_path.parents[2]
    return episode_id, {
        "task": task_name,
        "dataset_date": dataset_date,
        "episode": episode_name,
        "layout_id": layout,
        "style_id": style,
        "paths": {
            "ep_meta": _relative_path(meta_path, atomic_root),
            "states": _relative_path(meta_path.parent / "states.npz", atomic_root),
            "model": _relative_path(meta_path.parent / "model.xml.gz", atomic_root),
            "parquet": _relative_path(
                lerobot_root / "data" / "chunk-000" / f"{episode_name}.parquet",
                atomic_root,
            ),
        },
    }


def _add_bucket_episode(bucket_by_key, key, episode_id):
    bucket = bucket_by_key[key]
    bucket["count"] += 1
    bucket["episode_ids"].append(episode_id)


def summarize_atomic_layouts(atomic_root):
    meta_paths = _episode_meta_paths(atomic_root)

    tasks = defaultdict(
        lambda: {
            "total_episodes": 0,
            "unknown_layout_episodes": 0,
            "unknown_style_episodes": 0,
            "datasets": Counter(),
            "layouts": defaultdict(lambda: {"count": 0, "episode_ids": []}),
            "styles": defaultdict(lambda: {"count": 0, "episode_ids": []}),
            "layout_styles": defaultdict(
                lambda: defaultdict(lambda: {"count": 0, "episode_ids": []})
            ),
        }
    )
    by_layout = defaultdict(
        lambda: {
            "count": 0,
            "tasks": Counter(),
            "episode_ids": [],
        }
    )
    episodes = {}
    errors = []

    for meta_path in tqdm(meta_paths, desc="Reading ep_meta.json", unit="episode"):
        task_name, dataset_date = _task_and_dataset_date(atomic_root, meta_path)

        try:
            layout, style = _read_layout_style(meta_path)
        except Exception as exc:
            errors.append(
                {
                    "path": str(meta_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        layout_key = "unknown" if layout is None else str(layout)
        style_key = "unknown" if style is None else str(style)
        episode_id, episode = _episode_record(
            atomic_root, meta_path, task_name, dataset_date, layout, style
        )
        episodes[episode_id] = episode

        task_stats = tasks[task_name]
        task_stats["total_episodes"] += 1
        task_stats["datasets"][dataset_date] += 1
        _add_bucket_episode(task_stats["layouts"], layout_key, episode_id)
        _add_bucket_episode(task_stats["styles"], style_key, episode_id)
        _add_bucket_episode(task_stats["layout_styles"][layout_key], style_key, episode_id)

        by_layout[layout_key]["count"] += 1
        by_layout[layout_key]["tasks"][task_name] += 1
        by_layout[layout_key]["episode_ids"].append(episode_id)

        if layout_key == "unknown":
            task_stats["unknown_layout_episodes"] += 1
        if style_key == "unknown":
            task_stats["unknown_style_episodes"] += 1

    ordered_tasks = {}
    for task_name in sorted(tasks):
        task_stats = tasks[task_name]
        ordered_tasks[task_name] = {
            "total_episodes": task_stats["total_episodes"],
            "unknown_layout_episodes": task_stats["unknown_layout_episodes"],
            "unknown_style_episodes": task_stats["unknown_style_episodes"],
            "datasets": _sort_counter(task_stats["datasets"]),
            "layouts": _sort_bucket_map(task_stats["layouts"]),
            "styles": _sort_bucket_map(task_stats["styles"]),
            "layout_styles": _sort_nested_bucket_map(task_stats["layout_styles"]),
        }

    ordered_by_layout = {
        key: {
            "count": bucket["count"],
            "tasks": _sort_task_counts(bucket["tasks"]),
            "episode_ids": sorted(bucket["episode_ids"]),
        }
        for key, bucket in sorted(by_layout.items(), key=_counter_sort_key)
    }

    return {
        "summary": {
            "atomic_root": str(atomic_root),
            "total_tasks": len(ordered_tasks),
            "total_episodes": sum(task["total_episodes"] for task in ordered_tasks.values()),
            "total_ep_meta_files": len(meta_paths),
            "read_errors": len(errors),
            "unknown_layout_episodes": sum(
                task["unknown_layout_episodes"] for task in ordered_tasks.values()
            ),
            "unknown_style_episodes": sum(
                task["unknown_style_episodes"] for task in ordered_tasks.values()
            ),
        },
        "by_task": ordered_tasks,
        "by_layout": ordered_by_layout,
        "episodes": {key: episodes[key] for key in sorted(episodes)},
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize layout/style distribution for RoboCasa atomic task datasets."
    )
    parser.add_argument(
        "--atomic-root",
        type=Path,
        default=_default_atomic_root(),
        help="Path to datasets/v1.0/pretrain/atomic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to atomic_root/atomic_layout_summary.json.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    args = parser.parse_args()

    atomic_root = args.atomic_root
    if not atomic_root.exists():
        raise FileNotFoundError(f"Atomic dataset root does not exist: {atomic_root}")

    output_path = args.output or (atomic_root / "atomic_layout_summary.json")
    summary = summarize_atomic_layouts(atomic_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=args.indent, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(
        "Tasks: {total_tasks}, episodes: {total_episodes}, read errors: {read_errors}".format(
            **summary["summary"]
        )
    )


if __name__ == "__main__":
    main()
