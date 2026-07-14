# Kitchen Layout Cabinet Complexity Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal-only command that ranks kitchen layout YAML files by the number of enabled cabinet fixtures.

**Architecture:** A focused script owns YAML traversal, per-layout aggregation, deterministic sorting, terminal formatting, and CLI validation. Unit tests call its pure functions and `main()` directly so behavior is verified without constructing RoboCasa simulation environments.

**Tech Stack:** Python 3, PyYAML, argparse, pathlib, collections.Counter, pytest

---

## File Structure

- Create `robocasa/scripts/analyze_kitchen_layout_complexity.py`: cabinet type policy, recursive traversal, directory analysis, sorting, formatting, and CLI entry point.
- Create `tests/test_analyze_kitchen_layout_complexity.py`: focused unit and CLI tests using pytest temporary directories.

### Task 1: Count and rank cabinet fixtures

**Files:**
- Create: `robocasa/scripts/analyze_kitchen_layout_complexity.py`
- Create: `tests/test_analyze_kitchen_layout_complexity.py`

- [ ] **Step 1: Write failing tests for recursive counting and ranking**

```python
from robocasa.scripts.analyze_kitchen_layout_complexity import (
    count_cabinets,
    layout_sort_key,
)


def test_count_cabinets_recurses_and_ignores_non_cabinets():
    layout = {
        "main_group": {
            "bottom_row": [
                {"name": "counter", "type": "counter"},
                {"name": "base", "type": "hinge_cabinet"},
                {"name": "disabled", "type": "single_cabinet", "enable": False},
            ],
            "top_row": [
                {"name": "open", "type": "open_cabinet"},
                {"name": "panel", "type": "panel_cabinet"},
                {"name": "housing", "type": "housing_cabinet"},
                {"name": "group", "type": "stack"},
            ],
        },
        "room": {"walls": [{"name": "wall", "type": "wall"}]},
    }

    assert count_cabinets(layout) == {
        "hinge_cabinet": 1,
        "open_cabinet": 1,
        "panel_cabinet": 1,
        "housing_cabinet": 1,
    }


def test_layout_sort_key_uses_total_then_numeric_layout_id():
    results = [
        {"name": "layout010", "total": 2},
        {"name": "layout002", "total": 2},
        {"name": "custom", "total": 1},
    ]

    assert [item["name"] for item in sorted(results, key=layout_sort_key)] == [
        "custom",
        "layout002",
        "layout010",
    ]
```

- [ ] **Step 2: Run tests and verify they fail because the module is absent**

Run: `uv run pytest tests/test_analyze_kitchen_layout_complexity.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'robocasa.scripts.analyze_kitchen_layout_complexity'`.

- [ ] **Step 3: Implement cabinet traversal and deterministic sorting**

```python
import re
from collections import Counter


CABINET_TYPES = frozenset(
    {
        "hinge_cabinet",
        "single_cabinet",
        "open_cabinet",
        "panel_cabinet",
        "housing_cabinet",
    }
)


def count_cabinets(node):
    counts = Counter()
    if isinstance(node, dict):
        fixture_type = node.get("type")
        if fixture_type in CABINET_TYPES and node.get("enable", True) is not False:
            counts[fixture_type] += 1
        for value in node.values():
            counts.update(count_cabinets(value))
    elif isinstance(node, list):
        for value in node:
            counts.update(count_cabinets(value))
    return counts


def layout_sort_key(result):
    match = re.search(r"(\d+)$", result["name"])
    numeric_id = int(match.group(1)) if match else float("inf")
    return result["total"], numeric_id, result["name"]
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `uv run pytest tests/test_analyze_kitchen_layout_complexity.py -v`

Expected: `2 passed`.

- [ ] **Step 5: Commit the core behavior**

```bash
git add robocasa/scripts/analyze_kitchen_layout_complexity.py tests/test_analyze_kitchen_layout_complexity.py
git commit -m "feat: count cabinets in kitchen layouts"
```

### Task 2: Analyze directories and print terminal rankings

**Files:**
- Modify: `robocasa/scripts/analyze_kitchen_layout_complexity.py`
- Modify: `tests/test_analyze_kitchen_layout_complexity.py`

- [ ] **Step 1: Write failing CLI and error-isolation tests**

Append these imports and tests:

```python
from pathlib import Path

from robocasa.scripts.analyze_kitchen_layout_complexity import main


def _write_layout(path, fixture_types):
    fixtures = "\n".join(
        f"    - name: fixture_{index}\n      type: {fixture_type}"
        for index, fixture_type in enumerate(fixture_types)
    )
    path.write_text(f"main_group:\n  fixtures:\n{fixtures}\n", encoding="utf-8")


def test_main_prints_ranked_results_and_honors_top(tmp_path, capsys):
    _write_layout(tmp_path / "layout010.yaml", ["hinge_cabinet", "single_cabinet"])
    _write_layout(tmp_path / "layout002.yaml", ["open_cabinet"])

    exit_code = main([str(tmp_path), "--top", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "layout002" in captured.out
    assert "open_cabinet=1" in captured.out
    assert "layout010" not in captured.out
    assert captured.err == ""


def test_main_reports_bad_yaml_but_prints_valid_results(tmp_path, capsys):
    _write_layout(tmp_path / "layout001.yaml", ["hinge_cabinet"])
    (tmp_path / "layout002.yaml").write_text("broken: [", encoding="utf-8")

    exit_code = main([str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "layout001" in captured.out
    assert "layout002.yaml" in captured.err


def test_main_rejects_directory_without_yaml(tmp_path, capsys):
    exit_code = main([str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No YAML files found" in captured.err
```

- [ ] **Step 2: Run the new tests and verify missing CLI behavior fails**

Run: `uv run pytest tests/test_analyze_kitchen_layout_complexity.py -v`

Expected: import fails because `main` is not defined.

- [ ] **Step 3: Implement directory analysis, formatting, and CLI validation**

Add these imports and functions, plus the module entry point:

```python
import argparse
import sys
from pathlib import Path

import yaml


TYPE_DISPLAY_ORDER = tuple(sorted(CABINET_TYPES))


def default_layout_dir():
    return Path(__file__).resolve().parents[1] / "models" / "assets" / "scenes" / "kitchen_layouts" / "train"


def analyze_directory(layout_dir):
    paths = sorted([*layout_dir.glob("*.yaml"), *layout_dir.glob("*.yml")])
    if not paths:
        raise ValueError(f"No YAML files found in {layout_dir}")

    results = []
    errors = []
    for path in paths:
        try:
            layout = yaml.safe_load(path.read_text(encoding="utf-8"))
            counts = count_cabinets(layout)
            results.append(
                {"name": path.stem, "path": path, "total": sum(counts.values()), "counts": counts}
            )
        except (OSError, yaml.YAMLError) as exc:
            errors.append((path, exc))
    return sorted(results, key=layout_sort_key), errors


def format_result(rank, result):
    details = " ".join(
        f"{fixture_type}={result['counts'][fixture_type]}"
        for fixture_type in TYPE_DISPLAY_ORDER
        if result["counts"][fixture_type]
    )
    return f"{rank:>3}. {result['name']:<12} total={result['total']:<3} {details}".rstrip()


def build_parser():
    parser = argparse.ArgumentParser(description="Rank kitchen layouts by cabinet count.")
    parser.add_argument("layout_dir", nargs="?", type=Path, default=default_layout_dir())
    parser.add_argument("--top", type=int, help="show only the N simplest layouts")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.top is not None and args.top < 1:
        print("error: --top must be at least 1", file=sys.stderr)
        return 1
    if not args.layout_dir.is_dir():
        print(f"error: layout directory does not exist: {args.layout_dir}", file=sys.stderr)
        return 1
    try:
        results, errors = analyze_directory(args.layout_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    selected = results[: args.top] if args.top is not None else results
    print("Kitchen layouts ranked by enabled cabinet count (simplest first)")
    for rank, result in enumerate(selected, start=1):
        print(format_result(rank, result))
    for path, exc in errors:
        print(f"error: {path.name}: {exc}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the entire focused test file**

Run: `uv run pytest tests/test_analyze_kitchen_layout_complexity.py -v`

Expected: `5 passed`.

- [ ] **Step 5: Commit the CLI behavior**

```bash
git add robocasa/scripts/analyze_kitchen_layout_complexity.py tests/test_analyze_kitchen_layout_complexity.py
git commit -m "feat: rank kitchen layouts by cabinet count"
```

### Task 3: Verify against the train layouts

**Files:**
- Verify: `robocasa/scripts/analyze_kitchen_layout_complexity.py`
- Verify: `tests/test_analyze_kitchen_layout_complexity.py`

- [ ] **Step 1: Run focused tests from a clean invocation**

Run: `uv run pytest tests/test_analyze_kitchen_layout_complexity.py -v`

Expected: all 5 tests pass.

- [ ] **Step 2: Run the tool against the default train directory**

Run: `uv run python -m robocasa.scripts.analyze_kitchen_layout_complexity --top 10`

Expected: exit code 0 and ten ranked layout rows in ascending cabinet count.

- [ ] **Step 3: Inspect repository changes for accidental edits**

Run: `git status --short`

Expected: only pre-existing user changes remain; the analyzer and tests are committed.
