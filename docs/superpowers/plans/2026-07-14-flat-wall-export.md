# Flat Wall Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent, composable `--flat-wall` export option that replaces wall, brick, and backsplash textures with one solid color.

**Architecture:** Keep the transformation local to `export_kitchen_scene.py`, parallel to `_make_floor_flat`. Use small inline MJCF fixtures to test actual ElementTree output without constructing a RoboCasa environment.

**Tech Stack:** Python, argparse, xml.etree.ElementTree, pytest

---

### Task 1: Wall material transformation

**Files:**
- Create: `tests/test_export_kitchen_scene.py`
- Modify: `robocasa/scripts/export_kitchen_scene.py`

- [ ] **Step 1: Write the failing transformation test**

Create an inline MJCF containing wall, brick, backsplash, floor, and cabinet materials. Call `_make_walls_flat`, then assert the first three lose their texture references and use `0.88 0.87 0.82 1`, their texture assets are removed, and floor/cabinet assets remain.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_export_kitchen_scene.py::test_make_walls_flat_only_flattens_wall_surfaces -v`
Expected: FAIL because `_make_walls_flat` does not exist.

- [ ] **Step 3: Implement the minimal XML transformation**

Add `_make_walls_flat(xml)` beside `_make_floor_flat`. Match `wall`, `brick`, or `backsplash` in material names, texture names/files, and geom names/material references; clear matched texture references, apply the neutral wall color and low-gloss properties, and remove matched texture assets.

- [ ] **Step 4: Run the focused test**

Run the command from Step 2.
Expected: PASS.

### Task 2: CLI flag and composition

**Files:**
- Modify: `tests/test_export_kitchen_scene.py`
- Modify: `robocasa/scripts/export_kitchen_scene.py`

- [ ] **Step 1: Write failing CLI and composition tests**

Patch `sys.argv` with required arguments plus `--flat-wall` and assert `parse_args().flat_wall` is true. Apply `_make_floor_flat` followed by `_make_walls_flat` to inline MJCF and assert both targeted materials are texture-free while cabinet material is unchanged.

- [ ] **Step 2: Run the tests to verify failure**

Run: `uv run pytest tests/test_export_kitchen_scene.py -v`
Expected: FAIL because argparse rejects `--flat-wall` and `main()` does not compose it with floor flattening.

- [ ] **Step 3: Add and wire the flag**

Add `parser.add_argument("--flat-wall", action="store_true", ...)`. Preserve the precedence of `--styled-materials` and `--flat-materials`; otherwise invoke `_make_floor_flat` and `_make_walls_flat` with separate conditional statements so both targeted flags compose.

- [ ] **Step 4: Run focused tests and syntax verification**

Run: `uv run pytest tests/test_export_kitchen_scene.py -v`
Expected: all tests PASS.

Run: `uv run python -m py_compile robocasa/scripts/export_kitchen_scene.py tests/test_export_kitchen_scene.py`
Expected: exit code 0.

- [ ] **Step 5: Inspect the final diff**

Run: `git diff -- robocasa/scripts/export_kitchen_scene.py tests/test_export_kitchen_scene.py`
Expected: only the flag, focused helper/wiring, and tests are present.
