# Flat Countertop Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `--flat-countertop` export option that replaces only RoboCasa countertop work-surface textures with a solid color.

**Architecture:** Reuse the existing targeted-material transform pattern in `export_kitchen_scene.py`. Match countertop materials by exact RoboCasa material names (`counter_top` or suffix `_counter_top`) so counter bases and stove appliance materials remain unchanged.

**Tech Stack:** Python, `xml.etree.ElementTree`, pytest, RoboCasa MJCF exporter script.

---

## File Structure

- Modify: `robocasa/scripts/export_kitchen_scene.py`
  - Add the `--flat-countertop` CLI option.
  - Add `_make_countertops_flat(xml)` next to `_make_floor_flat()` and `_make_walls_flat()`.
  - Call `_make_countertops_flat()` in `main()` only in the targeted-material branch, after floor/wall handling.
- Modify: `tests/test_export_kitchen_scene.py`
  - Add tests for countertop material matching, shared texture cleanup, parser behavior, and composition with floor/wall.

### Task 1: Add Failing Countertop Tests

**Files:**
- Modify: `tests/test_export_kitchen_scene.py`

- [ ] **Step 1: Add helper-behavior test**

Append this test to `tests/test_export_kitchen_scene.py`:

```python
def test_make_countertops_flat_only_flattens_countertop_materials():
    make_countertops_flat = getattr(export_kitchen_scene, "_make_countertops_flat", None)
    assert callable(make_countertops_flat), "_make_countertops_flat must be implemented"

    xml = """
    <mujoco>
      <asset>
        <texture name="countertop_only" file="textures/counters/stone.png" />
        <texture name="shared_surface" file="textures/shared/surface.png" />
        <texture name="base_texture" file="textures/counters/base.png" />
        <texture name="stovetop_texture" file="textures/stove/glass.png" />
        <texture name="burner_texture" file="textures/stove/burner.png" />
        <material name="counter_top" texture="countertop_only" />
        <material name="island_counter_top" texture="shared_surface" />
        <material name="stove_counter_top" texture="countertop_only" />
        <material name="counter_base" texture="base_texture" />
        <material name="stovetop_glass" texture="stovetop_texture" />
        <material name="burner" texture="burner_texture" />
        <material name="decorative_material" texture="shared_surface" />
      </asset>
      <worldbody>
        <geom name="main_counter_surface" material="counter_top" />
        <geom name="island_surface" material="island_counter_top" />
        <geom name="stove_counter_surface" material="stove_counter_top" />
        <geom name="counter_base" material="counter_base" />
        <geom name="stovetop" material="stovetop_glass" />
        <geom name="burner" material="burner" />
        <geom name="decor" material="decorative_material" />
      </worldbody>
    </mujoco>
    """

    tree = ET.fromstring(make_countertops_flat(xml))
    countertop_rgba = "0.72 0.72 0.68 1"

    for material_name in ("counter_top", "island_counter_top", "stove_counter_top"):
        material = tree.find(f".//material[@name='{material_name}']")
        assert material is not None
        assert material.get("texture") is None
        assert material.get("rgba") == countertop_rgba
        assert material.get("reflectance") == "0.05"
        assert material.get("shininess") == "0.05"

    for material_name, texture_name in (
        ("counter_base", "base_texture"),
        ("stovetop_glass", "stovetop_texture"),
        ("burner", "burner_texture"),
        ("decorative_material", "shared_surface"),
    ):
        material = tree.find(f".//material[@name='{material_name}']")
        assert material is not None
        assert material.get("texture") == texture_name

    for geom_name in ("main_counter_surface", "island_surface", "stove_counter_surface"):
        geom = tree.find(f".//geom[@name='{geom_name}']")
        assert geom is not None
        assert geom.get("rgba") == countertop_rgba

    texture_names = {texture.get("name") for texture in tree.findall(".//texture")}
    assert "countertop_only" not in texture_names
    assert "shared_surface" in texture_names
    assert {"base_texture", "stovetop_texture", "burner_texture"}.issubset(texture_names)
```

- [ ] **Step 2: Add parser test**

Append this parser test:

```python
def test_parse_args_accepts_flat_countertop_independently(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_kitchen_scene",
            "--layout",
            "0",
            "--style",
            "0",
            "--output",
            str(tmp_path / "scene.xml"),
            "--flat-countertop",
        ],
    )

    args = export_kitchen_scene.parse_args()

    assert args.flat_countertop is True
    assert args.flat_floor is False
    assert args.flat_wall is False
```

- [ ] **Step 3: Add composition test**

Append this composition test:

```python
def test_flat_floor_flat_wall_and_flat_countertop_transformations_compose():
    make_walls_flat = getattr(export_kitchen_scene, "_make_walls_flat", None)
    make_countertops_flat = getattr(export_kitchen_scene, "_make_countertops_flat", None)
    assert callable(make_walls_flat), "_make_walls_flat must be implemented"
    assert callable(make_countertops_flat), "_make_countertops_flat must be implemented"

    xml = """
    <mujoco>
      <asset>
        <texture name="wall_texture" file="textures/walls/paint.png" />
        <texture name="floor_texture" file="textures/floor/wood.png" />
        <texture name="counter_texture" file="textures/counters/stone.png" />
        <texture name="cabinet_texture" file="textures/cabinets/wood.png" />
        <material name="wall_mat" texture="wall_texture" />
        <material name="floor_mat" texture="floor_texture" />
        <material name="island_counter_top" texture="counter_texture" />
        <material name="cabinet_mat" texture="cabinet_texture" />
      </asset>
      <worldbody>
        <geom name="wall" material="wall_mat" />
        <geom name="floor" material="floor_mat" />
        <geom name="island_surface" material="island_counter_top" />
        <geom name="cabinet" material="cabinet_mat" />
      </worldbody>
    </mujoco>
    """

    flattened = export_kitchen_scene._make_floor_flat(xml)
    flattened = make_walls_flat(flattened)
    tree = ET.fromstring(make_countertops_flat(flattened))

    assert tree.find(".//material[@name='wall_mat']").get("texture") is None
    assert tree.find(".//material[@name='floor_mat']").get("texture") is None
    assert tree.find(".//material[@name='island_counter_top']").get("texture") is None
    assert (
        tree.find(".//material[@name='cabinet_mat']").get("texture")
        == "cabinet_texture"
    )
```

- [ ] **Step 4: Run new tests and verify red**

Run:

```bash
uv run pytest tests/test_export_kitchen_scene.py::test_make_countertops_flat_only_flattens_countertop_materials tests/test_export_kitchen_scene.py::test_parse_args_accepts_flat_countertop_independently tests/test_export_kitchen_scene.py::test_flat_floor_flat_wall_and_flat_countertop_transformations_compose -v
```

Expected: FAIL because `_make_countertops_flat` and `--flat-countertop` do not exist yet.

### Task 2: Implement Countertop Flattening

**Files:**
- Modify: `robocasa/scripts/export_kitchen_scene.py`

- [ ] **Step 1: Add CLI option**

Add this parser argument immediately after `--flat-wall`:

```python
    parser.add_argument(
        "--flat-countertop",
        action="store_true",
        help="replace countertop textures with a flat material before web asset export",
    )
```

- [ ] **Step 2: Add `_make_countertops_flat`**

Add this function immediately after `_make_walls_flat(xml)`:

```python
def _make_countertops_flat(xml):
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    countertop_rgba = "0.72 0.72 0.68 1"
    countertop_material_names = set()
    countertop_texture_names = set()

    for material in tree.findall(".//material"):
        name = material.get("name", "")
        name_lower = name.lower()
        if name_lower != "counter_top" and not name_lower.endswith("_counter_top"):
            continue

        countertop_material_names.add(name)
        texture = material.get("texture", "")
        if texture:
            countertop_texture_names.add(texture)
        material.attrib.pop("texture", None)
        material.set("rgba", countertop_rgba)
        material.set("reflectance", "0.05")
        material.set("shininess", "0.05")

    for geom in tree.findall(".//geom"):
        if geom.get("material") in countertop_material_names:
            geom.set("rgba", countertop_rgba)

    remaining_texture_names = {
        material.get("texture")
        for material in tree.findall(".//material")
        if material.get("texture")
    }

    if asset is not None:
        for texture in list(asset.findall("texture")):
            name = texture.get("name", "")
            if name in countertop_texture_names and name not in remaining_texture_names:
                asset.remove(texture)

    return ET.tostring(tree, encoding="unicode")
```

- [ ] **Step 3: Wire `main()`**

Add this targeted transform after wall handling:

```python
        if args.flat_countertop:
            xml = _make_countertops_flat(xml)
```

- [ ] **Step 4: Run focused tests and verify green**

Run:

```bash
uv run pytest tests/test_export_kitchen_scene.py -v
```

Expected: all tests in `tests/test_export_kitchen_scene.py` PASS.

### Task 3: Final Verification

**Files:**
- Read: `robocasa/scripts/export_kitchen_scene.py`
- Read: `tests/test_export_kitchen_scene.py`

- [ ] **Step 1: Compile modified Python files**

Run:

```bash
uv run python -m py_compile robocasa/scripts/export_kitchen_scene.py tests/test_export_kitchen_scene.py
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 2: Inspect scoped git status**

Run:

```bash
git status --short -- robocasa/scripts/export_kitchen_scene.py tests/test_export_kitchen_scene.py docs/superpowers/plans/2026-07-15-flat-countertop-export.md
```

Expected: only the exporter, test file, and this plan are listed for this feature.

## Self-Review

- Spec coverage: the plan covers the new CLI option, exact material-name matching, preservation of stove appliance and counter-base materials, shared texture retention, composition with floor/wall, and precedence through `main()` targeted branch.
- Placeholder scan: no `TBD`, `TODO`, or vague implementation instructions remain.
- Type consistency: the plan consistently uses `_make_countertops_flat(xml)`, `args.flat_countertop`, `countertop_rgba`, and ElementTree XML strings.
