import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from robocasa.scripts import export_kitchen_scene


def test_load_object_cfgs_resolves_portable_asset_path(monkeypatch, tmp_path):
    assets_root = tmp_path / "robocasa" / "models" / "assets"
    monkeypatch.setattr(export_kitchen_scene.robocasa.models, "assets_root", assets_root)
    objects_yaml = tmp_path / "objects.yaml"
    objects_yaml.write_text(
        """objects:
- type: object
  name: apple_1
  obj_groups: robocasa/models/assets/objects/objaverse/apple/apple_2/model.xml
  placement:
    fixture: counter_main_group
""",
        encoding="utf-8",
    )

    configs = export_kitchen_scene._load_object_cfgs(objects_yaml)

    assert Path(configs[0]["obj_groups"]) == (
        assets_root / "objects" / "objaverse" / "apple" / "apple_2" / "model.xml"
    )


def test_make_walls_flat_only_flattens_wall_surfaces():
    make_walls_flat = getattr(export_kitchen_scene, "_make_walls_flat", None)
    assert callable(make_walls_flat), "_make_walls_flat must be implemented"

    xml = """
    <mujoco>
      <asset>
        <texture name="painted_wall" file="textures/walls/paint.png" />
        <texture name="red_brick" file="textures/masonry/red.png" />
        <texture name="tile" file="textures/backsplash/tile.png" />
        <texture name="floor_wood" file="textures/floor/wood.png" />
        <texture name="cabinet_wood" file="textures/cabinets/wood.png" />
        <material name="wall_mat" texture="painted_wall" />
        <material name="brick_mat" texture="red_brick" />
        <material name="tile_mat" texture="tile" />
        <material name="floor_mat" texture="floor_wood" />
        <material name="cabinet_mat" texture="cabinet_wood" />
      </asset>
      <worldbody>
        <geom name="main_wall" material="wall_mat" />
        <geom name="brick_surface" material="brick_mat" />
        <geom name="backsplash_panel" material="tile_mat" />
        <geom name="floor" material="floor_mat" />
        <geom name="cabinet" material="cabinet_mat" />
      </worldbody>
    </mujoco>
    """

    tree = ET.fromstring(make_walls_flat(xml))
    wall_rgba = "0.88 0.87 0.82 1"
    for material_name in ("wall_mat", "brick_mat", "tile_mat"):
        material = tree.find(f".//material[@name='{material_name}']")
        assert material is not None
        assert material.get("texture") is None
        assert material.get("rgba") == wall_rgba

    texture_names = {texture.get("name") for texture in tree.findall(".//texture")}
    assert texture_names == {"floor_wood", "cabinet_wood"}
    assert tree.find(".//material[@name='floor_mat']").get("texture") == "floor_wood"
    assert tree.find(".//material[@name='cabinet_mat']").get("texture") == "cabinet_wood"


def test_parse_args_accepts_flat_wall_independently(monkeypatch, tmp_path):
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
            "--flat-wall",
        ],
    )

    args = export_kitchen_scene.parse_args()

    assert args.flat_wall is True
    assert args.flat_floor is False


def test_flat_floor_and_flat_wall_transformations_compose():
    make_walls_flat = getattr(export_kitchen_scene, "_make_walls_flat", None)
    assert callable(make_walls_flat), "_make_walls_flat must be implemented"
    xml = """
    <mujoco>
      <asset>
        <texture name="wall_texture" file="textures/walls/paint.png" />
        <texture name="floor_texture" file="textures/floor/wood.png" />
        <texture name="cabinet_texture" file="textures/cabinets/wood.png" />
        <material name="wall_mat" texture="wall_texture" />
        <material name="floor_mat" texture="floor_texture" />
        <material name="cabinet_mat" texture="cabinet_texture" />
      </asset>
      <worldbody>
        <geom name="wall" material="wall_mat" />
        <geom name="floor" material="floor_mat" />
        <geom name="cabinet" material="cabinet_mat" />
      </worldbody>
    </mujoco>
    """

    flattened = export_kitchen_scene._make_floor_flat(xml)
    tree = ET.fromstring(make_walls_flat(flattened))

    assert tree.find(".//material[@name='wall_mat']").get("texture") is None
    assert tree.find(".//material[@name='floor_mat']").get("texture") is None
    assert (
        tree.find(".//material[@name='cabinet_mat']").get("texture")
        == "cabinet_texture"
    )


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


def test_parse_args_accepts_primary_pandaomron_auto_placement(monkeypatch, tmp_path):
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
            "--auto-place-primary-pandaomron",
        ],
    )

    args = export_kitchen_scene.parse_args()

    assert args.auto_place_primary_pandaomron is True


def test_auto_place_primary_pandaomron_updates_robot_base(monkeypatch):
    args = SimpleNamespace(extra_robot_footprint=0.8, extra_robot_margin=0.15)
    monkeypatch.setattr(
        export_kitchen_scene, "_layout_floor_rects", lambda _: [(0.0, 0.0, 2.0, 2.0)]
    )
    monkeypatch.setattr(export_kitchen_scene, "_fixture_obstacle_rects", lambda _: [])

    xml = """
    <mujoco>
      <worldbody>
        <body name="robot0_base" pos="9 9 0" quat="1 0 0 0" />
      </worldbody>
    </mujoco>
    """
    tree = ET.fromstring(export_kitchen_scene._auto_place_primary_pandaomron(xml, args))
    robot = tree.find("./worldbody/body[@name='robot0_base']")

    assert robot is not None
    assert robot.get("pos") != "9 9 0"
    assert robot.get("quat") != "1 0 0 0"


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
