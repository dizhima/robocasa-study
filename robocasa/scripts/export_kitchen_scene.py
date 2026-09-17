r"""Export generated RoboCasa kitchen scenes as MJCF XML.

Example usage:
uv run python -m robocasa.scripts.export_kitchen_scene `
  --layout-yaml robocasa/models/assets/scenes/custom_layouts/layout012_study.yaml `
  --objects-yaml robocasa/models/assets/scenes/custom_layouts/layout012_study_objects.yaml `
  --style 2 `
  --web-assets `
  --styled-materials `
  --extra-pandaomrons 2 `
  --mjb `
  --output exported_scenes/layout012_study.xml
"""

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import robosuite
import yaml
from robosuite.controllers import load_composite_controller_config
from robosuite.models.tasks import ManipulationTask

import robocasa
from robocasa.models.scenes import KitchenArena
from robocasa.models.scenes.scene_registry import get_layout_path


DEFAULT_STRETCH_MJCF = (
    Path.home()
    / "Documents"
    / "mujoco-skill-playground"
    / "assets"
    / "mujoco_menagerie"
    / "hello_robot_stretch"
    / "stretch.xml"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Kitchen")
    parser.add_argument("--layout", type=int)
    parser.add_argument(
        "--layout-yaml",
        type=Path,
        help="custom kitchen layout YAML path; use with --scene-only",
    )
    parser.add_argument("--style", type=int, required=True)
    parser.add_argument("--robot", type=str, default="PandaOmron")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objects-yaml",
        type=Path,
        help="YAML file containing RoboCasa object_cfgs to inject before env reset",
    )
    parser.add_argument(
        "--mjb",
        action="store_true",
        help="also compile the exported XML and write a same-name .mjb file",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="keep RoboCasa/robosuite asset paths relative instead of expanding them",
    )
    parser.add_argument(
        "--copy-assets",
        action="store_true",
        help="copy RoboCasa's models/assets directory next to the exported XML",
    )
    parser.add_argument(
        "--web-assets",
        action="store_true",
        help="copy RoboCasa/robosuite assets next to the XML and rewrite paths for browser use",
    )
    parser.add_argument(
        "--flat-floor",
        action="store_true",
        help="replace floor textures with a flat material before web asset export",
    )
    parser.add_argument(
        "--flat-wall",
        action="store_true",
        help="replace wall textures with a flat material before web asset export",
    )
    parser.add_argument(
        "--flat-countertop",
        action="store_true",
        help="replace countertop textures with a flat material before web asset export",
    )
    parser.add_argument(
        "--flat-materials",
        action="store_true",
        help="replace all textured materials with one flat solid color",
    )
    parser.add_argument(
        "--styled-materials",
        action="store_true",
        help="replace textures with a lightweight solid-color kitchen palette",
    )
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="export only the kitchen fixtures, without the robot model",
    )
    parser.add_argument(
        "--bake-robot-pose",
        action="store_true",
        help=(
            "write the robot base world pose and a baked_reset keyframe without "
            "changing joint refs"
        ),
    )
    parser.add_argument(
        "--replay-compatible-robot",
        action="store_true",
        help=(
            "export the PandaOmron root and mobile-base axes in the convention used by "
            "RoboCasa recorded states; use this for dataset qpos/state replay"
        ),
    )
    parser.add_argument(
        "--init-robot-base-ref",
        type=str,
        default="microwave",
        help="fixture name used by RoboCasa to place the robot base",
    )
    parser.add_argument(
        "--clutter-mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 disables clutter fixtures, 1 keeps them",
    )
    parser.add_argument(
        "--stretch-auto-place",
        action="store_true",
        help="add the Hello Robot Stretch MJCF and place it on the outer floor free space",
    )
    parser.add_argument(
        "--stretch-mjcf",
        type=Path,
        default=DEFAULT_STRETCH_MJCF,
        help="Stretch robot MJCF path used by --stretch-auto-place",
    )
    parser.add_argument(
        "--stretch-prefix",
        type=str,
        default="stretch0",
        help="name prefix for the inserted Stretch robot",
    )
    parser.add_argument(
        "--stretch-footprint",
        type=float,
        default=0.7,
        help="square footprint size, in meters, reserved for Stretch auto placement",
    )
    parser.add_argument(
        "--stretch-margin",
        type=float,
        default=0.2,
        help="minimum margin, in meters, between Stretch and kitchen fixtures",
    )
    parser.add_argument(
        "--extra-pandaomrons",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="clone and auto-place up to two additional PandaOmron robots",
    )
    parser.add_argument(
        "--auto-place-primary-pandaomron",
        action="store_true",
        help=(
            "auto-place the primary PandaOmron on free floor space instead of using "
            "the init robot base reference fixture"
        ),
    )
    parser.add_argument(
        "--extra-robot-footprint",
        type=float,
        default=0.8,
        help="square footprint size, in meters, reserved for each extra PandaOmron",
    )
    parser.add_argument(
        "--extra-robot-margin",
        type=float,
        default=0.15,
        help="minimum margin, in meters, around each extra PandaOmron",
    )
    args = parser.parse_args()
    if (args.layout is None) == (args.layout_yaml is None):
        parser.error("specify exactly one of --layout or --layout-yaml")
    if args.bake_robot_pose and args.scene_only:
        parser.error("--bake-robot-pose requires a robot export; omit --scene-only")
    if args.replay_compatible_robot and args.scene_only:
        parser.error("--replay-compatible-robot requires a robot export; omit --scene-only")
    if args.replay_compatible_robot and args.bake_robot_pose:
        parser.error("--replay-compatible-robot cannot be combined with --bake-robot-pose")
    if args.objects_yaml is not None and args.scene_only:
        parser.error("--objects-yaml requires a full environment export; omit --scene-only")
    if args.objects_yaml is not None and not args.objects_yaml.exists():
        parser.error(f"objects YAML does not exist: {args.objects_yaml}")
    if args.stretch_auto_place and not args.stretch_mjcf.exists():
        parser.error(f"Stretch MJCF does not exist: {args.stretch_mjcf}")
    if args.extra_pandaomrons and args.scene_only:
        parser.error("--extra-pandaomrons requires a robot export; omit --scene-only")
    if args.extra_pandaomrons and args.robot.lower() != "pandaomron":
        parser.error("--extra-pandaomrons requires --robot PandaOmron")
    if args.auto_place_primary_pandaomron and args.robot.lower() != "pandaomron":
        parser.error("--auto-place-primary-pandaomron requires --robot PandaOmron")
    if args.auto_place_primary_pandaomron and args.bake_robot_pose:
        parser.error(
            "--auto-place-primary-pandaomron cannot be combined with --bake-robot-pose"
        )
    if args.auto_place_primary_pandaomron and args.replay_compatible_robot:
        parser.error(
            "--auto-place-primary-pandaomron cannot be combined with "
            "--replay-compatible-robot"
        )
    if args.extra_pandaomrons and args.stretch_auto_place:
        parser.error("--extra-pandaomrons and --stretch-auto-place are mutually exclusive")
    if args.extra_robot_footprint <= 0:
        parser.error("--extra-robot-footprint must be greater than zero")
    if args.extra_robot_margin < 0:
        parser.error("--extra-robot-margin cannot be negative")
    return args


def _copy_assets(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _rewrite_asset_paths_for_web(xml, output_dir, extra_path_roots=None):
    tree = ET.fromstring(xml)
    compiler = tree.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    path_roots = [
        (
            Path(robocasa.__file__).parent / "models" / "assets",
            "robocasa_assets",
        ),
        (
            Path(robosuite.__file__).parent / "models" / "assets",
            "robosuite_assets",
        ),
    ]
    if extra_path_roots is not None:
        path_roots.extend(extra_path_roots)
    copied_files = set()

    for elem in tree.findall(".//*[@file]"):
        file_path = elem.get("file")
        if file_path is None:
            continue

        normalized = Path(file_path.replace("\\", "/"))
        for src_root, dst_name in path_roots:
            try:
                rel_path = normalized.resolve().relative_to(src_root.resolve())
            except (OSError, ValueError):
                continue
            elem.set("file", Path(dst_name, rel_path).as_posix())
            src_file = src_root / rel_path
            dst_file = output_dir / dst_name / rel_path
            copied_files.add((src_file, dst_file))
            break

    for src_file, dst_file in copied_files:
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    return ET.tostring(tree, encoding="unicode")


def _hide_helper_visuals(xml):
    tree = ET.fromstring(xml)
    for geom in tree.findall(".//geom"):
        if geom.get("group") == "0":
            geom.set("rgba", "1 0 0 0")
    for site in tree.findall(".//site"):
        site.set("rgba", "0 0 0 0")
    return ET.tostring(tree, encoding="unicode")


def _make_floor_flat(xml):
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    floor_texture_names = set()

    for material in tree.findall(".//material"):
        name = material.get("name", "").lower()
        texture = material.get("texture", "")
        texture_lower = texture.lower()
        if "floor" not in name and "floor" not in texture_lower:
            continue

        if texture:
            floor_texture_names.add(texture)
        material.attrib.pop("texture", None)
        material.set("rgba", "0.58 0.56 0.52 1")
        material.set("reflectance", "0.05")
        material.set("shininess", "0.05")

    for geom in tree.findall(".//geom"):
        name = geom.get("name", "").lower()
        material = geom.get("material", "").lower()
        if "floor" in name or "floor" in material:
            geom.set("rgba", "0.58 0.56 0.52 1")

    if asset is not None:
        for texture in list(asset.findall("texture")):
            name = texture.get("name", "")
            file_path = texture.get("file", "").replace("\\", "/").lower()
            if name in floor_texture_names or "floor" in name.lower() or "/floor/" in file_path:
                asset.remove(texture)

    return ET.tostring(tree, encoding="unicode")


def _make_walls_flat(xml):
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    wall_keywords = ("wall", "brick", "backsplash")
    wall_rgba = "0.88 0.87 0.82 1"
    texture_files = {}
    wall_material_names = {
        geom.get("material")
        for geom in tree.findall(".//geom")
        if geom.get("material")
        and any(keyword in geom.get("name", "").lower() for keyword in wall_keywords)
    }
    wall_texture_names = set()

    if asset is not None:
        texture_files = {
            texture.get("name", ""): texture.get("file", "")
            for texture in asset.findall("texture")
        }

    for material in tree.findall(".//material"):
        name = material.get("name", "")
        texture = material.get("texture", "")
        identifying_text = " ".join((name, texture, texture_files.get(texture, ""))).lower()
        if name not in wall_material_names and not any(
            keyword in identifying_text for keyword in wall_keywords
        ):
            continue

        wall_material_names.add(name)
        if texture:
            wall_texture_names.add(texture)
        material.attrib.pop("texture", None)
        material.set("rgba", wall_rgba)
        material.set("reflectance", "0.05")
        material.set("shininess", "0.05")

    for geom in tree.findall(".//geom"):
        identifying_text = " ".join(
            (geom.get("name", ""), geom.get("material", ""))
        ).lower()
        if geom.get("material") in wall_material_names or any(
            keyword in identifying_text for keyword in wall_keywords
        ):
            geom.set("rgba", wall_rgba)

    remaining_texture_names = {
        material.get("texture")
        for material in tree.findall(".//material")
        if material.get("texture")
    }

    if asset is not None:
        for texture in list(asset.findall("texture")):
            name = texture.get("name", "")
            if name in remaining_texture_names:
                continue
            identifying_text = " ".join((name, texture.get("file", ""))).lower()
            if name in wall_texture_names or any(
                keyword in identifying_text for keyword in wall_keywords
            ):
                asset.remove(texture)

    return ET.tostring(tree, encoding="unicode")


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


def _make_materials_flat(xml):
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    flat_rgba = "0.78 0.78 0.74 1"

    for material in tree.findall(".//material"):
        material.attrib.pop("texture", None)
        material.set("rgba", flat_rgba)
        material.set("reflectance", "0.05")
        material.set("shininess", "0.05")
        material.set("specular", "0.15")

    for geom in tree.findall(".//geom"):
        if geom.get("material") is None:
            geom.set("rgba", flat_rgba)

    if asset is not None:
        for texture in list(asset.findall("texture")):
            if texture.get("type") == "skybox":
                continue
            asset.remove(texture)

    return ET.tostring(tree, encoding="unicode")


def _body_names_for(elem, parent_map):
    names = []
    parent = parent_map.get(elem)
    while parent is not None:
        if parent.tag == "body" and parent.get("name"):
            names.append(parent.get("name", "").lower())
        parent = parent_map.get(parent)
    return names


def _classify_palette_text(*parts):
    name = " ".join(part or "" for part in parts).lower()

    if name.startswith(("robot0_", "mobilebase0_")) or " robot0_" in name or " mobilebase0_" in name:
        return "robot"
    if "floor" in name:
        return "wood"
    if "wall" in name or "brick" in name or "backsplash" in name:
        return "wall"
    if "sink" in name or "faucet" in name:
        return "sink"
    if "counter" in name or "island" in name:
        if "top" in name or "counter_top" in name:
            return "wood"
        return "cabinet"
    if "stove" in name or "stovetop" in name or "burner" in name or "cooktop" in name:
        return "black"
    if "microwave" in name:
        if "glass" in name or "door" in name or "screen" in name or "window" in name:
            return "glass"
        return "metal"
    if "oven" in name:
        if "glass" in name or "door" in name or "window" in name:
            return "glass"
        return "metal"
    if "fridge" in name or "refrigerator" in name or "hood" in name:
        return "metal"
    if "handle" in name or "knob" in name or "trim" in name:
        return "dark_metal"
    if "cab" in name or "drawer" in name or "shelf" in name or "housing" in name or "stack" in name or "box" in name:
        return "cabinet"
    return "cabinet"


def _apply_material_style(material, rgba):
    material.attrib.pop("texture", None)
    material.set("rgba", rgba)
    material.set("reflectance", "0.08")
    material.set("shininess", "0.12")
    material.set("specular", "0.18")


def _make_materials_styled(xml):
    tree = ET.fromstring(xml)
    asset = tree.find("asset")
    parent_map = {child: parent for parent in tree.iter() for child in list(parent)}
    palette = {
        "wall": "0.88 0.87 0.82 1",
        "wood": "0.72 0.52 0.32 1",
        "cabinet": "0.72 0.73 0.70 1",
        "metal": "0.45 0.47 0.48 1",
        "dark_metal": "0.20 0.22 0.22 1",
        "black": "0.03 0.035 0.035 1",
        "glass": "0.04 0.055 0.065 1",
        "sink": "0.64 0.66 0.64 1",
    }
    texture_files = {}
    robot_materials = set()
    object_materials = set()
    object_textures = set()
    object_body_names = {
        body.get("name", "").lower()
        for body in tree.findall(".//body")
        if body.get("name")
        and not body.get("name", "").startswith(("robot0_", "mobilebase0_"))
        and any(joint.get("type") == "free" for joint in body.findall("joint"))
    }

    if asset is not None:
        for texture in asset.findall("texture"):
            texture_name = texture.get("name")
            if texture_name is not None:
                texture_files[texture_name] = texture.get("file", "")

    for geom in tree.findall(".//geom"):
        if any(
            name.startswith(("robot0_", "mobilebase0_"))
            for name in _body_names_for(geom, parent_map)
        ):
            material_name = geom.get("material")
            if material_name is not None:
                robot_materials.add(material_name)
        if any(name in object_body_names for name in _body_names_for(geom, parent_map)):
            material_name = geom.get("material")
            if material_name is not None:
                object_materials.add(material_name)

    for material_name in object_materials:
        material = tree.find(f".//material[@name='{material_name}']")
        if material is not None and material.get("texture"):
            object_textures.add(material.get("texture"))

    for material in tree.findall(".//material"):
        material_name = material.get("name", "")
        texture_name = material.get("texture", "")
        texture_file = texture_files.get(texture_name, "")
        if material_name in object_materials:
            continue
        if material_name in robot_materials:
            material.attrib.pop("texture", None)
            continue
        target = _classify_palette_text(material_name, texture_name, texture_file)
        if target == "robot":
            material.attrib.pop("texture", None)
            continue
        rgba = palette[target]
        _apply_material_style(material, rgba)

    if asset is not None:
        for texture in list(asset.findall("texture")):
            if texture.get("type") == "skybox":
                continue
            if texture.get("name") in object_textures:
                continue
            asset.remove(texture)

    return ET.tostring(tree, encoding="unicode")


def _format_float(value):
    return f"{float(value):.12g}"


def _format_array(values):
    return " ".join(_format_float(value) for value in values)


def _bake_mobile_base_root_pose(tree, env):
    try:
        mobile_body_id = env.sim.model.body_name2id("mobilebase0_base")
    except KeyError:
        return False

    root_body = tree.find(".//body[@name='robot0_base']")
    if root_body is None:
        return False

    root_body.set("pos", _format_array(env.sim.data.body_xpos[mobile_body_id]))
    root_body.set("quat", _format_array(env.sim.data.body_xquat[mobile_body_id]))
    return True


def _bake_robot_pose_into_xml(xml, env):
    tree = ET.fromstring(xml)
    baked_mobile_root = _bake_mobile_base_root_pose(tree, env)
    if not baked_mobile_root:
        raise RuntimeError("Could not find the PandaOmron mobile base pose to bake")

    mobile_base_joints = {
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    }
    reset_qpos = env.sim.data.qpos.copy()
    for joint_name in mobile_base_joints:
        try:
            addr = env.sim.model.get_joint_qpos_addr(joint_name)
        except KeyError:
            continue
        if isinstance(addr, slice):
            reset_qpos[addr] = 0.0
        elif isinstance(addr, tuple):
            reset_qpos[addr[0] : addr[1]] = 0.0
        else:
            reset_qpos[addr] = 0.0

    keyframe = tree.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(tree, "keyframe")
    for key in list(keyframe.findall("key")):
        if key.get("name") == "baked_reset":
            keyframe.remove(key)
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "baked_reset",
            "qpos": " ".join(_format_float(v) for v in reset_qpos),
        },
    )

    return ET.tostring(tree, encoding="unicode")


def _make_robot_replay_compatible(xml):
    tree = ET.fromstring(xml)

    robot_base = tree.find(".//body[@name='robot0_base']")
    if robot_base is None:
        raise RuntimeError("Could not find robot0_base while making replay-compatible XML")
    robot_base.set("pos", "10.0 10.0 0.0")
    robot_base.set("quat", "1.0 0.0 0.0 0.0")

    mobile_axes = {
        "mobilebase0_joint_mobile_forward": "0 1 0",
        "mobilebase0_joint_mobile_side": "1 0 0",
        "mobilebase0_joint_mobile_yaw": "0 0 1",
    }
    for joint_name, axis in mobile_axes.items():
        joint = tree.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise RuntimeError(f"Could not find {joint_name} while making replay-compatible XML")
        joint.set("axis", axis)
        joint.attrib.pop("ref", None)

    keyframe = tree.find("keyframe")
    if keyframe is not None:
        for key in list(keyframe.findall("key")):
            if key.get("name") == "baked_reset":
                keyframe.remove(key)

    return ET.tostring(tree, encoding="unicode")


def _bake_object_poses_into_xml(xml, env):
    tree = ET.fromstring(xml)
    baked_count = 0

    for cfg in getattr(env, "object_cfgs", []):
        obj_name = cfg.get("name")
        if not obj_name:
            continue

        joint_name = f"{obj_name}_joint0"
        try:
            addr = env.sim.model.get_joint_qpos_addr(joint_name)
        except KeyError:
            continue

        if isinstance(addr, slice):
            start, stop = addr.start, addr.stop
        elif isinstance(addr, tuple):
            start, stop = addr
        else:
            continue

        if stop - start != 7:
            continue

        qpos = env.sim.data.qpos[start:stop]
        body = tree.find(f".//body[@name='{obj_name}_main']")
        if body is None:
            continue

        body.set("pos", _format_array(qpos[:3]))
        body.set("quat", _format_array(qpos[3:7]))
        baked_count += 1

    if baked_count == 0:
        raise RuntimeError("No object freejoint poses were found to bake into the exported XML")

    return ET.tostring(tree, encoding="unicode")


def _resolve_layout_id(args):
    layout_yaml = getattr(args, "layout_yaml", None)
    if layout_yaml is None:
        return args.layout

    with layout_yaml.open("r", encoding="utf-8") as f:
        layout_config = yaml.safe_load(f)
    if layout_config is None:
        raise ValueError(f"layout YAML is empty: {layout_yaml}")
    return layout_config


def _load_layout_config(args):
    if args.layout_yaml is not None:
        layout_path = args.layout_yaml
    else:
        layout_path = Path(get_layout_path(args.layout))

    with layout_path.open("r", encoding="utf-8") as f:
        layout_config = yaml.safe_load(f)
    if layout_config is None:
        raise ValueError(f"layout YAML is empty: {layout_path}")
    return layout_config


def _load_object_cfgs(objects_yaml):
    with objects_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"objects YAML must contain a mapping: {objects_yaml}")
    object_cfgs = data.get("objects")
    if not isinstance(object_cfgs, list) or len(object_cfgs) == 0:
        raise ValueError(f"objects YAML must contain a non-empty 'objects' list: {objects_yaml}")

    object_cfgs = copy.deepcopy(object_cfgs)
    required_keys = {"type", "name", "obj_groups", "placement"}
    for idx, cfg in enumerate(object_cfgs):
        if not isinstance(cfg, dict):
            raise ValueError(f"object entry {idx} must be a mapping")
        missing = required_keys - set(cfg)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"object entry {idx} is missing required key(s): {missing_text}")
        if cfg["type"] != "object":
            raise ValueError(f"object entry {idx} must use type: object")
        if not isinstance(cfg["placement"], dict):
            raise ValueError(f"object entry {idx} placement must be a mapping")

        obj_groups = cfg["obj_groups"]
        if isinstance(obj_groups, str) and obj_groups.endswith(".xml"):
            normalized = obj_groups.replace("\\", "/")
            marker = "/models/assets/objects/"
            if marker in normalized:
                suffix = normalized.split(marker, 1)[1]
                cfg["obj_groups"] = str(
                    Path(robocasa.models.assets_root) / "objects" / Path(suffix)
                )

    return object_cfgs


def _layout_floor_rects(args):
    layout_config = _load_layout_config(args)
    floors = layout_config.get("room", {}).get("floor", [])
    rects = []

    for floor in floors:
        if floor.get("type") != "floor" or floor.get("backing"):
            continue
        pos = floor.get("pos")
        size = floor.get("size")
        if pos is None or size is None or len(pos) < 2 or len(size) < 2:
            continue
        # Layout YAML floor sizes are half-extents in world x/y.
        rects.append((float(pos[0]), float(pos[1]), float(size[0]), float(size[1])))

    if not rects:
        raise RuntimeError("No usable floor region found for Stretch auto placement")
    return rects


def _fixture_obstacle_rects(args):
    arena = KitchenArena(
        layout_id=_resolve_layout_id(args),
        style_id=args.style,
        clutter_mode=args.clutter_mode,
    )
    obstacle_rects = []
    ignored_classes = {
        "Floor",
        "Wall",
        "WallAccessory",
        "UtensilRack",
        "Hood",
    }

    for cfg in arena.get_fixture_cfgs():
        model = cfg.get("model")
        if model is None or type(model).__name__ in ignored_classes:
            continue

        pos = getattr(model, "pos", None)
        size = getattr(model, "size", None)
        if pos is None or size is None or len(pos) < 2 or len(size) < 2:
            continue

        try:
            bottom_offset = getattr(model, "bottom_offset", None)
        except ValueError:
            bottom_offset = None
        if bottom_offset is not None and len(bottom_offset) >= 3 and len(pos) >= 3:
            bottom_z = float(pos[2]) + float(bottom_offset[2])
            if bottom_z > 1.0:
                continue

        half_x = abs(float(size[0])) / 2.0
        half_y = abs(float(size[1])) / 2.0
        if half_x < 0.08 or half_y < 0.08:
            continue
        obstacle_rects.append((float(pos[0]), float(pos[1]), half_x, half_y))

    return obstacle_rects


def _point_inside_rect(point, rect, clearance=0.0):
    x, y = point
    cx, cy, half_x, half_y = rect
    return (
        cx - half_x + clearance <= x <= cx + half_x - clearance
        and cy - half_y + clearance <= y <= cy + half_y - clearance
    )


def _rects_overlap(a, b, margin=0.0):
    ax, ay, ahx, ahy = a
    bx, by, bhx, bhy = b
    return (
        abs(ax - bx) < ahx + bhx + margin
        and abs(ay - by) < ahy + bhy + margin
    )


def _rect_distance(a, b):
    ax, ay, ahx, ahy = a
    bx, by, bhx, bhy = b
    dx = max(abs(ax - bx) - ahx - bhx, 0.0)
    dy = max(abs(ay - by) - ahy - bhy, 0.0)
    return math.hypot(dx, dy)


def _candidate_stretch_positions(floor_rect, clearance):
    cx, cy, half_x, half_y = floor_rect
    xmin, xmax = cx - half_x + clearance, cx + half_x - clearance
    ymin, ymax = cy - half_y + clearance, cy + half_y - clearance
    if xmin > xmax or ymin > ymax:
        return []

    xs = [cx]
    for frac in (0.25, 0.5, 0.75):
        xs.extend([cx - half_x * frac, cx + half_x * frac])
    xs = [min(max(x, xmin), xmax) for x in xs]

    candidates = []
    front_y = ymin
    back_y = ymax
    candidates.extend((x, front_y, "front") for x in xs)
    candidates.extend((x, min(ymin + half_y * 0.35, ymax), "front_mid") for x in xs)
    candidates.extend((x, back_y, "back") for x in xs)

    y_steps = 7
    x_steps = 9
    for yi in range(y_steps):
        y = ymin + (ymax - ymin) * yi / max(y_steps - 1, 1)
        candidates.append((xmin, y, "left"))
        candidates.append((xmax, y, "right"))
        for xi in range(x_steps):
            x = xmin + (xmax - xmin) * xi / max(x_steps - 1, 1)
            candidates.append((x, y, "grid"))

    seen = set()
    unique = []
    for x, y, band in candidates:
        key = (round(x, 3), round(y, 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append((x, y, band))
    return unique


def _score_stretch_position(candidate, floor_rect, obstacles, robot_half):
    x, y, band = candidate
    cx, cy, half_x, half_y = floor_rect
    robot_rect = (x, y, robot_half, robot_half)
    min_obstacle_distance = min(
        (_rect_distance(robot_rect, obstacle) for obstacle in obstacles),
        default=2.0,
    )
    centered_x = 1.0 - min(abs(x - cx) / max(half_x, 1e-6), 1.0)
    front_bias = 1.0 - min((y - (cy - half_y)) / max(2 * half_y, 1e-6), 1.0)
    band_bias = {
        "front": 3.0,
        "front_mid": 2.2,
        "left": 1.2,
        "right": 1.2,
        "grid": 0.4,
        "back": 0.1,
    }.get(band, 0.0)
    return band_bias + 1.5 * front_bias + centered_x + min(min_obstacle_distance, 1.0)


def _auto_place_stretch(args):
    floor_rects = _layout_floor_rects(args)
    obstacles = _fixture_obstacle_rects(args)
    robot_half = args.stretch_footprint / 2.0
    clearance = robot_half + args.stretch_margin

    best = None
    for floor_rect in floor_rects:
        for x, y, band in _candidate_stretch_positions(floor_rect, clearance):
            robot_rect = (x, y, robot_half, robot_half)
            if any(_rects_overlap(robot_rect, obstacle, args.stretch_margin) for obstacle in obstacles):
                continue
            score = _score_stretch_position((x, y, band), floor_rect, obstacles, robot_half)
            if best is None or score > best[0]:
                best = (score, x, y, floor_rect)

    if best is None:
        raise RuntimeError(
            "Could not find a collision-free Stretch placement on the floor. "
            "Try a smaller --stretch-footprint or --stretch-margin."
        )

    _, x, y, floor_rect = best
    floor_cx, floor_cy, _, _ = floor_rect
    yaw = math.atan2(floor_cy - y, floor_cx - x)
    quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    return (x, y, 0.0), quat


def _candidate_inward_yaw(candidate, floor_rect):
    x, y, band = candidate
    if band in {"front", "front_mid"}:
        return math.pi / 2.0
    if band == "back":
        return -math.pi / 2.0
    if band == "left":
        return 0.0
    if band == "right":
        return math.pi

    cx, cy, half_x, half_y = floor_rect
    edge_distances = {
        "left": abs(x - (cx - half_x)),
        "right": abs(x - (cx + half_x)),
        "front": abs(y - (cy - half_y)),
        "back": abs(y - (cy + half_y)),
    }
    nearest_band = min(edge_distances, key=edge_distances.get)
    return _candidate_inward_yaw((x, y, nearest_band), floor_rect)


def _body_xy_rect_if_on_floor(scene_root, body_name, floor_rects, robot_half):
    body = scene_root.find(f"./worldbody/body[@name='{body_name}']")
    if body is None or body.get("pos") is None:
        return None
    pos = [float(value) for value in body.get("pos").split()]
    if len(pos) < 2:
        return None
    if not any(_point_inside_rect(pos[:2], floor_rect) for floor_rect in floor_rects):
        return None
    return (pos[0], pos[1], robot_half, robot_half)


def _find_auto_pandaomron_placement(args, floor_rects, obstacles):
    robot_half = args.extra_robot_footprint / 2.0
    clearance = robot_half + args.extra_robot_margin
    best = None
    for floor_rect in floor_rects:
        for candidate in _candidate_stretch_positions(floor_rect, clearance):
            x, y, band = candidate
            robot_rect = (x, y, robot_half, robot_half)
            if any(
                _rects_overlap(robot_rect, obstacle, args.extra_robot_margin)
                for obstacle in obstacles
            ):
                continue
            score = _score_stretch_position(candidate, floor_rect, obstacles, robot_half)
            if best is None or score > best[0]:
                best = (score, x, y, band, floor_rect, robot_rect)

    if best is None:
        raise RuntimeError(
            "Could not find a collision-free PandaOmron placement on the floor. "
            "Try a smaller --extra-robot-footprint or --extra-robot-margin."
        )

    _, x, y, band, floor_rect, robot_rect = best
    yaw = _candidate_inward_yaw((x, y, band), floor_rect)
    quat = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    return (x, y, 0.0), quat, robot_rect


def _auto_place_primary_pandaomron(xml, args):
    scene_root = ET.fromstring(xml)
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Could not find worldbody while placing the primary PandaOmron")

    robot_body = worldbody.find("./body[@name='robot0_base']")
    if robot_body is None:
        raise RuntimeError("Could not find robot0_base to auto-place the primary PandaOmron")

    floor_rects = _layout_floor_rects(args)
    obstacles = _fixture_obstacle_rects(args)
    pos, quat, _ = _find_auto_pandaomron_placement(args, floor_rects, obstacles)
    robot_body.set("pos", _format_array(pos))
    robot_body.set("quat", _format_array(quat))
    return ET.tostring(scene_root, encoding="unicode")


def _auto_place_extra_pandaomrons(args, scene_root):
    floor_rects = _layout_floor_rects(args)
    obstacles = _fixture_obstacle_rects(args)
    robot_half = args.extra_robot_footprint / 2.0
    existing_robot = _body_xy_rect_if_on_floor(
        scene_root, "robot0_base", floor_rects, robot_half
    )
    if existing_robot is not None:
        obstacles.append(existing_robot)

    placements = []
    for _ in range(args.extra_pandaomrons):
        pos, quat, robot_rect = _find_auto_pandaomron_placement(
            args, floor_rects, obstacles
        )
        placements.append((pos, quat))
        obstacles.append(robot_rect)

    return placements


ROBOT_NAME_PREFIXES = ("robot0_", "mobilebase0_", "gripper0_")


def _element_uses_robot_names(element):
    return any(
        prefix in value
        for elem in element.iter()
        for value in elem.attrib.values()
        for prefix in ROBOT_NAME_PREFIXES
    )


def _robot_clone_rename_map(source_body, section_sources, robot_index):
    owned_elements = [source_body]
    owned_elements.extend(
        child for source_children in section_sources.values() for child in source_children
    )
    owned_names = {
        elem.get("name")
        for element in owned_elements
        for elem in element.iter()
        if elem.get("name") is not None
    }

    rename_map = {}
    for name in owned_names:
        renamed = name
        for source, target in {
            "robot0_": f"robot{robot_index}_",
            "mobilebase0_": f"mobilebase{robot_index}_",
            "gripper0_": f"gripper{robot_index}_",
        }.items():
            renamed = renamed.replace(source, target)
        if renamed == name:
            renamed = f"robot{robot_index}_{name}"
        rename_map[name] = renamed
    return rename_map


def _clone_for_robot_index(element, robot_index, rename_map):
    clone = copy.deepcopy(element)
    replacements = {
        "robot0_": f"robot{robot_index}_",
        "mobilebase0_": f"mobilebase{robot_index}_",
        "gripper0_": f"gripper{robot_index}_",
    }
    for elem in clone.iter():
        for attr, value in list(elem.attrib.items()):
            if value in rename_map:
                value = rename_map[value]
            for source, target in replacements.items():
                value = value.replace(source, target)
            elem.set(attr, value)
    return clone


def _merge_extra_pandaomrons(xml, args):
    scene_root = ET.fromstring(xml)
    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Could not find worldbody while cloning PandaOmron")

    source_body = worldbody.find("./body[@name='robot0_base']")
    if source_body is None:
        raise RuntimeError("Could not find robot0_base to clone as an extra PandaOmron")

    placements = _auto_place_extra_pandaomrons(args, scene_root)
    clonable_sections = ("asset", "contact", "tendon", "equality", "actuator", "sensor")
    section_sources = {}
    for section_name in clonable_sections:
        section = scene_root.find(section_name)
        if section is None:
            continue
        section_sources[section_name] = [
            child for child in list(section) if _element_uses_robot_names(child)
        ]

    for robot_index, (pos, quat) in enumerate(placements, start=1):
        rename_map = _robot_clone_rename_map(
            source_body, section_sources, robot_index
        )
        for section_name, source_children in section_sources.items():
            section = _get_or_create_top_level_section(scene_root, section_name)
            for child in source_children:
                section.append(
                    _clone_for_robot_index(child, robot_index, rename_map)
                )

        robot_body = _clone_for_robot_index(source_body, robot_index, rename_map)
        robot_body.set("pos", _format_array(pos))
        robot_body.set("quat", _format_array(quat))
        worldbody.append(robot_body)

    keyframe = scene_root.find("keyframe")
    if keyframe is not None:
        for key in list(keyframe.findall("key")):
            if key.get("name") == "baked_reset":
                keyframe.remove(key)

    return ET.tostring(scene_root, encoding="unicode")


def _prefix_value(value, rename_map):
    return rename_map.get(value, value)


def _prefix_mjcf(stretch_tree, prefix):
    root = stretch_tree
    for tag in ("mesh", "texture"):
        for asset in root.findall(f".//{tag}"):
            if asset.get("name") is None and asset.get("file"):
                asset.set("name", Path(asset.get("file")).stem)

    named_values = {
        elem.get("name")
        for elem in root.iter()
        if elem.get("name") is not None
    }
    class_values = {
        elem.get("class")
        for elem in root.iter("default")
        if elem.get("class") is not None
    }
    rename_map = {
        value: f"{prefix}_{value}"
        for value in named_values.union(class_values)
        if value is not None and not value.startswith(f"{prefix}_")
    }

    name_ref_attrs = {
        "material",
        "texture",
        "mesh",
        "joint",
        "tendon",
        "body1",
        "body2",
        "joint1",
        "joint2",
        "site",
    }
    class_ref_attrs = {"class", "childclass"}

    for elem in root.iter():
        if elem.get("name") in rename_map:
            elem.set("name", rename_map[elem.get("name")])
        if elem.tag == "default" and elem.get("class") in rename_map:
            elem.set("class", rename_map[elem.get("class")])
        for attr in name_ref_attrs:
            if elem.get(attr) in rename_map:
                elem.set(attr, _prefix_value(elem.get(attr), rename_map))
        for attr in class_ref_attrs:
            if elem.get(attr) in rename_map:
                elem.set(attr, _prefix_value(elem.get(attr), rename_map))


def _resolve_stretch_asset_paths(stretch_root, stretch_mjcf):
    compiler = stretch_root.find("compiler")
    assetdir = compiler.get("assetdir") if compiler is not None else None
    asset_root = stretch_mjcf.parent / assetdir if assetdir else stretch_mjcf.parent

    for elem in stretch_root.findall(".//*[@file]"):
        file_path = Path(elem.get("file").replace("\\", "/"))
        if file_path.is_absolute():
            continue
        elem.set("file", str((asset_root / file_path).resolve()))


def _get_or_create_top_level_section(root, tag):
    section = root.find(tag)
    if section is not None:
        return section

    section = ET.Element(tag)
    keyframe = root.find("keyframe")
    if keyframe is None:
        root.append(section)
    else:
        root.insert(list(root).index(keyframe), section)
    return section


def _merge_stretch_mjcf(xml, args):
    scene_root = ET.fromstring(xml)
    stretch_root = ET.parse(args.stretch_mjcf).getroot()
    _resolve_stretch_asset_paths(stretch_root, args.stretch_mjcf)
    _prefix_mjcf(stretch_root, args.stretch_prefix)

    compiler = scene_root.find("compiler")
    if compiler is not None:
        compiler.set("autolimits", "true")
        compiler.attrib.pop("inertiagrouprange", None)

    stretch_default = stretch_root.find("default")
    if stretch_default is not None and list(stretch_default):
        scene_default = _get_or_create_top_level_section(scene_root, "default")
        for child in list(stretch_default):
            scene_default.append(copy.deepcopy(child))

    stretch_asset = stretch_root.find("asset")
    if stretch_asset is not None:
        scene_asset = _get_or_create_top_level_section(scene_root, "asset")
        for child in list(stretch_asset):
            scene_asset.append(copy.deepcopy(child))

    pos, quat = _auto_place_stretch(args)
    stretch_worldbody = stretch_root.find("worldbody")
    scene_worldbody = scene_root.find("worldbody")
    if stretch_worldbody is None or scene_worldbody is None:
        raise RuntimeError("Could not find worldbody while merging Stretch MJCF")

    stretch_bodies = list(stretch_worldbody.findall("body"))
    if len(stretch_bodies) != 1:
        raise RuntimeError("Expected Stretch MJCF to contain exactly one root body")
    stretch_body = copy.deepcopy(stretch_bodies[0])
    stretch_body.set("pos", _format_array(pos))
    stretch_body.set("quat", _format_array(quat))
    scene_worldbody.append(stretch_body)

    for section_name in ("contact", "tendon", "equality", "actuator"):
        stretch_section = stretch_root.find(section_name)
        if stretch_section is None:
            continue
        scene_section = _get_or_create_top_level_section(scene_root, section_name)
        for child in list(stretch_section):
            scene_section.append(copy.deepcopy(child))

    keyframe = scene_root.find("keyframe")
    if keyframe is not None:
        for key in list(keyframe.findall("key")):
            if key.get("name") == "baked_reset":
                keyframe.remove(key)

    return ET.tostring(scene_root, encoding="unicode")


def _make_scene_only_xml(args):
    arena = KitchenArena(
        layout_id=_resolve_layout_id(args),
        style_id=args.style,
        clutter_mode=args.clutter_mode,
    )
    arena.set_origin([0, 0, 0])
    fixtures = [cfg["model"] for cfg in arena.get_fixture_cfgs()]
    model = ManipulationTask(
        mujoco_arena=arena,
        mujoco_robots=[],
        mujoco_objects=fixtures,
        enable_multiccd=True,
        enable_sleeping_islands=False,
    )
    return model.get_xml()


def _make_env_xml(args):
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "layout_and_style_ids": [[_resolve_layout_id(args), args.style]],
        "clutter_mode": args.clutter_mode,
        "translucent_robot": False,
        "init_robot_base_ref": args.init_robot_base_ref,
    }

    env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        renderer="mujoco",
        control_freq=20,
    )
    if args.objects_yaml is not None:
        if not hasattr(env, "_ep_meta") or env._ep_meta is None:
            env._ep_meta = {}
        env._ep_meta["object_cfgs"] = _load_object_cfgs(args.objects_yaml)
    env.reset()

    xml = env.model.get_xml()
    if not args.relative_paths:
        xml = env.edit_model_xml(xml)
    if args.replay_compatible_robot:
        xml = _make_robot_replay_compatible(xml)
    if args.bake_robot_pose:
        xml = _bake_robot_pose_into_xml(xml, env)
    if args.objects_yaml is not None:
        xml = _bake_object_poses_into_xml(xml, env)
    env.close()
    return xml


def _write_mjb(xml_path):
    import mujoco

    mjb_path = xml_path.with_suffix(".mjb")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mujoco.mj_saveModel(model, str(mjb_path))
    return mjb_path


def main():
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    xml = _make_scene_only_xml(args) if args.scene_only else _make_env_xml(args)
    if args.auto_place_primary_pandaomron:
        xml = _auto_place_primary_pandaomron(xml, args)

    if args.styled_materials:
        xml = _make_materials_styled(xml)
    elif args.flat_materials:
        xml = _make_materials_flat(xml)
    else:
        if args.flat_floor:
            xml = _make_floor_flat(xml)
        if args.flat_wall:
            xml = _make_walls_flat(xml)
        if args.flat_countertop:
            xml = _make_countertops_flat(xml)

    extra_asset_roots = []
    if args.extra_pandaomrons:
        xml = _merge_extra_pandaomrons(xml, args)
    if args.stretch_auto_place:
        xml = _merge_stretch_mjcf(xml, args)
        extra_asset_roots.append((args.stretch_mjcf.parent, "stretch_assets"))

    if args.web_assets:
        xml = _rewrite_asset_paths_for_web(xml, args.output.parent, extra_asset_roots)
        xml = _hide_helper_visuals(xml)

    args.output.write_text(xml, encoding="utf-8")

    if args.mjb:
        mjb_path = _write_mjb(args.output)
        print(f"Wrote {mjb_path}")

    if args.copy_assets:
        assets_src = Path(robocasa.__file__).parent / "models" / "assets"
        assets_dst = args.output.parent / "robocasa_assets"
        _copy_assets(assets_src, assets_dst)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
