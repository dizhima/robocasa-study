from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBOSUITE_ROOT = ROOT / "robosuite"
BUNDLE_ROOT = ROOT / "pandaomron_asset_bundle"
ASSETS_ROOT = BUNDLE_ROOT / "assets"
ENTRY_XML = BUNDLE_ROOT / "pandaomron.xml"


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def make_relative_asset_paths(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", "PandaOmron")

    source_assets = (ROBOSUITE_ROOT / "robosuite" / "models" / "assets").resolve()
    for elem in root.findall(".//*[@file]"):
        file_path = Path(elem.attrib["file"])
        if file_path.is_absolute():
            try:
                rel = file_path.resolve().relative_to(source_assets)
            except ValueError:
                continue
            elem.set("file", (Path("assets") / rel).as_posix())

    tree.write(xml_path, encoding="unicode")


def write_readme() -> None:
    readme = BUNDLE_ROOT / "README.md"
    readme.write_text(
        """# PandaOmron Asset Bundle

This folder contains a self-contained RoboSuite/RoboCasa-style PandaOmron MJCF asset bundle.

- `pandaomron.xml` is the generated entry MJCF. It is assembled using RoboSuite's runtime composition logic.
- `assets/robots/panda/` contains the Panda arm XML and meshes.
- `assets/bases/omron_mobile_base.xml` and `assets/bases/meshes/omron_mobile_base/` contain the Omron mobile base.
- `assets/grippers/panda_gripper.xml` and `assets/grippers/meshes/panda_gripper/` contain the Panda gripper.
- `controllers/` contains the default PandaOmron controller JSON files for reference.

Load `pandaomron.xml` from this directory, or keep the folder layout unchanged if you move it.
""",
        encoding="utf-8",
    )


def main() -> None:
    sys.path.insert(0, str(ROBOSUITE_ROOT))

    from robosuite.robots import ROBOT_CLASS_MAPPING

    source_assets = ROBOSUITE_ROOT / "robosuite" / "models" / "assets"
    source_controllers = ROBOSUITE_ROOT / "robosuite" / "controllers" / "config" / "robots"

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    copy_tree(source_assets / "robots" / "panda", ASSETS_ROOT / "robots" / "panda")
    copy_file(source_assets / "bases" / "omron_mobile_base.xml", ASSETS_ROOT / "bases" / "omron_mobile_base.xml")
    copy_tree(
        source_assets / "bases" / "meshes" / "omron_mobile_base",
        ASSETS_ROOT / "bases" / "meshes" / "omron_mobile_base",
    )
    copy_file(source_assets / "grippers" / "panda_gripper.xml", ASSETS_ROOT / "grippers" / "panda_gripper.xml")
    copy_tree(
        source_assets / "grippers" / "meshes" / "panda_gripper",
        ASSETS_ROOT / "grippers" / "meshes" / "panda_gripper",
    )
    copy_file(
        source_controllers / "default_pandaomron.json",
        BUNDLE_ROOT / "controllers" / "default_pandaomron.json",
    )
    copy_file(
        source_controllers / "default_pandaomron_whole_body_ik.json",
        BUNDLE_ROOT / "controllers" / "default_pandaomron_whole_body_ik.json",
    )

    robot = ROBOT_CLASS_MAPPING["PandaOmron"]("PandaOmron", idn=0)
    robot.load_model()
    robot.robot_model.save_model(str(ENTRY_XML), pretty=True)
    make_relative_asset_paths(ENTRY_XML)
    write_readme()

    print(f"Created {BUNDLE_ROOT}")
    print(f"Entry XML: {ENTRY_XML}")


if __name__ == "__main__":
    main()
