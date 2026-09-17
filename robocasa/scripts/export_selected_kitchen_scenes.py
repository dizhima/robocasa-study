"""Export the selected RoboCasa kitchen scene set for the React viewer."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from robocasa.scripts.export_kitchen_scene import (
    _hide_helper_visuals,
    _make_scene_only_xml,
    _rewrite_asset_paths_for_web,
)


SELECTED_SCENES = [
    (11, 27),
    (12, 11),
    (13, 20),
    (14, 23),
    (15, 15),
    (16, 16),
    (17, 32),
    (18, 26),
    (19, 13),
    (20, 55),
    (21, 58),
    (22, 30),
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "exported_scenes_web"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory where XML files and robocasa_assets are written",
    )
    parser.add_argument(
        "--clutter-mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 disables clutter fixtures, 1 keeps them",
    )
    return parser.parse_args()


def export_scene(layout, style, output_dir, clutter_mode):
    output_path = output_dir / f"layout{layout:03d}_style{style:03d}.xml"
    scene_args = SimpleNamespace(
        layout=layout,
        style=style,
        clutter_mode=clutter_mode,
    )

    xml = _make_scene_only_xml(scene_args)
    xml = _rewrite_asset_paths_for_web(xml, output_dir)
    xml = _hide_helper_visuals(xml)

    output_path.write_text(xml, encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting RoboCasa scenes:")
    for layout, style in SELECTED_SCENES:
        print(f"  Layout {layout} / Style {style}")

    for layout, style in SELECTED_SCENES:
        output_path = export_scene(
            layout=layout,
            style=style,
            output_dir=args.output_dir,
            clutter_mode=args.clutter_mode,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
