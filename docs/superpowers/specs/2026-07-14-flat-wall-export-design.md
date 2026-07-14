# Flat Wall Export Design

## Goal

Add an independent `--flat-wall` option to `export_kitchen_scene.py`. It replaces kitchen wall textures with a lightweight solid-color material without changing the floor, cabinets, objects, or robot materials.

## Behavior

- `--flat-wall` is a boolean command-line flag and is disabled by default.
- Wall materials are recognized from material names, texture names, texture file paths, and referencing geom names. Matching covers `wall`, `brick`, and `backsplash` identifiers.
- Matching materials have their texture reference removed and receive one neutral light wall color with low reflectance and shininess.
- Matching geoms receive the same color so wall geoms without a dedicated material are also flattened.
- Texture assets used by matched wall materials, or clearly identified as wall textures, are removed after references are cleared.
- `--flat-wall` and `--flat-floor` can be enabled together. Each transformation runs independently.
- Existing `--flat-materials` and `--styled-materials` remain whole-scene alternatives. When either is enabled, the targeted wall/floor transforms are unnecessary and are not applied.

## Implementation

Add `_make_walls_flat(xml)` beside `_make_floor_flat(xml)` and follow the existing ElementTree transformation pattern. Add the parser flag and update `main()` so targeted transformations are separate `if` statements under the existing whole-scene material-mode precedence.

No broad material refactor is included; the change stays localized to the exporter.

## Testing

Add focused tests using small inline MJCF strings. Verify that:

1. Wall, brick, and backsplash materials lose texture references and receive the wall solid color.
2. Corresponding texture assets are removed.
3. Unrelated floor and cabinet materials/textures remain unchanged.
4. Floor and wall flattening can be composed.
5. Argument parsing exposes `--flat-wall` independently.
