# Flat Countertop Export Design

## Goal

Add an independent `--flat-countertop` option to `export_kitchen_scene.py`. It replaces RoboCasa counter and island work-surface textures with a solid color while preserving counter bases and stove, stovetop, burner, and cooktop materials.

## Identification

RoboCasa's `Counter` fixture assigns work surfaces a material named `<fixture_name>_counter_top`; the source template uses `counter_top`. The exporter will identify countertop materials only when the material name equals `counter_top` or ends with `_counter_top`.

Texture paths and broad words such as `top`, `counter`, `island`, or `stove` will not independently trigger flattening. This prevents accidental changes to appliance tops and unrelated fixture parts.

## Behavior

- `--flat-countertop` is disabled by default.
- Matching materials lose their `texture` attribute and receive one neutral light-gray solid color with low reflectance and shininess.
- Geoms referencing matching materials receive the same color.
- A texture asset formerly used by a countertop is removed only when no remaining material references it.
- Counter bases, cabinets, objects, and stove appliance surfaces remain unchanged.
- `--flat-countertop`, `--flat-floor`, and `--flat-wall` can be combined independently.
- Existing `--flat-materials` and `--styled-materials` remain whole-scene alternatives and take precedence over targeted transformations.

## Implementation

Add `_make_countertops_flat(xml)` beside the existing targeted floor and wall transformations. Add the CLI flag and call the helper in the targeted-material branch of `main()`.

No fixture source files or RoboCasa texture-selection behavior will be changed.

## Testing

Focused inline-MJCF tests will verify:

1. `counter_top`, island-prefixed `_counter_top`, and stove-counter `_counter_top` materials become solid colored.
2. Counter bases and stove, stovetop, burner, and cooktop materials remain textured.
3. Orphaned countertop textures are removed, while textures shared with an unchanged material remain.
4. Countertop flattening composes with floor and wall flattening.
5. Argument parsing exposes `--flat-countertop` independently.
