# Kitchen Layout Cabinet Complexity Analyzer

## Goal

Provide a lightweight command-line script that ranks RoboCasa kitchen layout YAML files from simplest to most complex using only the number of cabinet fixtures. The tool is intended to help select simple training layouts without counting decorations, room geometry, counters, or appliances.

## Scope

The analyzer counts fixtures whose `type` is one of:

- `hinge_cabinet`
- `single_cabinet`
- `open_cabinet`
- `panel_cabinet`
- `housing_cabinet`

It ignores all other fixture types. In particular, `stack` is a layout grouping instruction rather than a physical cabinet and is not counted. Disabled cabinet entries (`enable: false`) are also not counted because they do not appear in the constructed scene.

## Interface

The script lives under `robocasa/scripts/` and accepts:

- an optional layout directory, defaulting to the repository's `robocasa/models/assets/scenes/kitchen_layouts/train` directory;
- `--top N` to show only the first N layouts; if omitted, all layouts are shown.

It prints a terminal table sorted by ascending total cabinet count, then by the numeric portion of the layout filename. Each row contains the layout name, total cabinet count, and nonzero per-type counts. It does not create CSV or JSON output.

## Processing Design

For every `.yaml` or `.yml` file directly in the selected directory, the script parses YAML with `yaml.safe_load`. It recursively traverses dictionaries and lists so cabinet fixtures are found regardless of their group or fixture-list key. A dictionary containing a recognized `type` contributes one count unless it has `enable: false`.

Each file produces an independent result. A malformed file produces a concise error on stderr while analysis continues for the remaining files. The process exits nonzero if the input directory is invalid, no YAML files are present, or any file fails to parse; successfully parsed layouts are still printed when individual files fail.

## Testing

Unit tests use temporary YAML files to verify:

- nested cabinet fixtures are counted by type;
- `stack`, counters, appliances, room geometry, and decorations are ignored;
- disabled cabinets are ignored;
- ranking uses total count and then numeric layout ID;
- `--top` limits terminal output;
- malformed YAML is reported without suppressing valid results.

Finally, the script is run against all train layouts to confirm it handles the existing dataset and to identify the lowest-cabinet layouts.
