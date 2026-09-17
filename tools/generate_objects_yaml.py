"""Generate RoboCasa object configuration YAML for a kitchen layout."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


CUP_GROUPS = ("mug", "cup")
FRUIT_GROUPS = ("apple", "banana", "orange", "lemon")
WIDE_PLACEMENT_CATEGORIES = ("banana",)
LONG_AXIS_BOUNDS = (-0.85, 0.85)
EDGE_DISTANCE_BOUNDS = (0.60, 0.85)
ATTEMPTS_PER_DISTANCE = 100
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "models"
    / "assets"
    / "scenes"
    / "custom_layouts"
)


def _collect_named_fixtures(value: Any, group_name: str | None = None) -> set[str]:
    """Collect fixture names derived from RoboCasa group and fixture names."""
    fixtures: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_group = key if group_name is None and "_group" in key else group_name
            fixtures.update(_collect_named_fixtures(child, child_group))

        name = value.get("name")
        fixture_type = value.get("type")
        if group_name and isinstance(name, str) and isinstance(fixture_type, str):
            fixtures.add(f"{name}_{group_name}")
    elif isinstance(value, list):
        for child in value:
            fixtures.update(_collect_named_fixtures(child, group_name))
    return fixtures


def collect_fixture_names(value: Any) -> set[str]:
    """Recursively collect explicit and group-qualified fixture identifiers."""
    fixtures = _collect_named_fixtures(value)
    if isinstance(value, dict):
        placement = value.get("placement")
        if isinstance(placement, dict) and isinstance(placement.get("fixture"), str):
            fixtures.add(placement["fixture"])
        for child in value.values():
            fixtures.update(collect_fixture_names(child))
    elif isinstance(value, list):
        for child in value:
            fixtures.update(collect_fixture_names(child))
    return fixtures


def collect_fixture_sizes(
    value: Any, group_name: str | None = None
) -> dict[str, tuple[float, float]]:
    """Map group-qualified fixture names to their planar layout dimensions."""
    fixture_sizes: dict[str, tuple[float, float]] = {}
    if isinstance(value, dict):
        name = value.get("name")
        fixture_type = value.get("type")
        size = value.get("size")
        if (
            group_name
            and isinstance(name, str)
            and isinstance(fixture_type, str)
            and isinstance(size, list)
            and len(size) >= 2
            and all(isinstance(dimension, (int, float)) for dimension in size[:2])
        ):
            fixture_sizes[f"{name}_{group_name}"] = (
                float(size[0]),
                float(size[1]),
            )

        for key, child in value.items():
            child_group = key if group_name is None and "_group" in key else group_name
            fixture_sizes.update(collect_fixture_sizes(child, child_group))
    elif isinstance(value, list):
        for child in value:
            fixture_sizes.update(collect_fixture_sizes(child, group_name))
    return fixture_sizes


def collect_fixture_types(
    value: Any, group_name: str | None = None
) -> dict[str, str]:
    """Map group-qualified fixture names to their declared layout types."""
    fixture_types: dict[str, str] = {}
    if isinstance(value, dict):
        name = value.get("name")
        fixture_type = value.get("type")
        if group_name and isinstance(name, str) and isinstance(fixture_type, str):
            fixture_types[f"{name}_{group_name}"] = fixture_type

        for key, child in value.items():
            child_group = key if group_name is None and "_group" in key else group_name
            fixture_types.update(collect_fixture_types(child, child_group))
    elif isinstance(value, list):
        for child in value:
            fixture_types.update(collect_fixture_types(child, group_name))
    return fixture_types


def collect_interior_fixture_refs(
    value: Any, group_name: str | None = None
) -> dict[str, str]:
    """Map fixtures to group-qualified fixtures embedded in their worktop."""
    interior_refs: dict[str, str] = {}
    if isinstance(value, dict):
        name = value.get("name")
        interior_obj = value.get("interior_obj")
        if group_name and isinstance(name, str) and isinstance(interior_obj, str):
            interior_refs[f"{name}_{group_name}"] = f"{interior_obj}_{group_name}"

        for key, child in value.items():
            child_group = key if group_name is None and "_group" in key else group_name
            interior_refs.update(collect_interior_fixture_refs(child, child_group))
    elif isinstance(value, list):
        for child in value:
            interior_refs.update(collect_interior_fixture_refs(child, group_name))
    return interior_refs


def choose_fixture(fixtures: set[str], requested: str | None) -> str:
    """Choose a placement fixture using the documented priority order."""
    if requested:
        return requested

    ordered = sorted(fixtures)
    if "island_island_group" in fixtures:
        return "island_island_group"

    numbered_primary_islands = [
        name for name in ordered if name.lower().startswith("island_island_group_")
    ]
    if numbered_primary_islands:
        return numbered_primary_islands[0]

    island_fixtures = [
        name
        for name in ordered
        if "island" in name.lower() and "_group" in name.lower()
    ]
    if island_fixtures:
        return island_fixtures[0]

    counter_fixtures = [
        name
        for name in ordered
        if "counter" in name.lower() and "_main_group" in name.lower()
    ]
    if counter_fixtures:
        return counter_fixtures[0]

    candidates = ", ".join(ordered) if ordered else "none"
    raise ValueError(
        "Could not find an island or main counter placement fixture. "
        f"Discovered fixture candidates: {candidates}. Use --fixture to specify one."
    )


def parse_spec(spec: str) -> list[tuple[str, str | None]]:
    """Parse ``--spec`` into an ordered list of (category, instance-or-None).

    Syntax: comma-separated entries of ``category[=instance][:count]``,
    e.g. ``milk=milk_0,apple,mug:2,bowl=bowl_1``.
    """
    entries: list[tuple[str, str | None]] = []
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        count = 1
        if ":" in entry:
            entry, count_text = entry.rsplit(":", 1)
            try:
                count = int(count_text)
            except ValueError as error:
                raise ValueError(f"invalid count in spec entry '{raw_entry}'") from error
            if count <= 0:
                raise ValueError(f"count must be positive in spec entry '{raw_entry}'")
        instance = None
        if "=" in entry:
            entry, instance = entry.split("=", 1)
            instance = instance.strip() or None
        category = entry.strip()
        if not category:
            raise ValueError(f"missing category in spec entry '{raw_entry}'")
        entries.extend([(category, instance)] * count)
    if not entries:
        raise ValueError("--spec did not contain any objects")
    return entries


ENV_OBJ_REGISTRIES = ("objaverse", "lightwheel")
PORTABLE_OBJECTS_PREFIX = "robocasa/models/assets/objects/"


def portable_mjcf_path(mjcf_path: str) -> str:
    """Return a repository-relative object path suitable for committed YAML."""
    normalized = mjcf_path.replace("\\", "/")
    marker = "/models/assets/objects/"
    if marker not in normalized:
        raise ValueError(f"object MJCF path is outside RoboCasa assets: {mjcf_path}")
    return PORTABLE_OBJECTS_PREFIX + normalized.split(marker, 1)[1]


def resolve_instance_mjcf_path(category: str, instance: str) -> str:
    """Return a portable model.xml path for a pinned object instance.

    The instance is looked up from the active registry rather than joined by
    hand, then stored relative to the repository so committed object YAML works
    on another machine. The custom exporter resolves it to the local assets root.
    """
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES

    registries = OBJ_CATEGORIES.get(category)
    if not registries:
        raise ValueError(f"unknown object category: '{category}'")
    for registry in ENV_OBJ_REGISTRIES:
        obj_cat = registries.get(registry)
        if obj_cat is None:
            continue
        for mjcf_path in obj_cat.mjcf_paths:
            if Path(mjcf_path).parent.name == instance:
                return portable_mjcf_path(mjcf_path)
    available = sorted(
        {
            Path(p).parent.name
            for registry in ENV_OBJ_REGISTRIES
            for oc in [registries.get(registry)]
            if oc is not None
            for p in oc.mjcf_paths
        }
    )
    raise ValueError(
        f"instance '{instance}' not found for category '{category}' in "
        f"registries {ENV_OBJ_REGISTRIES}. Available instances: {', '.join(available)}"
    )


def adaptive_min_distance(count: int, compensation: int = 1) -> float:
    """Return one normalized long-axis spacing for a two-lane layout."""
    if count <= 0:
        raise ValueError("object count must be positive")
    if compensation not in (1, 2):
        raise ValueError("--spacing-compensation must be 1 or 2")
    objects_per_lane = math.ceil(count / 2)
    long_axis_span = LONG_AXIS_BOUNDS[1] - LONG_AXIS_BOUNDS[0]
    return long_axis_span / (objects_per_lane + compensation - 1)


def sample_positions(
    rng: random.Random,
    count: int,
    fixture_size: tuple[float, float],
    spacing_compensation: int = 1,
) -> list[list[float]]:
    """Sample near both edges of the fixture's shorter planar axis."""
    if len(fixture_size) != 2 or any(dimension <= 0 for dimension in fixture_size):
        raise ValueError("fixture size must contain two positive planar dimensions")

    short_axis = 0 if fixture_size[0] <= fixture_size[1] else 1
    minimum_distance = adaptive_min_distance(count, spacing_compensation)
    positions: list[list[float]] = []
    for object_index in range(count):
        accepted = None
        for _ in range(ATTEMPTS_PER_DISTANCE):
            long_position = round(rng.uniform(*LONG_AXIS_BOUNDS), 3)
            edge_position = round(
                rng.choice((-1.0, 1.0))
                * rng.uniform(*EDGE_DISTANCE_BOUNDS),
                3,
            )
            candidate = (
                [edge_position, long_position]
                if short_axis == 0
                else [long_position, edge_position]
            )
            if all(
                math.dist(candidate, existing) >= minimum_distance
                for existing in positions
            ):
                accepted = candidate
                break
        if accepted is None:
            raise ValueError(
                f"Could not place object {object_index + 1} after "
                f"{ATTEMPTS_PER_DISTANCE} attempts at adaptive minimum distance "
                f"{minimum_distance:.4f}."
            )
        positions.append(accepted)
    return positions


def build_object_configs(
    fixture: str,
    entries: list[tuple[str, str | None]],
    positions: list[list[float]],
    full_depth_region: bool = True,
    sink_ref: str | None = None,
) -> list[dict[str, Any]]:
    """Build exporter-compatible object dictionaries.

    Each entry is (category, instance-or-None); a pinned instance resolves to
    its registry mjcf path so the sampler spawns exactly that object.
    """
    if len(entries) != len(positions):
        raise ValueError("entries and positions must have the same length")

    counters: Counter[str] = Counter()
    objects = []
    for (category, instance), position in zip(entries, positions):
        counters[category] += 1
        size = [0.25, 0.25]
        obj_groups = (
            resolve_instance_mjcf_path(category, instance)
            if instance is not None
            else category
        )
        placement: dict[str, Any] = {
            "fixture": fixture,
            "size": size,
            "pos": position,
            "rotation": [0.0, 0.0],
        }
        if sink_ref is not None:
            # Let Counter.get_reset_regions use the instantiated sink's actual
            # style-dependent footprint. Its left_right regions exclude the
            # sink opening instead of approximating it from layout percentages.
            placement["sample_region_kwargs"] = {
                "ref": sink_ref,
                "loc": "left_right",
            }
        elif full_depth_region:
            # Islands with an interior sink/stove split their top into strip
            # regions; sampling in a strip clusters objects beside the sink and
            # out of robot reach. full_depth_region drops those strips so pos
            # normalizes over the accessible full-depth slab.
            placement["sample_region_kwargs"] = {"full_depth_region": True}
        objects.append(
            {
                "type": "object",
                "name": f"{category}_{counters[category]}",
                "obj_groups": obj_groups,
                "graspable": True,
                "placement": placement,
            }
        )
    return objects


def generate_objects(
    fixture: str,
    cups: int,
    fruits: int,
    seed: int,
    fixture_size: tuple[float, float],
    spec: str | None = None,
    spacing_compensation: int = 1,
    sink_ref: str | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic object configurations.

    When ``spec`` is given it fully determines categories (and optionally
    pinned instances) in order; ``cups``/``fruits`` are ignored. Otherwise
    categories are sampled from the legacy cup/fruit groups.
    """
    rng = random.Random(seed)
    if spec is not None:
        entries = parse_spec(spec)
    else:
        if cups < 0 or fruits < 0:
            raise ValueError("--cups and --fruits must be non-negative")
        if cups + fruits == 0:
            raise ValueError("at least one object must be requested")
        entries = [(rng.choice(CUP_GROUPS), None) for _ in range(cups)]
        entries.extend((rng.choice(FRUIT_GROUPS), None) for _ in range(fruits))
    positions = sample_positions(
        rng, len(entries), fixture_size, spacing_compensation=spacing_compensation
    )
    return build_object_configs(fixture, entries, positions, sink_ref=sink_ref)


def generate_document(
    layout: Any,
    cups: int,
    fruits: int,
    seed: int,
    fixture: str | None,
    spec: str | None = None,
    spacing_compensation: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Generate the complete YAML document from loaded layout data."""
    if not isinstance(layout, (dict, list)):
        raise ValueError("layout YAML must contain a mapping or list")
    selected_fixture = choose_fixture(collect_fixture_names(layout), fixture)
    fixture_sizes = collect_fixture_sizes(layout)
    if selected_fixture not in fixture_sizes:
        known = ", ".join(sorted(fixture_sizes)) if fixture_sizes else "none"
        raise ValueError(
            f"Could not determine planar size for fixture '{selected_fixture}'. "
            f"Fixtures with numeric sizes: {known}."
        )
    fixture_types = collect_fixture_types(layout)
    interior_refs = collect_interior_fixture_refs(layout)
    interior_ref = interior_refs.get(selected_fixture)
    sink_ref = (
        interior_ref
        if interior_ref is not None and fixture_types.get(interior_ref) == "sink"
        else None
    )
    return {
        "objects": generate_objects(
            selected_fixture,
            cups,
            fruits,
            seed,
            fixture_size=fixture_sizes[selected_fixture],
            spec=spec,
            spacing_compensation=spacing_compensation,
            sink_ref=sink_ref,
        ),
    }


def resolve_output_path(layout_yaml: Path, output: Path | None) -> Path:
    """Return an explicit output or derive one in the custom-layout directory."""
    if output is not None:
        return output
    return DEFAULT_OUTPUT_DIR / f"{layout_yaml.stem}_objects.yaml"


def generate_file(
    layout_yaml: Path,
    output: Path,
    cups: int,
    fruits: int,
    seed: int,
    fixture: str | None,
    spec: str | None = None,
    spacing_compensation: int = 1,
) -> tuple[str, int]:
    """Read a layout, write objects YAML, and self-validate the result."""
    with layout_yaml.open("r", encoding="utf-8") as stream:
        layout = yaml.safe_load(stream)
    if layout is None:
        raise ValueError(f"layout YAML is empty: {layout_yaml}")

    document = generate_document(
        layout,
        cups,
        fruits,
        seed,
        fixture,
        spec=spec,
        spacing_compensation=spacing_compensation,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)

    with output.open("r", encoding="utf-8") as stream:
        validated = yaml.safe_load(stream)
    if not isinstance(validated, dict) or not isinstance(validated.get("objects"), list):
        raise ValueError(f"generated YAML failed self-validation: {output}")
    if not validated["objects"]:
        raise ValueError(f"generated YAML contains no objects: {output}")

    selected_fixture = validated["objects"][0]["placement"]["fixture"]
    return selected_fixture, len(validated["objects"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RoboCasa --objects-yaml content from a layout YAML."
    )
    parser.add_argument("--layout-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cups", type=int, default=3)
    parser.add_argument("--fruits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixture")
    parser.add_argument("--spacing-compensation", type=int, choices=(1, 2), default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RoboCasa --objects-yaml content from a layout YAML."
    )
    parser.add_argument("--layout-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cups", type=int, default=3)
    parser.add_argument("--fruits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixture")
    parser.add_argument(
        "--spacing-compensation",
        type=int,
        choices=(1, 2),
        default=1,
        help="extra virtual slot count used by adaptive normalized spacing",
    )
    parser.add_argument(
        "--spec",
        help=(
            "ordered object spec 'category[=instance][:count],...' "
            "(e.g. 'milk=milk_0,apple,mug:2,bowl=bowl_1'); overrides --cups/--fruits"
        ),
    )
    args = parser.parse_args(argv)
    output = resolve_output_path(args.layout_yaml, args.output)

    try:
        fixture, count = generate_file(
            args.layout_yaml,
            output,
            args.cups,
            args.fruits,
            args.seed,
            args.fixture,
            spec=args.spec,
            spacing_compensation=args.spacing_compensation,
        )
    except (OSError, yaml.YAMLError, ValueError) as error:
        parser.error(str(error))

    print(f"Generated {count} objects on fixture '{fixture}': {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
