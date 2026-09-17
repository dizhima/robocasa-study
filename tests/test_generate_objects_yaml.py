from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "generate_objects_yaml.py"
SPEC = spec_from_file_location("generate_objects_yaml", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
generate_objects_yaml = module_from_spec(SPEC)
SPEC.loader.exec_module(generate_objects_yaml)


def test_portable_mjcf_path_removes_machine_specific_prefix():
    source = (
        r"C:\Users\someone\src\robocasa\robocasa\models\assets\objects"
        r"\objaverse\apple\apple_2\model.xml"
    )

    assert generate_objects_yaml.portable_mjcf_path(source) == (
        "robocasa/models/assets/objects/objaverse/apple/apple_2/model.xml"
    )
