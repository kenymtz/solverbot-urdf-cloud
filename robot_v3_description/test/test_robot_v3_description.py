"""colcon test: validación estática completa de robot_v3_description (tools/validate_v3.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
import validate_v3  # noqa: E402


def test_static_validation():
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rep = validate_v3.Report()
    validate_v3.validate(pkg, rep, repo_root=None)
    assert not rep.failed, '\n'.join(f'{name}: {detail}' for name, _, detail in rep.failed)
