"""
Set the version of ETSAM and the ChimeraX plugin to the GitHub release's version.

All release artifacts share the release tag's version (v1.2.3 -> 1.2.3): the
installers (from ETSAM's pyproject.toml), the ChimeraX bundle (from its
pyproject.toml, set here) and the napari plugin (from the tag, via setuptools-scm).

Usage: python set_version.py <tag>    e.g. python set_version.py "$GITHUB_REF_NAME"
Prints the version.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FILES = [ROOT / "pyproject.toml", ROOT / "external_plugins" / "chimerax-etsam" / "pyproject.toml"]

tag = sys.argv[1] if len(sys.argv) == 2 else sys.exit(__doc__)
version = tag.removeprefix("v")
if not re.fullmatch(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", version):
    sys.exit(f"Release tag {tag!r} is not a version such as v1.2.3.")
for path in FILES:
    text = path.read_text()
    new, count = re.subn(r'(?m)^version = "[^"]*"$', f'version = "{version}"', text, count=1)
    if count != 1:
        sys.exit(f"No version line in {path}")
    path.write_text(new)
print(version)
