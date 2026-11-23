"""Guards for rules in CLAUDE.md's "Must never happen" table that no other test covers (#81)."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "argus_proof"

# AGPL-3.0 packages: importing one in-process puts proof under the AGPL. NudeNet is reached over HTTP through
# argus_cortex.moderation.NudeNetClient. argus-ci's `agpl-import` suite rule catches the same thing fleet-wide.
AGPL = {"nudenet", "ultralytics"}

# Optional-extra dependencies that must stay lazy: importing any proof module must not load them, so the core install
# works without them and each adapter reports is_available() instead of failing at import.
HEAVY = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "insightface",
    "onnxruntime",
    "pyiqa",
    "torchmetrics",
    "ImageReward",
    "fiftyone",
    "umap",
    "optuna",
    "polars",
    "krippendorff",
    "PIL",
    "imagehash",
)

# Model weights (Llama Guard, NudeNet, checkpoints) are never committed or shipped: they are fetched at run time
# under their own licences.
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".h5", ".pb", ".tflite")


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
        ):
            roots.add(node.args[0].value.split(".")[0])
    return roots


def test_no_agpl_package_is_imported_in_process() -> None:
    offenders = {
        str(path.relative_to(ROOT)): sorted(_imported_roots(ast.parse(path.read_text(encoding="utf-8"))) & AGPL)
        for path in SRC.rglob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_argus_puppet_is_reached_over_http_only() -> None:
    # Proof speaks puppet's wire (scoring/scorers/pose.py): the package would bring its own argus-cortex pin and
    # onnxruntime, and its weights belong in its service.
    importers = [
        str(path.relative_to(ROOT))
        for path in [*SRC.rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
        if "argus_puppet" in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert importers == []
    assert "argus-puppet" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _requirement_name(requirement: str) -> str | None:
    """The distribution name of a PEP 508 requirement, normalised the way pip compares names (PEP 503)."""
    token = requirement.split(";", 1)[0].strip()
    token = re.split(r"\s+", token, maxsplit=1)[0]
    token = token.split("[", 1)[0]
    token = re.split(r"[<>=!~]", token, maxsplit=1)[0].strip().strip("'\"")
    if not token:
        return None
    return re.sub(r"[-_.]+", "-", token).lower()


def _pyproject_dependency_names(pyproject: Path) -> set[str]:
    """Dependency names from project tables, dependency-groups and uv sources."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    raw: list[str] = []
    project = data.get("project") or {}
    raw.extend(spec for spec in project.get("dependencies") or [] if isinstance(spec, str))
    for group in (project.get("optional-dependencies") or {}).values():
        raw.extend(spec for spec in group if isinstance(spec, str))
    for group in (data.get("dependency-groups") or {}).values():
        if isinstance(group, list):
            raw.extend(spec for spec in group if isinstance(spec, str))
    sources = ((data.get("tool") or {}).get("uv") or {}).get("sources") or {}
    if isinstance(sources, dict):
        raw.extend(name for name in sources if isinstance(name, str))
    return {name for spec in raw if (name := _requirement_name(spec))}


def test_argus_echo_is_reached_over_http_only() -> None:
    # Proof speaks echo's wire (backends/echo.py): the package pins an argus-cortex proof can't share, and its engines
    # belong in their own images. PEP 503 folds argus_echo, Argus-Echo and argus.echo into the same name.
    importers = [
        str(path.relative_to(ROOT))
        for path in [*SRC.rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
        if "argus_echo" in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert importers == []
    assert "argus-echo" not in _pyproject_dependency_names(ROOT / "pyproject.toml")


def test_echo_dependency_spellings_normalise_to_one_name(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                'dependencies = ["Argus_Echo >= 1"]',
                "",
                "[project.optional-dependencies]",
                'extra = ["argus.echo[cpu]"]',
                "",
                "[dependency-groups]",
                'dev = ["argus-echo"]',
                "",
                "[tool.uv.sources]",
                'argus_echo = { path = "../argus-echo" }',
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert _pyproject_dependency_names(pyproject) == {"argus-echo"}


def test_the_import_scan_sees_every_form() -> None:
    tree = ast.parse(
        "import nudenet\nfrom ultralytics.models import YOLO\nimportlib.import_module('dyn.x')\n__import__('a')"
    )
    # One distinct root per form, so losing any one form (the dynamic import above all) fails here.
    assert _imported_roots(tree) == {"nudenet", "ultralytics", "dyn", "a"}


def test_importing_every_module_loads_no_heavy_dependency() -> None:
    # A fresh interpreter, so modules other tests imported don't count.
    script = (
        "import importlib, pkgutil, sys, argus_proof\n"
        "for m in pkgutil.walk_packages(argus_proof.__path__, 'argus_proof.'):\n"
        "    importlib.import_module(m.name)\n"
        f"print(','.join(h for h in {HEAVY!r} if h in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120)
    assert out.stdout.strip() == "", f"imported at module load: {out.stdout.strip()}"


def test_no_model_weights_are_committed() -> None:
    # -z: raw NUL-separated paths. Without it git quotes and C-escapes a name with
    # non-ASCII or special characters ("caf\303\251.safetensors"), hiding its suffix.
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
    ).stdout.split("\0")

    assert [p for p in tracked if p.lower().endswith(WEIGHT_SUFFIXES)] == []


def test_the_image_fetches_no_model_weights() -> None:
    # Llama Guard's licence restricts redistributing its weights, and proof's MIT image must not carry AGPL ones
    # (NudeNet's live in argus-retina's image): the image ships code only, and a host fetches weights itself. These
    # are the ways a Dockerfile would pull them in.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").lower()
    banned = ("huggingface", "hf_hub", "snapshot_download", "hf download", "llama", "nudenet", *WEIGHT_SUFFIXES)
    assert [b for b in banned if b in dockerfile] == []


def test_every_test_claude_md_cites_exists() -> None:
    # The "Must never happen" table names its checks; a renamed or deleted test must not leave a row pointing at
    # nothing, which would read as enforced.
    import re

    cited = set(re.findall(r"`(test_\w+)::(test_\w+)`", (ROOT / "CLAUDE.md").read_text(encoding="utf-8")))
    assert cited, "CLAUDE.md cites no tests: has the table moved?"
    missing = [
        f"{module}::{name}"
        for module, name in sorted(cited)
        if not (path := ROOT / "tests" / f"{module}.py").is_file()
        or not re.search(rf"^def {name}\(", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert missing == []
