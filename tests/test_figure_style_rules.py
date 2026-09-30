"""Static style rules for the code that draws thesis figures: sizes, colours and saving come from
say_no.utils.figstyle and say_no.utils.palette, never from literals in the producers."""
import ast
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

_spec = importlib.util.spec_from_file_location("render_thesis_figures", "src/scripts/render_thesis_figures.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

LIBRARIES = ["src/perturbation/figures.py", "src/cp/figure.py", "src/utils/wandb_plot.py", "src/perturbation/violins.py"]
PRODUCERS = sorted({e.script for e in R.REGISTRY.values()} | set(LIBRARIES))
# the legacy (non-thesis) YAML path scales tueplots bundles itself
FONT_SCALE_ALLOWED = {"src/utils/wandb_plot.py", "src/scripts/render_plot.py"}
SIZE_KWARGS = {"fontsize", "labelsize", "size", "titlesize"}
GREY = re.compile(r"0?\.\d+")
HEX = re.compile(r"#[0-9a-fA-F]{6}")


def _tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)


@pytest.mark.parametrize("path", PRODUCERS)
def test_no_direct_savefig(path):
    calls = [n.lineno for n in ast.walk(_tree(path))
             if isinstance(n, ast.Attribute) and n.attr == "savefig"]
    assert not calls, f"{path}:{calls} saves directly; use figstyle.save_figure / save_pinned"


@pytest.mark.parametrize("path", sorted(set(PRODUCERS) - FONT_SCALE_ALLOWED))
def test_no_font_scale(path):
    names = [n.lineno for n in ast.walk(_tree(path))
             if (isinstance(n, ast.Name) and "FONT_SCALE" in n.id.upper())
             or (isinstance(n, ast.keyword) and n.arg == "font_scale")]
    assert not names, f"{path}:{names} sets its own font scale; the profile decides"


@pytest.mark.parametrize("path", PRODUCERS)
def test_no_numeric_font_sizes(path):
    bad = [n.lineno for n in ast.walk(_tree(path)) if isinstance(n, ast.keyword) and n.arg in SIZE_KWARGS
           and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (int, float))]
    assert not bad, f"{path}:{bad} literal font size"


@pytest.mark.parametrize("path", PRODUCERS)
def test_no_colour_literals(path):
    bad = [(n.lineno, n.value) for n in ast.walk(_tree(path)) if isinstance(n, ast.Constant)
           and isinstance(n.value, str) and (GREY.fullmatch(n.value) or HEX.fullmatch(n.value))]
    assert not bad, f"{path}: colour literals {bad}; name them in say_no.utils.palette"


def _thesis_yamls():
    return sorted({e.argv[0] for e in R.REGISTRY.values() if e.script.endswith("render_plot.py")})


@pytest.mark.parametrize("path", _thesis_yamls())
def test_thesis_yaml_uses_the_thesis_profile(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert cfg["style"] == "thesis"
    assert not {"bundle", "bundle_kwargs", "font_scale", "figsize"} & set(cfg), path
    colours = [s["color"] for s in cfg.get("series", []) + cfg.get("points", []) if "color" in s]
    colours += cfg.get("bar_colors", [])
    assert not [c for c in colours if isinstance(c, str) and c.startswith("#")], f"{path}: raw hex colour"
