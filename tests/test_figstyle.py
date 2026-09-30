"""Contract of say_no.utils.figstyle: the two figure profiles (thesis, ICRA) and the one saver."""
import hashlib
from contextlib import contextmanager
import re
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from tueplots import bundles

from say_no.utils import figstyle as FS
from say_no.utils.wandb_plot import get_style

ICML = bundles.icml2024(usetex=False, column="half")
FRACTIONS = (FS.HALF, 0.62, 0.65, 0.75, FS.FULL)


@contextmanager
def _fig(width=FS.FULL, height=2.0, profile=FS.THESIS):
    """A small figure, yielded inside its style context (saving happens there too)."""
    with plt.rc_context(FS.style(profile, width=width, height=height)):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], label="series")
        ax.set_xlabel("x label")
        ax.legend()
        yield fig
        plt.close(fig)


def _mediabox_width_pt(pdf):
    m = re.search(rb"/MediaBox \[\s*([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", pdf.read_bytes())
    return float(m.group(3)) - float(m.group(1))


def test_textwidth_is_the_thesis_layout():
    assert FS.THESIS.column_in == pytest.approx(398.33864 / 72.27)
    assert FS.THESIS.text_in == FS.THESIS.column_in  # scrbook, one column
    assert (FS.ICRA.column_in, FS.ICRA.text_in) == (3.4, 7.0)


@pytest.mark.parametrize("frac", FRACTIONS)
def test_width_fraction_sets_figure_and_mediabox(tmp_path, frac):
    with _fig(width=frac) as fig:
        assert fig.get_size_inches()[0] == pytest.approx(frac * FS.THESIS.column_in)
        FS.save_figure(fig, tmp_path / "f.pdf", "test provenance")
    assert _mediabox_width_pt(tmp_path / "f.pdf") == pytest.approx(frac * FS.THESIS.column_in * 72, abs=0.01)


def test_thesis_font_sizes_are_icml2024_times_scale():
    rc = FS.style(FS.THESIS, width=FS.FULL, height=2.0)
    s = FS.THESIS.font_scale
    assert s == 1.3
    for key in ("axes.labelsize", "axes.titlesize", "font.size", "xtick.labelsize", "ytick.labelsize"):
        assert rc[key] == pytest.approx(ICML[key] * s)
    assert rc["legend.fontsize"] == pytest.approx(ICML["legend.fontsize"] * s)  # legend from rc on the thesis path
    assert FS.THESIS.legend_pt is None
    assert set(FS.thesis_text_sizes()) == {ICML["axes.labelsize"] * s, ICML["xtick.labelsize"] * s}


def test_font_family_is_one_profile_parameter():
    rc = FS.style(FS.THESIS, width=FS.FULL, height=2.0)
    assert rc["font.family"] == "serif"
    assert list(rc["font.serif"]) == list(FS.THESIS.serif)


def test_icra_rc_equals_the_pinned_paper_rc():
    """The ICRA outputs are byte-pinned; their rc must stay what the paper code always used."""
    rc = FS.style(FS.ICRA, width=1.0, height=2.0)
    ref = get_style("icml2024", usetex=False, column="half", font_scale=1.3)
    assert rc["figure.figsize"] == (3.4, 2.0)
    assert {k: v for k, v in rc.items() if k != "figure.figsize"} == \
           {k: v for k, v in ref.items() if k != "figure.figsize"}
    assert FS.ICRA.legend_pt == 9.0


def test_thesis_provenance_is_metadata_not_ink(tmp_path):
    """Run ids and paths stay off the page: the record goes into the PDF Subject (and PNG Description)."""
    from PIL import Image
    prov = "checkpoint x · step 1 · script.py"
    with _fig(width=FS.HALF) as fig:
        FS.save_figure(fig, tmp_path / "f.pdf", prov, preview=True)
        assert not [t for t in fig.texts if t.get_gid() == "stamp"]
        assert not [t for t in fig.texts if "checkpoint x" in t.get_text()]
    pdf = (tmp_path / "f.pdf").read_bytes()
    assert b"/Subject" in pdf  # non-ASCII record: a UTF-16BE string
    assert all(w.encode("utf-16-be") in pdf for w in ("checkpoint x", "rendered"))
    with Image.open(tmp_path / "f.png") as im:
        assert im.info["Description"] == FS.provenance_record(prov)


def test_stamp_is_inside_the_canvas_at_stamp_size():
    """Visible footer for outputs outside the thesis (e.g. diagnostic contact sheets)."""
    with _fig(width=FS.HALF) as fig:
        t = FS.stamp(fig, "checkpoint x · step 1 · script.py")
        assert t.get_fontsize() == FS.STAMP_PT
        assert "checkpoint x" in t.get_text() and "rendered" in t.get_text()
        fig.canvas.draw()
        bb, canvas = t.get_window_extent(), fig.bbox
    assert bb.x0 >= canvas.x0 and bb.y0 >= canvas.y0 and bb.x1 <= canvas.x1 and bb.y1 <= canvas.y1


def test_stamp_that_overflows_the_canvas_raises():
    with _fig(width=FS.HALF) as fig, pytest.raises(ValueError, match="stamp"):
        FS.stamp(fig, "x" * 400)


def test_provenance_is_required(tmp_path):
    with _fig() as fig, pytest.raises(ValueError):
        FS.save_figure(fig, tmp_path / "f.pdf", "")


def test_saving_outside_the_style_context_raises(tmp_path):
    with _fig() as fig:
        pass
    with pytest.raises(RuntimeError, match="rc_context"):
        FS.save_figure(fig, tmp_path / "f.pdf", "p")


def test_bytes_are_deterministic(tmp_path):
    shas = []
    for i in range(2):
        with _fig() as fig:
            FS.save_figure(fig, tmp_path / f"f{i}.pdf", "same provenance")
        shas.append(hashlib.sha256((tmp_path / f"f{i}.pdf").read_bytes()).hexdigest())
    assert shas[0] == shas[1]
    assert b"/CreationDate" not in (tmp_path / "f0.pdf").read_bytes()


def test_preview_png_on_request(tmp_path):
    from PIL import Image
    with _fig(width=FS.HALF, height=2.0) as fig:
        FS.save_figure(fig, tmp_path / "f.pdf", "p", preview=True)
    with Image.open(tmp_path / "f.png") as im:
        assert im.size == (round(FS.HALF * FS.THESIS.column_in * FS.PREVIEW_DPI), round(2.0 * FS.PREVIEW_DPI))
    with _fig() as fig:
        FS.save_figure(fig, tmp_path / "g.pdf", "p")
    assert not (tmp_path / "g.png").exists()


@pytest.mark.skipif(shutil.which("pdffonts") is None, reason="pdffonts (poppler) not installed")
def test_pdf_embeds_body_font_as_truetype(tmp_path):
    """Thesis figures print in the body font (Palatino), text and maths, and embed no Type 3 fonts."""
    with _fig() as fig:
        fig.text(0.5, 0.5, r"score $z_t > \tau$")
        FS.save_figure(fig, tmp_path / "f.pdf", "p")
    out = subprocess.run(["pdffonts", str(tmp_path / "f.pdf")], capture_output=True, text=True, check=True).stdout
    fonts = out.splitlines()[2:]
    assert fonts, "no font embedded"
    assert all("Palatino" in line or "Pagella" in line for line in fonts), out
    assert not any("Type 3" in line for line in fonts), out


def test_save_pinned_is_plain_savefig(tmp_path):
    """ICRA / legacy outputs: exactly fig.savefig(path, **kwargs), no stamp, no metadata change."""
    import os
    os.environ["SOURCE_DATE_EPOCH"] = "1767225600"
    try:
        with _fig(profile=FS.ICRA, width=1.0) as a:
            FS.save_pinned(a, tmp_path / "a.pdf", bbox_inches="tight")
        with _fig(profile=FS.ICRA, width=1.0) as b:
            b.savefig(tmp_path / "b.pdf", bbox_inches="tight")
    finally:
        del os.environ["SOURCE_DATE_EPOCH"]
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()
    assert not [t for t in a.texts if t.get_gid() == "stamp"]
