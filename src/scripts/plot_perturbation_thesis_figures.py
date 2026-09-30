"""The three AUROC figures of the Bridge/Fractal perturbation analysis, redrawn from its analysis JSON
(perturbation_thesis.json of analyse_perturbation_thesis.py) without rerunning the bootstrap.
Invocation: src/scripts/render_thesis_figures.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from say_no.perturbation import figures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, type=Path, help="perturbation_thesis.json")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures.write_all(d["sets"], d["roles"], args.out_dir, f"{args.json.as_posix()} · {Path(__file__).name}",
                      d["args"]["frame_readouts"])
    print(f"wrote {sorted(figures.FILES.values())} to {args.out_dir}")


if __name__ == "__main__":
    main()
