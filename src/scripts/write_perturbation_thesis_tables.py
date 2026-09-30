"""The Bridge/Fractal thesis tables (bf_*.tex of say_no.perturbation.thesis_tables), rewritten from the
analysis JSON (perturbation_thesis.json of analyse_perturbation_thesis.py) without rerunning the
bootstrap: for changes of wording only. Invocation: docs/REPRODUCE.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from say_no.perturbation import thesis_tables as T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", required=True, type=Path, help="perturbation_thesis.json")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))
    latency = Path(d["args"]["latency"])
    if not latency.exists():
        raise FileNotFoundError(latency)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    T.write_all(d["sets"], d["shift_spearman"], d["controls_caption"], d["roles"], latency, args.out_dir)
    print(f"wrote {sorted(T.FILES.values())} to {args.out_dir}")


if __name__ == "__main__":
    main()
