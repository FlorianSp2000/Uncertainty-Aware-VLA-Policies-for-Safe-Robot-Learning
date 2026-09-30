"""Export the raw actions of every episode in a rollout cache to one small .npz (key -> (T, 7)).

The host-side analysis needs the actions for an action-only baseline (how far each action lies
outside the finetune successes' bounds) without downloading the frames. Reads only the
``actions`` member of each cache entry.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    index = json.loads((data_dir / "index.json").read_text())
    out = {}
    for r in index:
        a = np.load(data_dir / r["pool"] / f"{r['key']}.npz", allow_pickle=True)["actions"]
        if a.shape != (r["length"], 7):
            raise ValueError(f"{r['key']}: actions {a.shape}, index length {r['length']}")
        out[r["key"]] = a.astype(np.float32)
    np.savez_compressed(args.out, **out)
    print(f"wrote {len(out)} episodes to {args.out}")


if __name__ == "__main__":
    main()
