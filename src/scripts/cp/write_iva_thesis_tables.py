"""Render the IVA thesis tables (.tex) from the saved episode_statistics.json.

The JSON comes from src/scripts/cp/analyse_episode_statistics.py; this script only formats it,
so style changes never re-run the evaluation. Table design and marking rule:
say_no.cp.thesis_tables.
"""

import argparse
import json
from pathlib import Path

from say_no.cp.thesis_tables import write_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-json", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    write_all(json.loads(a.stats_json.read_text(encoding="utf-8")), a.out_dir)


if __name__ == "__main__":
    main()
