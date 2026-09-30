"""Figure registries: which producer command draws which file, at which include width.

Instance: src/scripts/render_thesis_figures.py. The registry is the single place a figure's
invocation is written down.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    script: str                 # producer, relative to the repo root
    argv: tuple[str, ...]
    output: str                 # file the producer writes
    width: float | None         # include width as a fraction of \textwidth; None = not checked
    wandb: bool = False         # pulls from the W&B API
    slow: bool = False          # minutes, or a >0.5 GB dump

    def command(self) -> tuple[str, ...]:
        return (self.script, *self.argv)


def run(registry: dict[str, Entry], argv: list[str] | None = None, *, description: str,
        copy_allowed: bool) -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="registry names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print name, width, command; run nothing")
    if copy_allowed:
        ap.add_argument("--copy", type=Path, metavar="DIR", help="copy each built file to DIR/<name>")
    args = ap.parse_args(argv)
    if not Path("src/scripts").is_dir():
        raise RuntimeError("run from the repository root")
    names = args.only or list(registry)
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise ValueError(f"not in the registry: {unknown}")

    if args.dry_run:
        for n in names:
            e = registry[n]
            print(f"{n}\twidth={e.width}\t{' '.join(e.command())}")
        return

    done: set[tuple[str, ...]] = set()
    for n in names:
        e = registry[n]
        if e.command() in done:
            continue
        t0 = time.time()
        subprocess.run([sys.executable, *e.command()], check=True)
        done.add(e.command())
        print(f"[{time.time() - t0:5.0f}s] {' '.join(e.command())}", flush=True)
    for n in names:
        out = Path(registry[n].output)
        if not out.exists():
            raise FileNotFoundError(f"{n}: producer did not write {out}")
        if getattr(args, "copy", None):
            shutil.copyfile(out, args.copy / n)
            print(f"copied {out} -> {args.copy / n}")
