"""Convert SAFE's OpenVLA-on-WidowX rollouts into the per-episode rollout cache.

Source: ``openvla_widowx.zip`` released with SAFE (gu2025) -- 532 rollouts of the
OXE-pretrained OpenVLA on a real WidowX in SAFE's lab, 8 tasks, success label in the file name
(``--succ0/1``) and again inside the pickle (``episode_success``). Per episode:

* ``.mp4``  -- one 640x480 frame per policy step, 5 Hz, fixed 50 steps
* ``.csv``  -- one row per policy step: the executed action ``dx..dgripper`` in Bridge units
  (OpenVLA un-normalises with the Bridge statistics), plus token entropies we do not use
* ``.pkl``  -- hidden states, ``task_description``, ``episode_success`` (needs torch to load)

The output is the cache format ``octo/data/real_robot.py`` already reads (index.json + one
``.npz`` per episode holding PNG frames, raw actions and per-frame prompts), so the finetune,
eval and analysis code is shared with the LIBERO rollouts rather than copied.

The split is written into the index as the episode's ``pool``: successes are divided, per task,
into ``finetune`` / ``calib`` / ``test``; every failure is ``test``. Failures are never trained or
calibrated on -- the claim under test is that the critic flags them without having seen one.

Frames are resized once here to the 256x256 the critic reads (Bridge's own 640x480 frames were
squashed to that size the same way), with area interpolation so the downscale does not alias.

Runs in ``train_q_ood.sif`` (the pickles need torch). Run by
``slurm/convert-safe-rollouts.sbatch``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ACTION_COLUMNS = ["action/dx", "action/dy", "action/dz", "action/droll", "action/dpitch",
                  "action/dyaw", "action/dgripper"]
IMAGE_SIZE = 256
FILE_RE = re.compile(r"^(?P<stem>.+)--ep(?P<ep>\d+)--succ(?P<succ>[01])\.mp4$")


def task_of(folder: str) -> str:
    """``task_put_the_carrot_on_plate_2`` and ``put_the_carrot_on_plate_1`` -> ``put_the_carrot_on_plate``.

    SAFE recorded each task in several sessions, one folder each; one folder lacks the prefix.
    """
    return re.sub(r"_\d+$", "", folder.removeprefix("task_"))


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def read_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    return np.stack(frames)


def read_actions(path: Path) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["action/timestep"]) for r in rows]
    if steps != list(range(len(rows))):
        raise ValueError(f"{path}: timesteps are not 0..{len(rows) - 1}")
    return np.array([[float(r[c]) for c in ACTION_COLUMNS] for r in rows], dtype=np.float32)


def scan(raw_dir: Path) -> list[dict]:
    rows = []
    for mp4 in sorted(raw_dir.glob("*/*.mp4")):
        m = FILE_RE.match(mp4.name)
        if not m:
            raise ValueError(f"unexpected file name {mp4}")
        folder = mp4.parent.name
        rows.append(dict(folder=folder, task=task_of(folder), stem=mp4.name[:-4],
                         episode=int(m["ep"]), success=m["succ"] == "1",
                         source=f"{folder}/{mp4.name[:-4]}"))
    return rows


def assign_pools(rows: list[dict], n_finetune: int, n_calib: int, seed: int,
                 unseen_tasks: tuple = ()) -> None:
    """Split the successes per task in proportion to the requested totals; failures -> test.

    ``unseen_tasks``: every rollout of these tasks goes to test, so the critic is neither
    finetuned nor calibrated on them (SAFE's seen / unseen protocol).

    Largest-remainder allocation per task so the totals come out exact while every task keeps
    its share, then a seeded shuffle inside each task decides which episodes go where.
    """
    missing = set(unseen_tasks) - {r["task"] for r in rows}
    if missing:
        raise ValueError(f"unseen tasks {missing} not in the data")
    for r in rows:
        if r["task"] in unseen_tasks:
            r["pool"] = "test"
    succ = [r for r in rows if r["success"] and r["task"] not in unseen_tasks]
    n_succ = len(succ)
    if n_finetune + n_calib >= n_succ:
        raise ValueError(f"{n_finetune} finetune + {n_calib} calib leaves no test successes "
                         f"out of {n_succ}")
    tasks = sorted({r["task"] for r in succ})
    per_task = {t: sorted([r for r in succ if r["task"] == t], key=lambda r: r["source"])
                for t in tasks}

    def allocate(total: int, sizes: dict) -> dict:
        exact = {t: total * sizes[t] / n_succ for t in tasks}
        out = {t: int(np.floor(v)) for t, v in exact.items()}
        for t in sorted(tasks, key=lambda t: exact[t] - out[t], reverse=True)[:total - sum(out.values())]:
            out[t] += 1
        return out

    sizes = {t: len(v) for t, v in per_task.items()}
    k_ft = allocate(n_finetune, sizes)
    k_cal = allocate(n_calib, sizes)
    rng = np.random.default_rng(seed)
    for t in tasks:
        eps = per_task[t]
        if k_ft[t] + k_cal[t] >= len(eps):
            raise ValueError(f"task {t}: {len(eps)} successes cannot give {k_ft[t]} finetune + "
                             f"{k_cal[t]} calib and still leave one for test")
        order = rng.permutation(len(eps))
        for rank, i in enumerate(order):
            eps[i]["pool"] = ("finetune" if rank < k_ft[t]
                              else "calib" if rank < k_ft[t] + k_cal[t] else "test")
    for r in rows:
        if not r["success"]:
            r["pool"] = "test"


def _convert_one(args) -> dict:
    row, raw_dir, out_dir = args
    base = Path(raw_dir) / row["folder"] / row["stem"]
    import torch  # noqa: F401  -- the pickles hold torch tensors
    with open(f"{base}.pkl", "rb") as f:
        meta = pickle.load(f)
    if bool(meta["episode_success"]) != row["success"]:
        raise ValueError(f"{row['source']}: file name says success={row['success']}, "
                         f"pickle says {meta['episode_success']}")

    frames = read_frames(Path(f"{base}.mp4"))
    actions = read_actions(Path(f"{base}.csv"))
    if len(frames) != len(actions):
        raise ValueError(f"{row['source']}: {len(frames)} frames vs {len(actions)} action rows")
    if len(meta["hidden_states"]) != len(actions):
        raise ValueError(f"{row['source']}: {len(meta['hidden_states'])} hidden states vs "
                         f"{len(actions)} action rows")

    # Bridge instructions are lower case; SAFE's are title case ("Put the Red Block into the Pot").
    instruction = meta["task_description"].strip().lower()
    key = f"{row['task']}__ep{row['episode']:03d}__{row['folder']}__{'ok' if row['success'] else 'fail'}"
    dst = Path(out_dir) / row["pool"]
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst / f"{key}.npz",
        images=np.array([_png(f) for f in frames], dtype=object),
        actions=actions,
        prompts=np.array([instruction] * len(actions), dtype=object),
    )
    return dict(key=key, pool=row["pool"], condition=row["task"], source=row["source"],
                length=len(actions), instruction=instruction, success=row["success"],
                feasible=True, prompt_switches=[], corrupt=False, corrupt_reason=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="the unzipped openvla_widowx directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_finetune", type=int, required=True)
    ap.add_argument("--n_calib", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--unseen_tasks", default="",
                    help="comma-separated tasks held out of finetune and calibration (all -> test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="convert at most N episodes; preflight only")
    args = ap.parse_args()

    raw_dir, out = Path(args.raw_dir), Path(args.out_dir)
    rows = scan(raw_dir)
    unseen = tuple(t for t in args.unseen_tasks.split(",") if t)
    assign_pools(rows, args.n_finetune, args.n_calib, args.seed, unseen)
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    out.mkdir(parents=True, exist_ok=True)
    print(f"converting {len(rows)} episodes from {raw_dir}", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_convert_one, (r, str(raw_dir), str(out))) for r in rows]
        for n, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if n % 50 == 0:
                print(f"  {n}/{len(rows)}", flush=True)
    records.sort(key=lambda r: r["source"])
    (out / "index.json").write_text(json.dumps(records, indent=1))
    (out / "split.json").write_text(json.dumps(dict(
        seed=args.seed, n_finetune=args.n_finetune, n_calib=args.n_calib, limit=args.limit,
        unseen_tasks=list(unseen),
        pools={r["key"]: r["pool"] for r in records}), indent=1))

    instr = {}
    for r in records:
        instr.setdefault(r["condition"], set()).add(r["instruction"])
    print("\ninstructions per task:")
    for t, s in sorted(instr.items()):
        print(f"  {t:32s} {sorted(s)}")
    print("\npool x outcome per task:")
    for t in sorted(instr):
        c = {(p, o): sum(1 for r in records if r["condition"] == t and r["pool"] == p
                         and r["success"] == o)
             for p in ("finetune", "calib", "test") for o in (True, False)}
        print(f"  {t:32s} finetune {c[('finetune', True)]:3d}  calib {c[('calib', True)]:3d}  "
              f"test succ {c[('test', True)]:3d}  test fail {c[('test', False)]:3d}")
    print(f"\nwrote {out / 'index.json'} ({len(records)} episodes)")


if __name__ == "__main__":
    main()
