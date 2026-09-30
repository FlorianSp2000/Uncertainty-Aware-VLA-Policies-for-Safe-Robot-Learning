"""Convert pi0-FAST LIBERO-10 rollouts (SAFE format) into the per-episode cache the real-robot
path reads.

Source: HF ``oldTOM/pi0-libero-rollouts`` (``pi0fast-libero_10.zip``), 500 simulated rollouts,
10 tasks, written by SAFE's rollout code:

* ``env_records/task{k}--ep{i}--succ{0,1}.{mp4,pkl}`` -- agent-view video, one 224x224 frame per
  env step from the end of the 10 wait steps; the pickle carries ``task_description``,
  ``episode_success``, ``end_step``, ``replan_steps`` (5), ``num_steps_wait`` (10)
* ``policy_records/step_*--task_{k}--ep_{i}--t_{t}--meta.pkl`` -- one per policy call at env
  step t = 10, 15, ...; ``actions`` is the (1, 10, 7) chunk, of which the first 5 were executed

One cache frame per policy call: the frame at env step t and the first action of the chunk
emitted there -- the action executed at that frame. Five env steps per frame keeps an episode at
28-104 frames, inside the horizon of gamma = 0.98 (the WidowX cache is strided for the same
reason).

Gripper: LIBERO commands -1 = open, +1 = close; Bridge/Fractal, on which the critic was
pretrained, use 1 = open, 0 = close. Mapped here, ``(1 - g) / 2``, so the binarised {-1, +1} the
critic sees keeps the pretraining meaning. Pose dimensions stay raw; the loader normalises them
with the bounds of the finetune successes (LIBERO's action scale is not Bridge's).

Every failure in this set is a timeout (520 env steps) while successes end at 150-450, so episode
length alone separates the classes. That is handled in the analysis (common-prefix evaluation),
not here; the cache keeps full episodes.

The split is written into the index as ``pool`` exactly as in ``convert_safe_rollouts.py``.
Runs in ``train_q_ood.sif`` on the login node (cv2 on the compute nodes picks up an
incompatible host libGL). Invoked by ``slurm/convert-libero-rollouts.sbatch``.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.convert_safe_rollouts import _png, assign_pools  # noqa: E402

IMAGE_SIZE = 256
ENV_RE = re.compile(r"^task(?P<task>\d+)--ep(?P<ep>\d+)--succ(?P<succ>[01])\.pkl$")
REC_RE = re.compile(r"--task_(?P<task>\d+)--ep_(?P<ep>\d+)--t_(?P<t>\d+)--meta\.pkl$")


class _Unpickler(pickle.Unpickler):
    """The policy records were pickled under numpy 2 (``numpy._core``); the container has 1.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def read_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_CUBIC))
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames decoded")
    return np.stack(frames)


def scan(raw_dir: Path) -> list[dict]:
    records = defaultdict(dict)
    for p in (raw_dir / "policy_records").glob("*--meta.pkl"):
        m = REC_RE.search(p.name)
        if not m:
            raise ValueError(f"unexpected policy record name {p.name}")
        records[(int(m["task"]), int(m["ep"]))][int(m["t"])] = p
    rows = []
    for p in sorted((raw_dir / "env_records").glob("*.pkl")):
        m = ENV_RE.match(p.name)
        if not m:
            raise ValueError(f"unexpected env record name {p.name}")
        task, ep = int(m["task"]), int(m["ep"])
        rows.append(dict(task_id=task, task=f"task{task}", episode=ep, success=m["succ"] == "1",
                         stem=p.name[:-4], source=f"env_records/{p.name[:-4]}",
                         policy_records={t: str(q) for t, q in records[(task, ep)].items()}))
    return rows


def _convert_one(args) -> dict:
    row, raw_dir, out_dir = args
    env = pickle.load(open(Path(raw_dir) / "env_records" / f"{row['stem']}.pkl", "rb"))
    if bool(env["episode_success"]) != row["success"]:
        raise ValueError(f"{row['stem']}: file name success={row['success']}, pickle "
                         f"{env['episode_success']}")
    wait, replan = env["num_steps_wait"], env["replan_steps"]
    frames = read_frames(Path(raw_dir) / "env_records" / f"{row['stem']}.mp4")
    # One frame per env step from the first policy call on; step_done is recorded per frame.
    if len(env["step_done"]) != len(frames):
        raise ValueError(f"{row['stem']}: {len(frames)} frames vs {len(env['step_done'])} step_done")

    calls = sorted(row["policy_records"])
    expected = list(range(wait, wait + replan * env["model_infer_times"], replan))
    if calls != expected:
        raise ValueError(f"{row['stem']}: policy calls {calls[:3]}..{calls[-3:]} "
                         f"({len(calls)}), expected {len(expected)} from t={wait} every {replan}")
    actions, prompts = [], []
    for t in calls:
        rec = _Unpickler(open(row["policy_records"][t], "rb")).load()
        chunk = np.asarray(rec["actions"], dtype=np.float32)
        if chunk.shape != (1, 10, 7):
            raise ValueError(f"{row['stem']} t={t}: action chunk {chunk.shape}")
        # Flat keys ("run/timestep"), not a nested dict.
        if rec["run/timestep"] != t or rec["run/task_id"] != row["task_id"]:
            raise ValueError(f"{row['stem']} t={t}: record says timestep {rec['run/timestep']}, "
                             f"task {rec['run/task_id']}")
        a = chunk[0, 0].copy()
        a[6] = (1.0 - a[6]) / 2.0  # LIBERO -1 open / +1 close -> Bridge 1 open / 0 close
        actions.append(a)
        prompts.append(rec["prompt"])
    idx = np.array(calls) - wait
    if idx[-1] >= len(frames):
        raise ValueError(f"{row['stem']}: last call at frame {idx[-1]}, video has {len(frames)}")
    if len(set(prompts)) != 1 or prompts[0] != env["task_description"]:
        raise ValueError(f"{row['stem']}: prompts {set(prompts)} vs {env['task_description']!r}")

    instruction = env["task_description"].strip().lower()
    key = f"{row['task']}__ep{row['episode']:03d}__{'ok' if row['success'] else 'fail'}"
    dst = Path(out_dir) / row["pool"]
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / f"{key}.npz",
                        images=np.array([_png(frames[i]) for i in idx], dtype=object),
                        actions=np.stack(actions),
                        prompts=np.array([instruction] * len(idx), dtype=object))
    return dict(key=key, pool=row["pool"], condition=row["task"], source=row["source"],
                length=len(idx), instruction=instruction, success=row["success"], feasible=True,
                prompt_switches=[], corrupt=False, corrupt_reason=None,
                env_end_step=int(env["end_step"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="the unzipped pi0fast-libero_10 directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_finetune", type=int, required=True)
    ap.add_argument("--n_calib", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="convert at most N episodes; preflight only")
    args = ap.parse_args()

    raw_dir, out = Path(args.raw_dir), Path(args.out_dir)
    rows = scan(raw_dir)
    assign_pools(rows, args.n_finetune, args.n_calib, args.seed)
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
        pools={r["key"]: r["pool"] for r in records}), indent=1))

    print("\npool x outcome per task (frames per episode min-max):")
    for t in sorted({r["condition"] for r in records}, key=lambda s: int(s[4:])):
        rs = [r for r in records if r["condition"] == t]
        c = {(p, o): sum(1 for r in rs if r["pool"] == p and r["success"] == o)
             for p in ("finetune", "calib", "test") for o in (True, False)}
        ln = {o: [r["length"] for r in rs if r["success"] == o] for o in (True, False)}
        rng = {o: f"{min(v)}-{max(v)}" if v else "--" for o, v in ln.items()}
        print(f"  {t:7s} {rs[0]['instruction'][:60]:60s} ft {c[('finetune', True)]:3d} "
              f"cal {c[('calib', True)]:3d} test succ {c[('test', True)]:3d} fail "
              f"{c[('test', False)]:3d}  len succ {rng[True]} fail {rng[False]}")
    print(f"\nwrote {out / 'index.json'} ({len(records)} episodes)")


if __name__ == "__main__":
    main()
