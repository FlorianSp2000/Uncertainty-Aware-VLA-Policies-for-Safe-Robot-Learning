"""Per-episode cache writer shared by convert_safe_rollouts.py and convert_libero_rollouts.py.

(The loader also supports a real-robot demonstration dataset from preliminary experiments that are not part of the thesis.)
Original description of the real-robot conversion:

Input is ``datasets/real_robot/raw_data_recomposed``, the nine delivered archives merged into
one directory tree. The recordings hold 20.7 GB of pickled RGB *and* depth for two cameras;
re-reading that at every job start would cost more than the training step, and the ensemble we
finetune from has a single-image encoder, so a second view cannot be fed without changing the
architecture and giving up the pretrained weights. This runs once and writes one ``.npz`` per
episode holding PNG-encoded external frames, the 7-D actions and the per-frame prompts, which
``octo/data/real_robot.py`` reads directly.

Actions are written **raw**. Normalisation is applied at load time from statistics computed by
``experiments/utils/real_robot_actions.py``, so changing the action treatment never means
re-running this job.

"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.utils import real_robot_raw as raw  # noqa: E402

# Below this an episode cannot even carry one bootstrapped transition. The delivered set holds
# one such recording (a single frame, zero action, timestamp 0.0) -- broken at source.
MIN_EPISODE_FRAMES = 2


def _png(frame: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def _slug_prompt(text: str) -> str:
    """``Pick up the green cup and place it on the blue circle.`` -> ``gcup_blue``.

    Kept verbatim from the zip-reading version: the slug is part of every cache key, and the
    eval dumps and results already written reference those keys.
    """
    m = re.search(r"the (\w+) cup and place it on the (\w+) circle", text)
    if not m:
        raise ValueError(f"unrecognised prompt: {text!r}")
    return f"{m.group(1)[0]}cup_{m.group(2)}"


def _episode_key(row: dict, success, instruction: str) -> str:
    """Short, stable name: what kind of episode, which condition, which outcome, which index.

    Not derived verbatim from the source path -- one rollout directory is named after the whole
    prompt sentence, which pushes the file past Windows' 260-character path limit and reads
    badly in a directory listing. Full provenance stays in index.json's ``source``.
    """
    parts = Path(row["rel"]).parts
    stem = Path(row["rel"]).stem
    if row["pool"] in ("teleop", "scene"):
        return f"{row['pool']}__{row['condition']}__{stem.replace('recorded_traj_', 'traj')}"
    outcome = "ok" if success else "fail"
    if len(parts) == 4:  # normal_rollout nests one directory per prompt
        return f"rollout__{row['condition']}__{_slug_prompt(parts[1])}__{outcome}__{stem}"
    return f"rollout__{row['condition']}__{outcome}__{stem}"


def _convert_one(args) -> dict:
    row, out_dir, instruction = args
    episode = raw.load(row["path"], with_wrist=False)

    if episode["kind"] == "rollout":
        prompts = episode["prompts"]
        instruction = prompts[0]
    else:
        prompts = [instruction] * episode["n"]

    key = _episode_key(row, row["success"], instruction)
    record = dict(key=key, pool=row["pool"], condition=row["condition"], source=row["rel"],
                  length=episode["n"], instruction=instruction, success=row["success"],
                  feasible=None if row["pool"] == "scene" else row["feasible"],
                  prompt_switches=[i for i in range(1, len(prompts))
                                   if prompts[i] != prompts[i - 1]],
                  corrupt=False, corrupt_reason=None)

    if episode["n"] < MIN_EPISODE_FRAMES:
        # Recorded, not silently dropped: the count the results report quotes has to match the
        # episodes that actually reached training.
        record.update(corrupt=True,
                      corrupt_reason=f"{episode['n']} frame(s), fewer than {MIN_EPISODE_FRAMES}")
        return record

    actions = episode["actions"]
    if actions.shape != (episode["n"], raw.ACTION_DIM):
        raise ValueError(f"{row['rel']}: actions {actions.shape}, "
                         f"expected {(episode['n'], raw.ACTION_DIM)}")

    dst = Path(out_dir) / row["pool"]
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst / f"{key}.npz",
        images=np.array([_png(f) for f in episode["external"]], dtype=object),
        actions=actions.astype(np.float32),
        prompts=np.array(prompts, dtype=object),
    )
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="the recomposed directory tree")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="convert at most N episodes per pool; preflight only")
    args = ap.parse_args()

    root = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = raw.index(root)
    instructions = {r["condition"]: raw.instruction_of(root, r["condition"])
                    for r in rows if r["pool"] == "teleop"}
    if len(instructions) != 6:
        raise ValueError(f"expected 6 task descriptions, found {sorted(instructions)}")

    jobs = []
    for row in rows:
        if row["pool"] == "teleop":
            instruction = instructions[row["condition"]]
        elif row["pool"] == "scene":
            # The leave-one-out scenes carry no instruction of their own; the folder name is
            # all there is, and it is not a sentence. They are not training data.
            instruction = row["condition"]
        else:
            instruction = ""  # read per frame from the record itself
        jobs.append((row, str(out), instruction))

    if args.limit:
        capped, seen = [], {}
        for j in jobs:
            pool = j[0]["pool"]
            seen[pool] = seen.get(pool, 0) + 1
            if seen[pool] <= args.limit:
                capped.append(j)
        jobs = capped
    print(f"converting {len(jobs)} episodes from {root}", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_convert_one, j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    records.sort(key=lambda r: r["source"])
    (out / "index.json").write_text(json.dumps(records, indent=1))

    # A cache that still holds episodes its index does not list is a trap: a key that changed,
    # a recording withdrawn from the source, or -- the case that found this -- an episode now
    # marked corrupt whose file an earlier conversion had already written. Skipped under
    # --limit, where the index is a subset on purpose.
    if not args.limit:
        expected = {Path(r["pool"]) / f"{r['key']}.npz" for r in records if not r["corrupt"]}
        stale = [p for p in out.glob("*/*.npz") if p.relative_to(out) not in expected]
        for p in stale:
            print(f"  removing stale {p.relative_to(out)}")
            p.unlink()
        print(f"pruned {len(stale)} stale episode files")

    counts = {}
    for r in records:
        k = (r["pool"], r["condition"])
        counts[k] = counts.get(k, 0) + 1
    print("\nwrote", out / "index.json")
    for k in sorted(counts):
        print(f"  {k[0]:8s} {k[1]:28s} {counts[k]:4d}")

    corrupt = [r for r in records if r["corrupt"]]
    print(f"\n{len(records) - len(corrupt)} usable, {len(corrupt)} corrupt")
    for r in corrupt:
        print(f"  SKIPPED {r['source']}: {r['corrupt_reason']}")


if __name__ == "__main__":
    main()
