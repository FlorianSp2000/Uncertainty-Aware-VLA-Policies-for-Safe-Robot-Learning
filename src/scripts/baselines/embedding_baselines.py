"""CLIP detectors on the Bridge/Fractal perturbation sets, scored on the rows of the ensemble dumps.

CLIP image novelty (Mahalanobis / cosine kNN to training-split originals) and CLIP image-instruction
alignment, per (trajectory, frame), for every condition of the reference dumps: language swaps use
the dump's swapped strings, image conditions are rebuilt with the cluster's own code
(say_no.perturbation.images, verified against the dump's logged pixel statistics). Outputs:
clip_scores_<set>.hdf5 (read by analyse_perturbation_thesis.py), embedding_baselines.{json,md}
(CLIP-only AUROC), two contact sheets, the embedding cache. Invocation:
docs/REPRODUCE.md, step 2a.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

from say_no.baselines import embedding as E
from say_no.baselines.figures import contact_sheet
from say_no.perturbation.images import rebuild_like_dump
from say_no.perturbation.load import load_models, reviewed_frames, write_detector_scores

IMG_KEYS = ("first_image", "last_image")
INP_KEYS = ("first_image_inpainted", "last_image_inpainted")
ALIGN, NOVELTY = "clip_alignment", {m: f"clip_{m}" for m in E.FeatureNovelty.METHODS}


def _pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)["trajectories"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--ref_pkl", "--swap_pkl", "--inpaint_pkl", "--swap_hdf5", "--inpaint_hdf5", "--out_dir", "--cache"):
        ap.add_argument(a, required=True, type=Path)
    ap.add_argument("--clip_model", required=True)
    ap.add_argument("--clip_revision", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n_boot", type=int, required=True)
    ap.add_argument("--n_examples", type=int, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = _pkl(args.ref_pkl)
    sets = {"val_n200": load_models([("val", args.swap_hdf5)])["val"],
            "n300_inpainted": load_models([("inp", args.inpaint_hdf5)])["inp"]}
    rows = {"val_n200": E.rows_for_dump(_pkl(args.swap_pkl), sets["val_n200"].dataset, sets["val_n200"].trajectory_ids),
            "n300_inpainted": reviewed_frames(args.inpaint_pkl, sets["n300_inpainted"].trajectory_ids,
                                              sets["n300_inpainted"].dataset)}
    for s, m in sets.items():
        if not all(t["language"] == l for t, l in zip(rows[s], m.language)):
            raise ValueError(f"{s}: instruction mismatch between pkl and dump")

    # (n, 2, H, W, 3) per set and image condition; 'orig' everywhere
    frames = {s: {"orig": np.stack([np.stack([t[k] for k in IMG_KEYS]) for t in rows[s]])} for s in sets}
    edited, image_info = rebuild_like_dump(rows["n300_inpainted"], args.inpaint_hdf5)
    frames["n300_inpainted"].update(edited)
    ref_frames = [t[k] for t in ref for k in IMG_KEYS]
    ref_keys = {E.image_key(x) for x in ref_frames}
    if any(E.image_key(x) in ref_keys for s in frames for v in frames[s].values() for x in v.reshape(-1, *v.shape[2:])):
        raise ValueError("reference frames overlap the evaluation frames")

    texts = sorted({l for m in sets.values() for l in m.language} |
                   {x for m in sets.values() for v in m.swap_language.values() for x in v})
    order = [(s, c) for s in frames for c in frames[s]]
    flat = ref_frames + [x for s, c in order for x in frames[s][c].reshape(-1, *frames[s][c].shape[2:])]
    img_emb, txt_emb = E.cached_clip_embed(flat, texts, args.cache, args.clip_model, args.clip_revision)
    tix = {t: i for i, t in enumerate(texts)}
    emb_ref = img_emb[:len(ref_frames)].reshape(len(ref), 2, -1)
    emb, o = {s: {} for s in frames}, len(ref_frames)
    for s, c in order:
        n = frames[s][c].shape[0]
        emb[s][c] = img_emb[o:o + 2 * n].reshape(n, 2, -1)
        o += 2 * n

    def align(img, strings):  # (n, 2, D) frames x (n,) instructions -> (n, 2)
        t = np.repeat(txt_emb[[tix[x] for x in strings]][:, None], 2, axis=1)
        return E.alignment_score(img.reshape(-1, img.shape[-1]), t.reshape(-1, img.shape[-1])).reshape(-1, 2)

    groups = np.array([t["dataset"] for t in ref])
    common = {"clip_model": args.clip_model, "clip_revision": args.clip_revision, "k": args.k, "n_boot": args.n_boot,
              "reference": f"{args.ref_pkl} originals, first+last frame, n={len(ref)} trajectories, fit per dataset, min over datasets",
              "rendered": dt.datetime.now().isoformat(timespec="seconds"),
              "both": "mean of the two frame scores", "positive class": "perturbed input; negative: the same trajectory's original"}
    results = {}
    for s, m in sets.items():
        scores = {ALIGN: {"orig": align(emb[s]["orig"], m.language)}}
        for c, sw in m.swap_language.items():  # frames unchanged: novelty is 0.5 by construction, not scored
            scores[ALIGN][c] = align(emb[s]["orig"], sw)
        for c in emb[s]:
            if c != "orig":
                scores[ALIGN][c] = align(emb[s][c], m.language)
        for meth, rd in NOVELTY.items():
            scores[rd] = E.frame_novelty(lambda meth=meth: E.FeatureNovelty(meth, k=args.k), emb_ref, groups, emb[s])
        meta = {**common, "rows_of": str(m.paths[0]), "n_trajectories": len(m.trajectory_ids),
                **({"images": image_info} if s == "n300_inpainted" else {})}
        write_detector_scores(args.out_dir / f"clip_scores_{s}.hdf5", meta, m, scores)
        conds = {}
        for rd, per in scores.items():
            for c, v in per.items():
                if c != "orig":
                    conds.setdefault(c, {}).update(E.auroc_rows(rd, per["orig"], v, n_boot=args.n_boot))
        results[f"CLIP, {s}"] = {"meta": meta, "conditions": conds}
    (args.out_dir / "embedding_baselines.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "embedding_baselines.md").write_text(E.markdown_table(results), encoding="utf-8")

    prov = f"src/scripts/baselines/embedding_baselines.py · {args.clip_model}@{args.clip_revision[:8]}"
    n, inp, m = args.n_examples, rows["n300_inpainted"], sets["n300_inpainted"]
    own_i, own_e = align(emb["n300_inpainted"]["orig"], m.language), align(emb["n300_inpainted"]["inpaint"], m.language)
    drop = (own_e - own_i).ravel()
    idx = np.argsort(-drop)
    ex = [{"images": [(inp[j // 2][IMG_KEYS[j % 2]], f"original, {('first', 'last')[j % 2]}"),
                      (inp[j // 2][INP_KEYS[j % 2]], "inpainted")],
           "caption": f"{'largest' if r < n else 'smallest'} drop · {inp[j // 2]['dataset']} · \"{inp[j // 2]['language']}\" · "
                      f"cos {-own_i.ravel()[j]:.3f} -> {-own_e.ravel()[j]:.3f}"}
          for r, j in enumerate(np.r_[idx[:n], idx[-n:]])]
    contact_sheet(ex, args.out_dir / "clip_alignment_drop_extremes_inpainting.png",
                  f"CLIP cosine(frame, own instruction), original vs inpainted frame: {n} largest and {n} smallest drops", prov)
    c, val, m = "lang_swap_train", rows["val_n200"], sets["val_n200"]
    sw = m.swap_language[c]
    own_v, swp_v = align(emb["val_n200"]["orig"], m.language), align(emb["val_n200"]["orig"], sw)
    idx = np.argsort(-(swp_v - own_v).ravel())
    ex = [{"images": [(val[j // 2][IMG_KEYS[j % 2]], f"{val[j // 2]['dataset']}, {('first', 'last')[j % 2]} frame")],
           "caption": f"{'largest' if r < n else 'smallest'} drop · own: \"{val[j // 2]['language']}\" (cos {-own_v.ravel()[j]:.3f}) · "
                      f"swapped: \"{sw[j // 2]}\" (cos {-swp_v.ravel()[j]:.3f})"}
          for r, j in enumerate(np.r_[idx[:n], idx[-n:]])]
    contact_sheet(ex, args.out_dir / "clip_alignment_drop_extremes_language_swap.png",
                  f"CLIP cosine(frame, instruction), own vs swapped (training-pool swap): {n} largest and {n} smallest drops",
                  prov, per_row=4)
    print(E.markdown_table(results))


if __name__ == "__main__":
    main()
