"""Bridge/Fractal perturbation-sensitivity analysis (host side, pure numpy + h5py).

load     — HDF5 dumps of `eval_perturbation_sensitivity.py` (+ the Sept-2025 legacy inpainting dumps)
scores   — readouts of an ensemble per frame (spread, mean, single member, time-corrected z)
evaluate — AUROC with paired CI per condition x readout x frame, operating points, member subsets
tables   — booktabs writer
figures  — paper-style figures with provenance stamp
"""
