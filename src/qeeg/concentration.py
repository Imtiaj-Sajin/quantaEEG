"""Does quantum kernel concentration worsen with qubit count, on real EEG?

Thanasilp, Wang, Cerezo & Holmes (Nat. Commun. 15, 2024) prove that quantum
kernel values concentrate exponentially in the number of qubits toward a fixed
value, collapsing the model to a trivial one. Their demonstrations are largely
synthetic. This module runs the same test on biological data.

Because a d-channel EEG covariance, trace-normalised, is a density matrix on a
d-dimensional Hilbert space, sweeping the channel count over powers of two
sweeps the qubit count directly: 4, 8, 16, 32, 64 channels = 2, 3, 4, 5, 6
qubits. No re-encoding, no change of method -- only the register size moves.

The reported statistic is the variance of the off-diagonal Gram entries. Under
exponential concentration it should fall geometrically with qubit count; a
model whose kernel variance is below shot noise cannot learn anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pyriemann.estimation import Covariances

from .quantum import (
    CircuitKernel,
    bures_distance_sq,
    fidelity_kernel,
    hs_distance_sq,
    hs_overlap_kernel,
    median_bandwidth,
    to_density_matrices,
)

from .data import CH_ORDER

QUBIT_SIZES = (4, 8, 16, 32, 64)


def _offdiag(K: np.ndarray) -> np.ndarray:
    return K[~np.eye(len(K), dtype=bool)]


def _normalise(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    return K / np.outer(d, d)


def kernel_stats(K: np.ndarray, name: str, n_qubits: int, dim: int) -> dict:
    off = _offdiag(K)
    return {
        "kernel": name,
        "n_channels": dim,
        "n_qubits": n_qubits,
        "mean": float(off.mean()),
        "std": float(off.std()),
        "variance": float(off.var()),
        "min": float(off.min()),
        "max": float(off.max()),
        # Shots needed to resolve typical kernel differences on hardware:
        # estimating tr(rho sigma) to precision eps costs O(1/eps^2) shots.
        "shots_for_1sigma": (
            float(1.0 / off.var()) if off.var() > 1e-18 else float("inf")
        ),
    }


def run_subject(ep_all, sizes=QUBIT_SIZES) -> list[dict]:
    """Sweep register size for one subject loaded with all 64 channels."""
    rows = []
    name_to_idx = {c: i for i, c in enumerate(ep_all.ch_names)}
    order = [c for c in CH_ORDER if c in name_to_idx]

    for dim in sizes:
        if dim > len(order):
            continue
        idx = [name_to_idx[c] for c in order[:dim]]
        X = ep_all.X[:, idx, :]
        n_qubits = int(np.log2(dim))

        C = Covariances(estimator="oas").fit_transform(X)
        rho = to_density_matrices(C)

        # Raw overlap kernels: the quantities a SWAP test would estimate.
        rows.append(kernel_stats(
            _normalise(hs_overlap_kernel(rho)), "HS-overlap", n_qubits, dim))
        rows.append(kernel_stats(
            _normalise(fidelity_kernel(rho)), "Fidelity", n_qubits, dim))

        # Bandwidth-corrected versions: does the remedy survive scaling?
        Dh = hs_distance_sq(rho)
        rows.append(kernel_stats(
            np.exp(-median_bandwidth(Dh) * Dh), "HS-RBF", n_qubits, dim))
        Db = bures_distance_sq(rho)
        rows.append(kernel_stats(
            np.exp(-median_bandwidth(Db) * Db), "Bures-RBF", n_qubits, dim))

        for r in rows[-4:]:
            r["subject"] = ep_all.subject
    return rows


def run_circuit_sweep(ep_all, qubit_range=(2, 3, 4, 5, 6)) -> list[dict]:
    """Same test for IQP circuit kernels, with and without entanglement."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from pyriemann.tangentspace import TangentSpace

    name_to_idx = {c: i for i, c in enumerate(ep_all.ch_names)}
    order = [c for c in CH_ORDER if c in name_to_idx][:16]
    idx = [name_to_idx[c] for c in order]
    C = Covariances(estimator="oas").fit_transform(ep_all.X[:, idx, :])

    rows = []
    for nq in qubit_range:
        feats = Pipeline([
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=nq, random_state=0)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]).fit_transform(C)
        for entangle in (True, False):
            K = CircuitKernel(nq, n_layers=2, entangle=entangle)(feats)
            label = f"IQP-{'entangled' if entangle else 'product'}"
            r = kernel_stats(K, label, nq, nq)
            r["subject"] = ep_all.subject
            rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Kernel concentration vs qubit count on real EEG")
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args(argv)

    from .data import load_many

    # Load with the full 64-channel montage so every register size is a subset.
    chans = [c for c in CH_ORDER]
    subjects = list(range(args.start, args.start + args.subjects))
    print(f"Loading {len(subjects)} subjects with {len(chans)} channels ...")
    eps = load_many(subjects, channels=chans)
    print(f"  usable: {len(eps)} subjects")
    if not eps:
        return 1

    rows, crows = [], []
    for i, ep in enumerate(eps, 1):
        rows.extend(run_subject(ep))
        crows.extend(run_circuit_sweep(ep))
        print(f"  [{i}/{len(eps)}] S{ep.subject:03d} done")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows + crows)
    df.to_csv(out / "concentration_raw.csv", index=False)

    agg = (
        df.groupby(["kernel", "n_qubits", "n_channels"])
        .agg(variance=("variance", "mean"),
             mean=("mean", "mean"),
             std=("std", "mean"),
             n_subjects=("subject", "nunique"))
        .reset_index()
        .sort_values(["kernel", "n_qubits"])
    )
    agg.to_csv(out / "concentration_summary.csv", index=False)

    # Decay factor per added qubit: the empirical test of the theory.
    decay = []
    for k, g in agg.groupby("kernel"):
        g = g.sort_values("n_qubits")
        v = g["variance"].to_numpy()
        q = g["n_qubits"].to_numpy()
        if len(v) >= 2 and np.all(v > 0):
            slope = np.polyfit(q, np.log(v), 1)[0]
            decay.append({
                "kernel": k,
                "log_variance_slope_per_qubit": float(slope),
                "variance_factor_per_qubit": float(np.exp(slope)),
                "variance_first": float(v[0]),
                "variance_last": float(v[-1]),
                "qubits": f"{q[0]}->{q[-1]}",
            })
    dec = pd.DataFrame(decay).sort_values("variance_factor_per_qubit")
    dec.to_csv(out / "concentration_decay.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== KERNEL VARIANCE vs QUBITS (mean over subjects) ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n=== EXPONENTIAL DECAY FIT ===")
    print("(variance_factor_per_qubit < 1 => concentration worsens with scale)")
    print(dec.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    (out / "concentration_meta.json").write_text(json.dumps({
        "subjects_used": [e.subject for e in eps],
        "channel_sizes": list(QUBIT_SIZES),
        "n_trials_per_subject": {str(e.subject): int(len(e)) for e in eps},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
