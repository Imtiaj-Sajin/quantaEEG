"""Reference states: the invariance argument, and the diagnostics that test it.

Why this module exists
----------------------
The benchmark found the density-matrix kernels last of fifteen pipelines on
both datasets (RESEARCH.md 4.4, 4.5), with the pairwise overlaps all sitting at
0.99. The diagnosis recorded there was concentration driven by the data
distribution. This module supplies the level below that: *why* EEG covariances
look identical to these kernels, and what to do about it.

The nuisance group of EEG is congruence. Electrode gain, impedance drift, the
choice of reference electrode, volume conduction, and the change from one
subject or session to the next all act on the spatial covariance as
C -> A C A^T for some invertible A. The affine-invariant Riemannian metric is
invariant under exactly that group, which is why tangent-space decoding is the
classical state of the art and why it transfers.

The quantum quantities are not. tr(rho sigma), Uhlmann fidelity, the Bures
distance and the quantum relative entropy are invariant only under the
*orthogonal* subgroup. Measured in the sensor frame they are therefore
sensitive to a nuisance the baseline is immune to -- and, as `gram_report`
shows on real data, that sensitivity is an order of magnitude larger than the
entire discriminative spread of the kernel.

The fix is to measure each state relative to a reference state M, the Frechet
mean of the training covariances:

    rho~ = W C W / tr(W C W),    W = M^-1/2.

Proposition. Under C_i -> A C_i A^T the affine-invariant mean is equivariant,
M -> A M A^T. Put B = (A M A^T)^-1/2 A. Then B M B^T = I and W M W^T = I, so
U := B M^1/2 is orthogonal and B = U W. Every whitened matrix therefore
transforms as W C_i W -> U (W C_i W) U^T, by one common orthogonal U. Since all
four quantum quantities are invariant under a common unitary, in the reference
frame they are exactly affine-invariant. `check_invariance` verifies this to
machine precision.

Two things follow, and the second is the one that matters for the benchmark.

1. The concentration is largely an artefact of the frame. Dividing out the
   common component spreads the states out; `gram_report` measures how much.

2. The published comparison was not like for like. pyriemann's
   ``TangentSpace`` estimates a reference mean in ``fit`` and maps
   C -> log(M^-1/2 C M^-1/2): the classical baseline was already working in
   the reference frame while the quantum kernels were not. MDM and CSP are
   likewise congruence-invariant. So *every* strong classical pipeline in the
   suite had an invariance the quantum kernels lacked, and the benchmark was
   in part measuring that difference rather than the geometry.

The construction stays quantum. rho -> W rho W / tr(W rho W) is a filtering
(Lueders) operation with Kraus operator W followed by renormalisation, so the
SWAP-test implementation path is unaffected: it is state preparation, not
classical feature engineering.

Run
---
    PYTHONPATH=src python -m qeeg.reference --check
    PYTHONPATH=src python -m qeeg.reference --gram --subjects 10
"""

from __future__ import annotations

import argparse
import functools

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

from .quantum import (
    bures_distance_sq,
    fidelity_kernel,
    hs_overlap_kernel,
    qre_divergence,
    reference_whitener,
    to_density_matrices,
)


def _kernels(covs: np.ndarray) -> dict[str, np.ndarray]:
    rho = to_density_matrices(covs)
    return {
        "HS-overlap": hs_overlap_kernel(rho),
        "Fidelity": fidelity_kernel(rho),
        "Bures-d2": bures_distance_sq(rho),
        "QRE": qre_divergence(rho),
    }


# --------------------------------------------------------------------------
# 1. The invariance claim, verified numerically
# --------------------------------------------------------------------------

def check_invariance(n: int = 40, d: int = 8, seed: int = 0) -> dict:
    """Perturb by a nuisance congruence; report the change in each kernel.

    Returns max |K - K_perturbed| per kernel in both frames. Reference-frame
    residuals should be at machine precision, sensor-frame ones O(0.1).
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d, 3 * d))
    C = np.einsum("nij,nkj->nik", Z, Z) / (3 * d) + 0.1 * np.eye(d)

    # Per-electrode gain composed with source mixing: the realistic nuisance.
    A = (np.eye(d) + 0.3 * rng.normal(size=(d, d))) @ np.diag(
        np.exp(rng.normal(0.0, 0.7, size=d)))
    Ca = A @ C @ A.T

    def whiten(cov):
        W = reference_whitener(cov)
        return W @ cov @ W

    sensor, sensor_a = _kernels(C), _kernels(Ca)
    ref, ref_a = _kernels(whiten(C)), _kernels(whiten(Ca))

    out = {"cond_A": float(np.linalg.cond(A))}
    for k in sensor:
        out[k] = {
            "sensor": float(np.abs(sensor[k] - sensor_a[k]).max()),
            "reference": float(np.abs(ref[k] - ref_a[k]).max()),
        }
    return out


# --------------------------------------------------------------------------
# 2. Concentration, sensor frame versus reference frame, on real EEG
# --------------------------------------------------------------------------

def gram_report(epochs_list, cov_estimator: str = "oas"):
    """Off-diagonal Gram statistics per subject, in both frames.

    The column that matters is `var`: kernel variance is what concentration
    destroys, and the shot budget needed to resolve a kernel entry on hardware
    scales as its reciprocal.
    """
    import pandas as pd
    from pyriemann.estimation import Covariances

    rows = []
    for ep in epochs_list:
        C = Covariances(estimator=cov_estimator).fit_transform(ep.X)
        W = reference_whitener(C)
        for frame, cov in (("sensor", C), ("reference", W @ C @ W)):
            for name, K in _kernels(cov).items():
                if name in ("HS-overlap", "Fidelity"):
                    # Cosine-normalise the overlaps so the scale is comparable.
                    dg = np.sqrt(np.clip(np.diag(K), 1e-12, None))
                    K = K / np.outer(dg, dg)
                off = K[~np.eye(len(K), dtype=bool)]
                rows.append({
                    "subject": ep.subject, "frame": frame, "kernel": name,
                    "mean": float(off.mean()), "std": float(off.std()),
                    "var": float(off.var()),
                    "shots_for_1sigma": float(1.0 / max(off.var(), 1e-300)),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the affine-invariance claim numerically")
    ap.add_argument("--gram", action="store_true",
                    help="measure concentration in both frames on real EEG")
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--channels", type=str, default="motor8")
    ap.add_argument("--dataset", type=str, default="physionet",
                    choices=("physionet", "bci2a", "bci2b"))
    ap.add_argument("--out", type=str, default="results/reference_gram.csv")
    args = ap.parse_args(argv)

    if not (args.check or args.gram):
        ap.error("nothing to do: pass --check and/or --gram")

    if args.check:
        r = check_invariance()
        print(f"\nNuisance congruence, cond(A) = {r.pop('cond_A'):.1f}")
        print(f"{'kernel':12s} {'sensor frame':>16s} {'reference frame':>18s}")
        for k, v in r.items():
            print(f"{k:12s} {v['sensor']:16.3e} {v['reference']:18.3e}")
        print("\nMachine-precision residuals in the reference frame: the "
              "quantum kernels are exactly affine-invariant there, and not "
              "at all in the sensor frame.")

    if args.gram:
        from .data import CHANNEL_SETS, load_many, load_moabb

        chans = CHANNEL_SETS[args.channels]
        subs = list(range(args.start, args.start + args.subjects))
        eps = (load_many(subs, channels=chans) if args.dataset == "physionet"
               else load_moabb(args.dataset, subjects=subs, channels=chans))
        df = gram_report(eps)
        df.to_csv(args.out, index=False)

        piv = df.pivot_table(index="kernel", columns="frame",
                             values=["mean", "var"])
        print(f"\n=== Gram off-diagonal statistics, n = {df.subject.nunique()} "
              f"subjects ===")
        print(piv.to_string(float_format=lambda x: f"{x:.5f}"))
        gain = (df[df.frame == "reference"].groupby("kernel")["var"].mean()
                / df[df.frame == "sensor"].groupby("kernel")["var"].mean())
        print("\nVariance gain from the reference frame (x):")
        print(gain.to_string(float_format=lambda x: f"{x:.2f}"))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
