"""Opening figure: from scalp EEG to a quantum state.

Every panel is computed from real recordings (PhysioNet EEGMMIDB), not drawn or
simulated. The point of the figure is to make the paper's central identity
concrete for a reader who is not already fluent in both fields: a band-passed
EEG trial has a spatial covariance, and that covariance, normalised to unit
trace, is literally a density matrix.

Panels:
  (a) band-passed sensorimotor traces for a left-hand and a right-hand trial
  (b) scalp topography of mu/beta power, right minus left, showing the
      contralateral desynchronisation the decoders actually use
  (c) the trial covariance matrix over the eight decoding channels
  (d) the same matrix after trace normalisation, with its eigenvalue spectrum,
      which is a probability distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import figures as F  # noqa: E402
from .data import MOTOR_8, CH_ORDER, load_subject  # noqa: E402
from .quantum import to_density_matrices  # noqa: E402


def _band_power(X: np.ndarray) -> np.ndarray:
    """Mean band power per channel over trials, for band-passed data."""
    return np.mean(np.var(X, axis=-1), axis=0)


def fig_eeg(out: Path, subject: int = 1) -> bool:
    import mne

    mne.set_log_level("ERROR")

    # 64 channels for the topography, 8 for everything the decoders see.
    ep64 = load_subject(subject, channels=CH_ORDER)
    ep8 = load_subject(subject, channels=MOTOR_8)

    left = ep64.X[ep64.y == 0]
    right = ep64.X[ep64.y == 1]
    contrast = np.log(_band_power(right)) - np.log(_band_power(left))

    info = mne.create_info(ep64.ch_names, ep64.sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))

    F._style()
    # Layout follows the reading order of the pipeline: signal, scalp map,
    # covariance, density matrix, spectrum. Panels are laid out so that
    # (a) and (b) form the "what the data looks like" row and (c) to (e) the
    # "what we turn it into" row.
    fig = plt.figure(figsize=(11.8, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.0],
                          height_ratios=[1.05, 1.0], wspace=0.38, hspace=0.55)

    # ---- (a) traces -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:2])
    show = ["C3", "Cz", "C4"]
    idx = [ep64.ch_names.index(c) for c in show]
    t = np.arange(ep64.X.shape[-1]) / ep64.sfreq
    offset = 0.0
    step = 4.5 * np.std(ep64.X[:, idx, :])
    for k, (ci, name) in enumerate(zip(idx, show)):
        for trial, colour, lab in ((left[0], F.BLUE, "left hand"),
                                   (right[0], F.ORANGE, "right hand")):
            ax.plot(t, trial[ci] * 1e6 / 1e6 + offset, color=colour,
                    linewidth=1.0, alpha=0.9,
                    label=lab if k == 0 else None)
        ax.text(-0.03, offset, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9, color=F.INK,
                fontweight="bold")
        offset -= step
    ax.set_xlabel("Time from cue (s)")
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])
    F._despine(ax, keep=("bottom",))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title("(a)  Band-passed trials, 8 to 30 Hz", loc="left",
                 fontsize=9.5, fontweight="bold", color=F.INK, pad=8)

    # ---- (b) topography -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    lim = float(np.max(np.abs(contrast)))
    im, _ = mne.viz.plot_topomap(
        contrast, info, axes=ax, show=False, cmap="RdBu_r",
        vlim=(-lim, lim), contours=4, sensors=True,
        # Fill the head outline, but clamp the border to the mean so the
        # extrapolation cannot invent saturated colour outside the array.
        extrapolate="head", border="mean",
    )
    ax.set_title("(b)  Mu/beta power,\nright minus left", loc="left",
                 fontsize=9.5, fontweight="bold", color=F.INK, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("log power ratio", fontsize=7.5)

    # ---- (c) covariance -------------------------------------------------
    from pyriemann.estimation import Covariances

    C = Covariances(estimator="oas").fit_transform(ep8.X)
    ax = fig.add_subplot(gs[1, 0])
    m = ax.imshow(C[0] / C[0].max(), cmap="viridis")
    ax.set_xticks(range(len(MOTOR_8)))
    ax.set_yticks(range(len(MOTOR_8)))
    ax.set_xticklabels(MOTOR_8, rotation=90, fontsize=6.5)
    ax.set_yticklabels(MOTOR_8, fontsize=6.5)
    ax.set_title("(c)  Spatial covariance $C$ (scaled)", loc="left",
                 fontsize=9.5, fontweight="bold", color=F.INK, pad=8)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.06)
    cb.ax.tick_params(labelsize=7.5)

    # ---- (d) density matrix + spectrum ----------------------------------
    rho = to_density_matrices(C)
    ax = fig.add_subplot(gs[1, 1])
    m = ax.imshow(rho[0], cmap="viridis")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"(d)  $\rho = C/\mathrm{tr}\,C$", loc="left",
                 fontsize=9.5, fontweight="bold", color=F.INK, pad=8)
    ax.text(0.5, -0.10, f"trace = {np.trace(rho[0]):.3f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=F.INK_2)
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.06)
    cb.ax.tick_params(labelsize=7.5)

    ax = fig.add_subplot(gs[1, 2])
    w = np.linalg.eigvalsh(rho[0])[::-1]
    ax.bar(np.arange(1, len(w) + 1), w, color=F.AQUA, width=0.68,
           edgecolor=F._surface(), linewidth=1.2)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Population")
    ax.set_xticks(range(1, len(w) + 1))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    F._despine(ax)
    ax.set_title(r"(e)  Spectrum of $\rho$, sums to 1", loc="left",
                 fontsize=9.5, fontweight="bold", color=F.INK, pad=8)

    F._caption(fig, (
        f"All panels computed from PhysioNet EEGMMIDB subject "
        f"S{subject:03d}; nothing is simulated.\n"
        "The contralateral pattern in (b) is the physiological signal every "
        "decoder in this study exploits."
    ))
    F._save(fig, out, "fig0_eeg_to_state")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EEG to quantum state figure")
    ap.add_argument("--results", default="results")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args(argv)

    F.PAPER = args.paper
    out = Path(args.results) / ("figures_paper" if args.paper else "figures")
    print(f"Writing EEG figure to {out.resolve()}")
    fig_eeg(out, subject=args.subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
