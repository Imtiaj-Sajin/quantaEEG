"""PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) loader.

Task: left-fist vs right-fist motor imagery (runs 4, 8, 12), the standard
binary MI protocol used by MOABB's ``PhysionetMI`` / ``LeftRightImagery``.

Design choices are deliberately conservative so the classical baselines are
strong; a quantum method that only wins against a crippled baseline proves
nothing (Bowles et al., arXiv:2403.07059).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Subjects with documented annotation / sampling-rate defects in EEGMMIDB.
BAD_SUBJECTS = (88, 89, 92, 100, 104)

# Runs containing imagined left-fist (T1) vs right-fist (T2) movement.
MI_RUNS = (4, 8, 12)

# Sensorimotor montage. 3 qubits' worth (8 channels) of amplitude encoding,
# centred on the hand area of motor cortex where mu/beta ERD lives.
MOTOR_8 = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]
MOTOR_16 = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CPz", "CP4",
]

# Nested motor-centric ordering over the full 10-10 montage. Each power-of-two
# prefix is a valid register size: 4/8/16/32/64 channels = 2/3/4/5/6 qubits.
# Used for the channel-scaling experiments, where the point is that the
# density-matrix kernels never vectorise the covariance and so do not suffer
# the O(n_ch^2) feature blow-up that breaks tangent-space methods.
CH_ORDER = [
    "C3", "Cz", "C4", "CPz",
    "FC3", "FCz", "FC4", "CP3",
    "CP4", "C1", "C2", "C5", "C6", "FC1", "FC2", "CP1",
    "CP2", "FC5", "FC6", "CP5", "CP6", "F3", "Fz", "F4",
    "P3", "Pz", "P4", "T7", "T8", "P7", "P8", "Fpz",
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
    "F5", "F1", "F2", "F6", "F8", "FT7", "FT8", "T9",
    "T10", "TP7", "TP8", "P5", "P1", "P2", "P6", "PO7",
    "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]

MOTOR_32 = CH_ORDER[:32]
ALL_64 = CH_ORDER[:64]

CHANNEL_SETS = {
    "motor8": MOTOR_8,
    "motor16": MOTOR_16,
    "motor32": MOTOR_32,
    "all64": ALL_64,
}


@dataclass
class Epochs:
    """Epoched trials for one subject."""

    X: np.ndarray  # (n_trials, n_channels, n_times)
    y: np.ndarray  # (n_trials,) int labels, 0 = left hand, 1 = right hand
    subject: int
    ch_names: list[str]
    sfreq: float
    # Recording session per trial, when the dataset has more than one. Enables
    # cross-session evaluation; None for single-session datasets.
    session: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return (
            f"Epochs(subject={self.subject}, n_trials={len(self.y)}, "
            f"shape={self.X.shape}, balance={np.bincount(self.y).tolist()})"
        )


def load_subject(
    subject: int,
    channels: list[str] | None = None,
    fmin: float = 8.0,
    fmax: float = 30.0,
    tmin: float = 0.5,
    tmax: float = 3.5,
    resample: float | None = 128.0,
) -> Epochs:
    """Load and preprocess one subject's left/right MI trials.

    Band-pass 8-30 Hz spans mu and beta, the physiologically motivated bands
    for motor imagery. The 0.5-3.5 s window skips the cue-evoked transient.
    """
    import mne
    from mne.datasets import eegbci

    if channels is None:
        channels = MOTOR_8

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.set_log_level("ERROR")

        paths = eegbci.load_data(subjects=subject, runs=list(MI_RUNS), update_path=True)
        raws = []
        for p in paths:
            r = mne.io.read_raw_edf(p, preload=True)
            eegbci.standardize(r)  # strip trailing dots -> standard 10-05 names
            raws.append(r)
        raw = mne.concatenate_raws(raws)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))

        missing = [c for c in channels if c not in raw.ch_names]
        if missing:
            raise ValueError(f"subject {subject}: missing channels {missing}")
        raw.pick(channels)

        raw.filter(fmin, fmax, method="iir", verbose=False)

        events, event_id = mne.events_from_annotations(raw)
        # T1 = imagined left fist, T2 = imagined right fist. T0 = rest (dropped).
        wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
        if len(wanted) != 2:
            raise ValueError(f"subject {subject}: unexpected event ids {event_id}")

        ep = mne.Epochs(
            raw, events, event_id=wanted,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False,
        )
        if resample is not None:
            ep.resample(resample)

        X = ep.get_data(copy=True).astype(np.float64)
        codes = ep.events[:, -1]
        y = (codes == wanted["T2"]).astype(int)  # 0 = left, 1 = right
        sfreq = float(ep.info["sfreq"])

    return Epochs(X=X, y=y, subject=subject, ch_names=list(channels), sfreq=sfreq)


def load_many(
    subjects: list[int],
    min_trials: int = 30,
    **kwargs,
) -> list[Epochs]:
    """Load several subjects, skipping known-bad and malformed recordings."""
    out = []
    for s in subjects:
        if s in BAD_SUBJECTS:
            continue
        try:
            ep = load_subject(s, **kwargs)
        except Exception as exc:  # noqa: BLE001 - corrupt EDFs are expected
            print(f"  [skip] subject {s}: {type(exc).__name__}: {exc}")
            continue
        if len(ep) < min_trials or len(np.unique(ep.y)) < 2:
            print(f"  [skip] subject {s}: only {len(ep)} usable trials")
            continue
        out.append(ep)
    return out


# --------------------------------------------------------------------------
# MOABB-backed datasets
# --------------------------------------------------------------------------
#
# PhysioNet gives breadth (many subjects, few trials each). The BCI Competition
# datasets give the opposite: few subjects, many trials. Running the identical
# pipeline suite on both tests whether the ranking of methods is a property of
# the methods or of the trial count, which is a limitation any single-dataset
# benchmark has to concede.

MOABB_DATASETS = {
    # BCI Competition IV-2a: 9 subjects, 22 EEG channels, 288 MI trials each
    # over 2 sessions. The de facto standard benchmark in the quantum-EEG
    # literature, which is why it is included here.
    "bci2a": "BNCI2014_001",
    # BCI Competition IV-2b: 9 subjects, 3 bipolar channels, 5 sessions.
    "bci2b": "BNCI2014_004",
}

# The 8 sensorimotor channels used for PhysioNet all exist in the 2a montage,
# so the two datasets can be compared at an identical register size.
BCI2A_MOTOR_8 = MOTOR_8
BCI2A_MOTOR_16 = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4", "Fz",
]


def load_moabb(
    dataset: str = "bci2a",
    subjects: list[int] | None = None,
    channels: list[str] | None = None,
    fmin: float = 8.0,
    fmax: float = 30.0,
    resample: float = 128.0,
    min_trials: int = 30,
) -> list[Epochs]:
    """Load a MOABB left-hand versus right-hand motor-imagery dataset.

    Band-pass and resampling match the PhysioNet pipeline. The epoch window is
    left at the dataset's own standard interval rather than forced to match,
    since each competition dataset defines its own cue timing; the comparison
    of interest is whether the *ranking* of pipelines changes, not whether the
    absolute accuracies coincide.
    """
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("moabb").setLevel(logging.ERROR)
    import mne

    mne.set_log_level("ERROR")
    import moabb.datasets as mds
    from moabb.paradigms import LeftRightImagery

    if dataset not in MOABB_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"choose from {sorted(MOABB_DATASETS)}")
    ds = getattr(mds, MOABB_DATASETS[dataset])()
    if subjects is None:
        subjects = list(ds.subject_list)

    paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, resample=resample)

    out: list[Epochs] = []
    for s in subjects:
        try:
            # return_epochs=True gives the channel names alongside the data,
            # so the recording is decoded once rather than twice.
            ep_mne, y, meta = paradigm.get_data(
                dataset=ds, subjects=[int(s)], return_epochs=True)
        except Exception as exc:  # noqa: BLE001 - a missing recording is not fatal
            print(f"  [skip] subject {s}: {type(exc).__name__}: {exc}")
            continue

        ch_names = list(ep_mne.ch_names)
        if channels:
            missing = [c for c in channels if c not in ch_names]
            if missing:
                print(f"  [skip] subject {s}: missing channels {missing}")
                continue
            ep_mne = ep_mne.copy().pick(list(channels))
            names = list(channels)
        else:
            names = ch_names
        X = ep_mne.get_data(copy=False)

        y_int = (np.asarray(y) == "right_hand").astype(int)
        if len(y_int) < min_trials or len(np.unique(y_int)) < 2:
            print(f"  [skip] subject {s}: only {len(y_int)} usable trials")
            continue

        out.append(Epochs(
            X=np.asarray(X, dtype=np.float64),
            y=y_int,
            subject=int(s),
            ch_names=names,
            sfreq=float(resample),
            session=np.asarray(meta["session"].values),
        ))
    return out
