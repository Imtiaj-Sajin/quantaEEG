"""Filter-bank pipelines: the FBCSP-class bar, and a principled way to add qubits.

Two problems solved by one construction.

1. **Baseline credibility.** The suite's classical baselines are single-band
   (8--30 Hz). While the result was "quantum loses", a referee could not object:
   a stronger baseline only strengthens a negative claim. Once the
   reference-state kernels reach parity (RESEARCH.md 4.6) that protection is
   gone, and the first thing a Journal of Neural Engineering referee will ask
   is whether parity survives against filter-bank CSP, which is the method the
   MI literature actually treats as the bar.

2. **Register size.** RESEARCH.md 4.1b(b) found that density-matrix kernels
   concentrate from the data rather than from qubit count, and are *relieved*
   by adding dimensions -- the opposite of circuit kernels. The corollary was
   that the density-matrix route should be run wider. A filter bank is the
   principled way to do that without adding electrodes: filtering into
   ``n_bands`` sub-bands and stacking gives an ``n_bands * n_ch`` dimensional
   covariance, which is still one SPD matrix and therefore still one density
   matrix, now on ``log2(n_bands * n_ch)`` qubits.

The default bank is four sub-bands **inside** the 8--30 Hz passband the loader
already applies, so no reloading, no change of protocol, and results stay
directly comparable to the single-band runs. With the 8-channel sensorimotor
montage that is 8 x 4 = 32 dimensions: exactly 5 qubits, up from 3.

Leakage: filtering is stateless and identical for train and test, so the
transformers here add no fitted state beyond what the estimators after them
already refit per fold.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace

from .pipelines import DensityKernelSVC, SPDKernelSVC

# Four sub-bands spanning the 8-30 Hz passband the loader already applies.
# mu (8-12), low beta (12-16), mid beta (16-22), high beta (22-30). Four is
# not arbitrary: 4 bands x 8 channels = 32 = 2**5, a whole number of qubits.
DEFAULT_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 22.0), (22.0, 30.0))

_FB_CACHE: dict = {}


def _filter_bands(X, bands, sfreq, method):
    import mne

    return np.concatenate(
        [mne.filter.filter_data(X, sfreq, lo, hi, method=method, verbose=False)
         for lo, hi in bands], axis=1)


class FilterBank(BaseEstimator, TransformerMixin):
    """Band-pass into sub-bands and stack along the channel axis.

    (n_trials, n_ch, n_times) -> (n_trials, n_bands * n_ch, n_times).

    Stacking rather than keeping the bands separate is deliberate on the
    quantum side: the covariance of the stacked signal is a single SPD matrix
    of size (n_bands*n_ch), hence a single density matrix, with the cross-band
    blocks carrying inter-band coupling rather than being discarded.
    """

    def __init__(self, bands=DEFAULT_BANDS, sfreq: float = 128.0,
                 method: str = "iir"):
        self.bands = bands
        self.sfreq = sfreq
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Filtering is stateless and depends only on the input bytes, but
        # nested CV re-runs it for every grid point on the same training fold.
        # A small content-keyed cache turns that into one filter pass per
        # distinct fold, which is most of the suite's runtime.
        key = (hash(X.tobytes()), X.shape, self.bands, self.sfreq, self.method)
        hit = _FB_CACHE.get(key)
        if hit is not None:
            return hit
        out = _filter_bands(X, self.bands, self.sfreq, self.method)
        if len(_FB_CACHE) > 32:          # bounded: folds are revisited, not kept
            _FB_CACHE.clear()
        _FB_CACHE[key] = out
        return out


class PerBand(BaseEstimator, TransformerMixin):
    """Apply a per-band feature extractor and concatenate the results.

    Input is the stacked output of `FilterBank`; this splits it back into
    bands, fits one clone of `base` per band, and concatenates the features.
    That is the FBCSP construction when `base` is CSP, and filter-bank tangent
    space when it is Covariances + TangentSpace.
    """

    def __init__(self, base=None, n_bands: int = len(DEFAULT_BANDS)):
        self.base = base
        self.n_bands = n_bands

    def _split(self, X):
        X = np.asarray(X)
        n_ch = X.shape[1] // self.n_bands
        return [X[:, i * n_ch:(i + 1) * n_ch, :] for i in range(self.n_bands)]

    def fit(self, X, y=None):
        from sklearn.base import clone

        self.models_ = []
        for Xb in self._split(X):
            m = clone(self.base)
            m.fit(Xb, y)
            self.models_.append(m)
        return self

    def transform(self, X):
        return np.concatenate(
            [m.transform(Xb) for m, Xb in zip(self.models_, self._split(X))],
            axis=1)


def _cov(est: str = "oas"):
    return Covariances(estimator=est)


def make_filterbank_pipelines(
    bands=DEFAULT_BANDS, sfreq: float = 128.0, seed: int = 0
) -> dict[str, Pipeline]:
    """Filter-bank suite: FBCSP-class baselines and wide-register quantum kernels.

    Every pipeline sees exactly the same filter bank, so the classical and
    quantum families differ in what they do with it, not in what they get.
    """
    n_bands = len(bands)

    def fb():
        return FilterBank(bands=bands, sfreq=sfreq)

    pipes: dict[str, Pipeline] = {}

    # ---- Filter-bank classical baselines: the real bar ------------------
    # FBCSP: CSP per band, log-variance features concatenated, LDA.
    pipes["classical/FBCSP+LDA"] = Pipeline([
        ("fb", fb()),
        ("perband", PerBand(base=Pipeline([("cov", _cov()),
                                           ("csp", CSP(nfilter=4, log=True))]),
                            n_bands=n_bands)),
        ("clf", LDA(solver="lsqr", shrinkage="auto")),
    ])

    # Filter-bank tangent space: per-band Riemannian projection, concatenated.
    pipes["classical/FB-TS+LR"] = Pipeline([
        ("fb", fb()),
        ("perband", PerBand(base=Pipeline([("cov", _cov()),
                                           ("ts", TangentSpace(metric="riemann"))]),
                            n_bands=n_bands)),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=5000)),
    ])

    pipes["classical/FB-TS+RBF-SVM"] = Pipeline([
        ("fb", fb()),
        ("perband", PerBand(base=Pipeline([("cov", _cov()),
                                           ("ts", TangentSpace(metric="riemann"))]),
                            n_bands=n_bands)),
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])

    # ---- Wide-register quantum kernels ----------------------------------
    # One covariance over all stacked bands: n_bands*n_ch dimensions, i.e.
    # log2(n_bands*n_ch) qubits. No dimensionality reduction anywhere.
    def dens(kern, whiten=None):
        return Pipeline([("fb", fb()), ("cov", _cov()),
                         ("clf", DensityKernelSVC(kernel=kern, C=1.0,
                                                  whiten=whiten))])

    for kern, short in (("hs", "HS-overlap"), ("fidelity", "Fidelity"),
                        ("hs_rbf", "HS-RBF"), ("bures_rbf", "Bures-RBF"),
                        ("qre_rbf", "QRE-RBF")):
        pipes[f"quantum/FB-{short}-SVM"] = dens(kern)
        pipes[f"quantum/FB-{short}-ref-SVM"] = dens(kern, "riemann")

    # ---- Controls --------------------------------------------------------
    # Same wide covariance, classical geometries: isolates the geometry.
    pipes["control/FB-riemann-kernel-SVM"] = Pipeline([
        ("fb", fb()), ("cov", _cov()), ("clf", SPDKernelSVC(metric="riemann"))])
    pipes["control/FB-logeuclid-kernel-SVM"] = Pipeline([
        ("fb", fb()), ("cov", _cov()), ("clf", SPDKernelSVC(metric="logeuclid"))])

    return pipes


def make_filterbank_grids() -> dict[str, dict]:
    C_GRID = [0.1, 1.0, 10.0]
    GM = [0.25, 1.0, 4.0]
    g = {
        "classical/FBCSP+LDA": {"perband__base__csp__nfilter": [2, 4, 6]},
        "classical/FB-TS+LR": {"clf__C": C_GRID},
        "classical/FB-TS+RBF-SVM": {"clf__C": C_GRID,
                                    "clf__gamma": ["scale", 0.01, 0.1]},
        "control/FB-riemann-kernel-SVM": {"clf__C": C_GRID},
        "control/FB-logeuclid-kernel-SVM": {"clf__C": C_GRID},
    }
    for short in ("HS-overlap", "Fidelity", "HS-RBF", "Bures-RBF", "QRE-RBF"):
        rbf = short.endswith("RBF")
        for suffix in ("", "-ref"):
            grid = {"clf__C": C_GRID}
            if rbf:
                grid["clf__gamma_mult"] = GM
            g[f"quantum/FB-{short}{suffix}-SVM"] = grid
    return g
