"""Classical and quantum pipelines, and the controls that make them comparable.

Every quantum pipeline here is paired with a classical control that sees the
*same features at the same dimensionality*. Without that pairing, an apparent
quantum win is usually just a difference in preprocessing.

Leakage discipline: any transform with fitted state (tangent-space reference
mean, scalers, PCA) lives inside an sklearn Pipeline, so it is refit on the
training split of every CV fold. The density-matrix kernels have no fitted
state, but they are still wrapped in an estimator that stores only training
covariances, so the same code path is fold-safe.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace

from .quantum import (
    CircuitKernel,
    bures_distance_sq,
    fidelity_kernel,
    hs_distance_sq,
    hs_overlap_kernel,
    median_bandwidth,
    min_eig,
    psd_project,
    qre_divergence,
    reference_whitener,
    to_density_matrices,
)

N_QUBITS = 4  # feature-map width for circuit kernels; also the PCA rank


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

class DensityKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM on a quantum density-matrix kernel over EEG covariance matrices.

    Input X is a stack of SPD covariance matrices, shape (n_trials, n_ch, n_ch).
    Each is normalised to unit trace, making it a density matrix on
    log2(n_ch) qubits.

    Kernels
    -------
    ``hs``            tr(rho sigma), the raw SWAP-test overlap.
    ``fidelity``      F(rho, sigma), the Uhlmann fidelity.
    ``hs_rbf``        exp(-gamma * ||rho - sigma||_HS^2).
    ``bures_rbf``     exp(-gamma * d_Bures^2), i.e. the quantum-information
                      metric with a tunable bandwidth.
    ``qre_rbf``       exp(-gamma * J), J the symmetrised quantum relative
                      entropy.

    The ``_rbf`` variants exist because the raw overlaps concentrate on EEG
    data (all pairwise values ~0.99); exponentiating the induced distance
    keeps the quantum geometry while restoring scale sensitivity.
    ``gamma="median"`` uses the median heuristic fitted on the training split.

    Reference state
    ---------------
    ``whiten`` selects the frame the states are expressed in. ``None`` is the
    sensor frame: rho = C/tr(C) as the electrodes deliver it. A metric name
    ("riemann", "logeuclid", "euclid") instead measures each state relative to
    the Frechet mean M of the *training* covariances, rho -> W C W / tr(W C W)
    with W = M^-1/2.

    This is not cosmetic. In the sensor frame these kernels are invariant only
    under orthogonal congruence, while EEG's nuisance group is the full
    congruence group (electrode gain, referencing, source mixing, session and
    subject changes). In the reference frame they become exactly
    affine-invariant, matching the invariance that makes tangent-space
    decoding the classical state of the art -- and note that pyriemann's
    ``TangentSpace`` already whitens by that same mean internally, so a
    sensor-frame quantum kernel is not being compared like for like against
    it. See `qeeg.quantum.reference_whitener` and `qeeg.reference`.
    """

    def __init__(
        self,
        kernel: str = "bures_rbf",
        C: float = 1.0,
        gamma: float | str = "median",
        gamma_mult: float = 1.0,
        normalize: bool = True,
        whiten: str | None = None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.gamma_mult = gamma_mult
        self.normalize = normalize
        self.whiten = whiten

    # -- kernel families ------------------------------------------------
    def _is_distance_kernel(self) -> bool:
        return self.kernel in ("hs_rbf", "bures_rbf", "qre_rbf")

    def _dist_sq(self, A, B=None):
        if self.kernel == "hs_rbf":
            return hs_distance_sq(A, B)
        if self.kernel == "bures_rbf":
            return bures_distance_sq(A, B)
        if self.kernel == "qre_rbf":
            return qre_divergence(A, B)
        raise ValueError(f"{self.kernel!r} is not a distance kernel")

    def _to_states(self, X):
        """Covariances -> density matrices, in the configured frame."""
        X = np.asarray(X, dtype=np.float64)
        if self.whiten:
            X = self.whitener_ @ X @ self.whitener_
        return to_density_matrices(X)

    def _overlap(self, A, B=None):
        if self.kernel == "hs":
            return hs_overlap_kernel(A, B)
        if self.kernel == "fidelity":
            return fidelity_kernel(A, B, squared=True)
        if self.kernel == "sqrt_fidelity":
            return fidelity_kernel(A, B, squared=False)
        raise ValueError(f"unknown kernel {self.kernel!r}")

    # -- sklearn API ----------------------------------------------------
    def fit(self, X, y):
        if self.whiten:
            # Reference state from the training split only: no leakage.
            self.whitener_ = reference_whitener(X, metric=self.whiten)
        self.rho_train_ = self._to_states(X)

        if self._is_distance_kernel():
            D = self._dist_sq(self.rho_train_)
            # Bandwidth is fitted on training data only: no leakage.
            base = (
                median_bandwidth(D)
                if isinstance(self.gamma, str) and self.gamma == "median"
                else float(self.gamma)
            )
            self.gamma_ = float(self.gamma_mult) * base
            K = np.exp(-self.gamma_ * D)
        else:
            K = self._overlap(self.rho_train_)
            if self.normalize:
                self.diag_train_ = np.sqrt(np.clip(np.diag(K), 1e-12, None))
                K = K / np.outer(self.diag_train_, self.diag_train_)

        self.min_eig_ = min_eig(K)
        K = psd_project(K)
        self.svc_ = SVC(kernel="precomputed", C=self.C)
        self.svc_.fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def _test_gram(self, X):
        rho = self._to_states(X)
        if self._is_distance_kernel():
            return np.exp(-self.gamma_ * self._dist_sq(rho, self.rho_train_))
        K = self._overlap(rho, self.rho_train_)
        if self.normalize:
            d_test = np.sqrt(np.clip(np.diag(self._overlap(rho)), 1e-12, None))
            K = K / np.outer(d_test, self.diag_train_)
        return K

    def predict(self, X):
        return self.svc_.predict(self._test_gram(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_gram(X))


class CircuitKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM on a parameterised-circuit embedding kernel.

    Input X is a plain feature matrix (n_samples, n_qubits), already scaled
    into a rotation-angle range by the surrounding pipeline.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = 2,
        entangle: bool = True,
        entangler: str = "iqp",
        C: float = 1.0,
        scale: float = 1.0,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.entangler = entangler
        self.C = C
        self.scale = scale

    def _make(self):
        return CircuitKernel(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entangle=self.entangle,
            entangler=self.entangler,
            scale=self.scale,
        )

    def fit(self, X, y):
        self.kernel_ = self._make()
        self.states_train_ = self.kernel_.states(X)
        K = np.abs(self.states_train_.conj() @ self.states_train_.T) ** 2
        self.svc_ = SVC(kernel="precomputed", C=self.C)
        self.svc_.fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def _test_gram(self, X):
        S = self.kernel_.states(X)
        return np.abs(S.conj() @ self.states_train_.T) ** 2

    def predict(self, X):
        return self.svc_.predict(self._test_gram(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_gram(X))


class SPDKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM on a classical SPD-manifold kernel, the density kernels' twin.

    pyriemann supplies ``riemann``, ``logeuclid`` and ``euclid`` kernels on
    SPD matrices but ships no classifier that uses them directly; the standard
    pipelines project to the tangent space first. Using them as precomputed
    SVM kernels gives the density-matrix kernels a comparator that differs from
    them in the *geometry* alone -- same input, same classifier, same tuning
    budget, only the metric changes.

    Both pyriemann kernels are centred on a reference matrix, defaulting to the
    Frechet mean of the training set, so these are reference-frame methods by
    construction. That is the point of the pairing.
    """

    def __init__(self, metric: str = "riemann", C: float = 1.0):
        self.metric = metric
        self.C = C

    def fit(self, X, y):
        from pyriemann.geometry.kernel import kernel
        from pyriemann.geometry.mean import mean_covariance

        X = np.asarray(X, dtype=np.float64)
        self._kernel = kernel
        self.cref_ = mean_covariance(X, metric=self.metric)
        self.X_train_ = X
        K = kernel(X, X, Cref=self.cref_, metric=self.metric)
        self.min_eig_ = min_eig(K)
        self.svc_ = SVC(kernel="precomputed", C=self.C)
        self.svc_.fit(psd_project(K), y)
        self.classes_ = self.svc_.classes_
        return self

    def _test_gram(self, X):
        return self._kernel(np.asarray(X, dtype=np.float64), self.X_train_,
                            Cref=self.cref_, metric=self.metric)

    def predict(self, X):
        return self.svc_.predict(self._test_gram(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._test_gram(X))


class LogVariance(BaseEstimator):
    """Log-variance of each channel: the classic minimal MI feature."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(np.var(X, axis=-1) + 1e-12)


# --------------------------------------------------------------------------
# Pipeline registry
# --------------------------------------------------------------------------

def _cov(est: str = "oas"):
    """Shrinkage covariance. With ~45 trials, OAS beats the sample estimator."""
    return Covariances(estimator=est)


def make_pipelines(
    n_qubits: int = N_QUBITS, seed: int = 0, suite: str = "core"
) -> dict[str, Pipeline]:
    """Return the benchmark suite, keyed by name.

    Groups
    ------
    classical/*  strong, standard MI baselines (the bar to clear)
    quantum/*    quantum kernels
    control/*    dimension-matched classical controls and ablations

    Parameters
    ----------
    suite : {"core", "extended"}
        ``core`` is the 15-pipeline suite the published results were produced
        with; it is the default so those runs reproduce byte for byte.
        ``extended`` adds the reference-state quantum kernels, the quantum
        relative-entropy kernel, and the Riemannian/log-Euclidean kernel
        controls that are the density-matrix kernels' exact classical twins.
    """
    if suite not in ("core", "extended"):
        raise ValueError(f"unknown suite {suite!r}")
    pipes: dict[str, Pipeline] = {}

    # ---- Classical baselines -------------------------------------------
    pipes["classical/logvar+LDA"] = Pipeline([
        ("feat", LogVariance()),
        ("clf", LDA(solver="lsqr", shrinkage="auto")),
    ])

    pipes["classical/CSP+LDA"] = Pipeline([
        ("cov", _cov()),
        ("csp", CSP(nfilter=4, log=True)),
        ("clf", LDA(solver="lsqr", shrinkage="auto")),
    ])

    pipes["classical/MDM"] = Pipeline([
        ("cov", _cov()),
        ("clf", MDM(metric="riemann")),
    ])

    pipes["classical/TS+LR"] = Pipeline([
        ("cov", _cov()),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])

    pipes["classical/TS+RBF-SVM"] = Pipeline([
        ("cov", _cov()),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ])

    # ---- Quantum: density-matrix kernels on full covariances -----------
    # No dimensionality reduction: the density matrix uses every channel.
    def _dens(kern, whiten=None):
        return Pipeline([
            ("cov", _cov()),
            ("clf", DensityKernelSVC(kernel=kern, C=1.0, whiten=whiten)),
        ])

    # Raw overlaps: reported to document the concentration pathology.
    pipes["quantum/HS-overlap-SVM"] = _dens("hs")
    pipes["quantum/Fidelity-SVM"] = _dens("fidelity")
    # Bandwidth-corrected quantum-geometric kernels: the working versions.
    pipes["quantum/HS-RBF-SVM"] = _dens("hs_rbf")
    pipes["quantum/Bures-RBF-SVM"] = _dens("bures_rbf")

    if suite == "extended":
        # The same four kernels, plus the quantum relative entropy, evaluated
        # relative to the training-set reference state instead of in the sensor
        # frame. This is the like-for-like comparison against TangentSpace,
        # which whitens by that same mean before it does anything else.
        pipes["quantum/HS-overlap-ref-SVM"] = _dens("hs", "riemann")
        pipes["quantum/Fidelity-ref-SVM"] = _dens("fidelity", "riemann")
        pipes["quantum/HS-RBF-ref-SVM"] = _dens("hs_rbf", "riemann")
        pipes["quantum/Bures-RBF-ref-SVM"] = _dens("bures_rbf", "riemann")
        pipes["quantum/QRE-RBF-SVM"] = _dens("qre_rbf")
        pipes["quantum/QRE-RBF-ref-SVM"] = _dens("qre_rbf", "riemann")

    # ---- Quantum: circuit embedding kernels on reduced TS features -----
    def _circuit(entangle: bool, entangler: str = "iqp", layers: int = 2):
        return Pipeline([
            ("cov", _cov()),
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=seed)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
            ("clf", CircuitKernelSVC(
                n_qubits=n_qubits, n_layers=layers,
                entangle=entangle, entangler=entangler, C=1.0,
            )),
        ])

    pipes["quantum/IQP-kernel-SVM"] = _circuit(entangle=True, entangler="iqp")
    pipes["quantum/CNOT-kernel-SVM"] = _circuit(entangle=True, entangler="ring")

    # ---- Controls ------------------------------------------------------
    # (a) Entanglement ablation: identical circuit, entanglers deleted.
    #     If this matches the entangled version, "quantumness" is decorative.
    pipes["control/IQP-no-entangle"] = _circuit(entangle=False)

    # (b) Dimension-matched classical kernels: same PCA(n_qubits) features,
    #     ordinary kernels. This is the honest comparison for the circuit
    #     kernels, which only ever see n_qubits dimensions.
    def _matched(clf):
        return Pipeline([
            ("cov", _cov()),
            ("ts", TangentSpace(metric="riemann")),
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=seed)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
            ("clf", clf),
        ])

    pipes["control/PCA-matched-RBF"] = _matched(SVC(kernel="rbf", C=1.0, gamma="scale"))
    pipes["control/PCA-matched-linear"] = _matched(SVC(kernel="linear", C=1.0))

    # (c) Euclidean control for the density kernels: tr(rho sigma) on
    #     trace-normalised covariances is a linear kernel in HS space, so a
    #     plain linear SVM on vectorised covariances is its classical twin.
    pipes["control/logeuclid-TS+LR"] = Pipeline([
        ("cov", _cov()),
        ("ts", TangentSpace(metric="logeuclid")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])

    if suite == "extended":
        # (d) The density-matrix kernels' exact classical twins: SPD kernels
        #     from pyriemann, used as SVM kernels rather than as tangent-space
        #     projections. `kernel_riemann` is the affine-invariant geometry
        #     the quantum kernels are competing against; `kernel_logeuclid` is
        #     the flat-log one. Both are centred on the training Frechet mean
        #     by pyriemann itself, which is exactly the reference-state
        #     construction, so these pair with the quantum/*-ref pipelines.
        pipes["control/riemann-kernel-SVM"] = Pipeline([
            ("cov", _cov()), ("clf", SPDKernelSVC(metric="riemann")),
        ])
        pipes["control/logeuclid-kernel-SVM"] = Pipeline([
            ("cov", _cov()), ("clf", SPDKernelSVC(metric="logeuclid")),
        ])

    return pipes


GROUPS = ("classical", "quantum", "control")
