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

    The two ``_rbf`` variants exist because the raw overlaps concentrate on
    EEG data (all pairwise values ~0.99); exponentiating the induced distance
    keeps the quantum geometry while restoring scale sensitivity.
    ``gamma="median"`` uses the median heuristic fitted on the training split.
    """

    def __init__(
        self,
        kernel: str = "bures_rbf",
        C: float = 1.0,
        gamma: float | str = "median",
        gamma_mult: float = 1.0,
        normalize: bool = True,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.gamma_mult = gamma_mult
        self.normalize = normalize

    # -- kernel families ------------------------------------------------
    def _is_distance_kernel(self) -> bool:
        return self.kernel in ("hs_rbf", "bures_rbf")

    def _dist_sq(self, A, B=None):
        if self.kernel == "hs_rbf":
            return hs_distance_sq(A, B)
        if self.kernel == "bures_rbf":
            return bures_distance_sq(A, B)
        raise ValueError(f"{self.kernel!r} is not a distance kernel")

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
        self.rho_train_ = to_density_matrices(X)

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
        rho = to_density_matrices(X)
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


def make_pipelines(n_qubits: int = N_QUBITS, seed: int = 0) -> dict[str, Pipeline]:
    """Return the full benchmark suite, keyed by name.

    Groups
    ------
    classical/*  strong, standard MI baselines (the bar to clear)
    quantum/*    quantum kernels
    control/*    dimension-matched classical controls and ablations
    """
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
    def _dens(kern):
        return Pipeline([("cov", _cov()), ("clf", DensityKernelSVC(kernel=kern, C=1.0))])

    # Raw overlaps: reported to document the concentration pathology.
    pipes["quantum/HS-overlap-SVM"] = _dens("hs")
    pipes["quantum/Fidelity-SVM"] = _dens("fidelity")
    # Bandwidth-corrected quantum-geometric kernels: the working versions.
    pipes["quantum/HS-RBF-SVM"] = _dens("hs_rbf")
    pipes["quantum/Bures-RBF-SVM"] = _dens("bures_rbf")

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

    return pipes


GROUPS = ("classical", "quantum", "control")
