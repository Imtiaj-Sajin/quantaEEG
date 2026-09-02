"""Quantum and quantum-inspired kernels for EEG covariance matrices.

Two families are implemented.

1. Density-matrix kernels (`hs_overlap_kernel`, `fidelity_kernel`).
   A trial's spatial covariance C is symmetric positive definite, so
   rho = C / tr(C) is a bona fide quantum density matrix on log2(n_channels)
   qubits. Quantum information geometry then applies to EEG directly, with no
   ad-hoc squeezing of features into rotation angles:

   - tr(rho sigma) -- Hilbert-Schmidt overlap. This is exactly the quantity a
     SWAP test estimates, and it is provably PSD (an inner product in
     Hilbert-Schmidt space), hence a valid SVM kernel with no repair needed.
   - F(rho, sigma) -- Uhlmann fidelity, the quantum generalisation of the
     Bhattacharyya coefficient. Its induced distance is the Bures metric,
     which coincides with the Bures-Wasserstein distance that appears in the
     Riemannian EEG literature as a distance but, to our knowledge, never as
     a kernel.

   HONEST SCOPE: at 8-64 channels both are classically computable in
   O(n_ch^3). They are quantum-information-geometric, not quantum-speedup.
   The hardware-native route is a SWAP test between purifications.

2. Circuit embedding kernels (`CircuitKernel`). Classical features are
   embedded in a parameterised circuit and the kernel is the state overlap
   |<phi(x)|phi(z)>|^2. Entanglement can be switched off, which is the
   ablation Bowles et al. (arXiv:2403.07059) identified as the decisive test
   of whether "quantumness" is doing any work.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


# --------------------------------------------------------------------------
# Density-matrix construction
# --------------------------------------------------------------------------

def to_density_matrices(covs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Normalise SPD covariance matrices to unit trace, giving density matrices.

    Parameters
    ----------
    covs : (n_trials, n_ch, n_ch) SPD matrices.

    Returns
    -------
    (n_trials, n_ch, n_ch) matrices with tr(rho) = 1 and rho >= 0.
    """
    covs = np.asarray(covs, dtype=np.float64)
    # Symmetrise, then clip eigenvalues to kill numerical drift below zero.
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    out = np.empty_like(covs)
    for i, C in enumerate(covs):
        w, V = eigh(C)
        w = np.clip(w, eps, None)
        C = (V * w) @ V.T
        out[i] = C / np.trace(C)
    return out


def von_neumann_entropy(rhos: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """S(rho) = -tr(rho log rho) in nats: a scalar spatial-mixedness feature."""
    out = np.empty(len(rhos))
    for i, r in enumerate(rhos):
        w = np.clip(eigh(r, eigvals_only=True), eps, None)
        w = w / w.sum()
        out[i] = -float(np.sum(w * np.log(w)))
    return out


def purity(rhos: np.ndarray) -> np.ndarray:
    """tr(rho^2), in [1/d, 1]. High purity means spatially focal activity."""
    return np.einsum("nij,nji->n", rhos, rhos)


# --------------------------------------------------------------------------
# Density-matrix kernels
# --------------------------------------------------------------------------

def hs_overlap_kernel(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """K_ij = tr(rho_i sigma_j), the SWAP-test observable. Provably PSD.

    Since tr(rho sigma) is a genuine inner product on Hermitian matrices, the
    Gram matrix is positive semi-definite by construction.
    """
    B = A if B is None else B
    # tr(A_i B_j) = sum_kl A_i[k,l] B_j[l,k]; both operands are symmetric.
    return np.einsum("ikl,jkl->ij", A, B)


def _sqrtm_spd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, V = eigh(M)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def fidelity_kernel(
    A: np.ndarray,
    B: np.ndarray | None = None,
    squared: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Uhlmann fidelity F(rho, sigma) = (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    `squared=True` returns F itself; `squared=False` returns sqrt(F), the
    Bhattacharyya-like overlap whose induced metric is the Bures distance.
    """
    symmetric = B is None
    B = A if symmetric else B

    sqrtA = np.stack([_sqrtm_spd(a, eps) for a in A])
    K = np.empty((len(A), len(B)))
    # One batched eigendecomposition per row rather than one per pair: at 288
    # trials this is the difference between ~83k Python iterations and 288.
    for i in range(len(A)):
        Si = sqrtA[i]
        M = Si @ B @ Si                       # (n_B, d, d) by broadcasting
        M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
        w = np.clip(np.linalg.eigvalsh(M), eps, None)
        K[i] = np.sqrt(w).sum(axis=1)
    if symmetric:
        # Fidelity is symmetric; averaging removes any numerical asymmetry.
        K = 0.5 * (K + K.T)
    return K**2 if squared else K


# --------------------------------------------------------------------------
# Quantum distances and bandwidth-parameterised kernels
# --------------------------------------------------------------------------
#
# Raw overlap kernels concentrate badly on EEG covariances: every pair of
# trials has tr(rho sigma) ~ 0.99, so the Gram matrix is nearly rank-one and
# an SVM cannot separate anything. This is the kernel-concentration pathology
# documented across QML (Thanasilp et al., "Exponential concentration in
# quantum kernel methods"). The standard remedy is to keep the quantum
# *geometry* but restore scale sensitivity by exponentiating the induced
# distance with a tunable bandwidth, exactly as an RBF kernel does for the
# Euclidean distance.

def hs_distance_sq(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Squared Hilbert-Schmidt distance tr((rho - sigma)^2), pairwise."""
    B = A if B is None else B
    ovl = hs_overlap_kernel(A, B)
    pa = np.einsum("nij,nji->n", A, A)
    pb = np.einsum("nij,nji->n", B, B)
    D = pa[:, None] + pb[None, :] - 2.0 * ovl
    return np.clip(D, 0.0, None)


def bures_distance_sq(
    A: np.ndarray, B: np.ndarray | None = None, eps: float = 1e-12
) -> np.ndarray:
    """Squared Bures distance 2(1 - sqrt(F)), the quantum-information metric.

    For unit-trace density matrices this is the Bures-Wasserstein distance,
    the geodesic distance of the Fubini-Study/Bures metric on state space.
    """
    root_f = fidelity_kernel(A, B, squared=False, eps=eps)
    return np.clip(2.0 * (1.0 - root_f), 0.0, None)


def median_bandwidth(D: np.ndarray) -> float:
    """Median heuristic: gamma = 1 / median(off-diagonal squared distance).

    A scale-free default that puts the kernel in its sensitive regime,
    matching the convention used for classical RBF kernels.
    """
    n = D.shape[0]
    if n > 1 and D.shape[0] == D.shape[1]:
        off = D[~np.eye(n, dtype=bool)]
    else:
        off = D.ravel()
    med = float(np.median(off))
    return 1.0 / med if med > 1e-12 else 1.0


def psd_project(K: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues of a symmetric Gram matrix to zero.

    A guard, not a routine step: it makes the SVM solve unconditionally
    well-posed. Use `min_eig` to check whether it was ever needed.
    """
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    if w.min() >= -tol:
        return K
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T


def min_eig(K: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric Gram matrix (a PSD diagnostic)."""
    return float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())


# --------------------------------------------------------------------------
# Parameterised-circuit embedding kernels
# --------------------------------------------------------------------------

class CircuitKernel:
    """State-overlap kernel from a PennyLane feature map.

    The kernel is evaluated by preparing each embedded state once on a
    state-vector simulator and forming the Gram matrix from those states.
    This is mathematically identical to running the adjoint-circuit ("kernel
    estimation") circuit at infinite shots, but costs O(n * 2^q) rather than
    O(n^2) circuit executions.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, and the number of input features consumed per layer.
    n_layers : int
        Repetitions of the (rotation, entangler) block, i.e. feature-map depth.
    entangle : bool
        If False, every entangling gate is removed. The kernel then factorises
        into single-qubit kernels and is classically trivial. This is the
        ablation that isolates what entanglement contributes.
    entangler : {"iqp", "ring"}
        "iqp" uses data-dependent ZZ couplings (the Havlicek et al.
        ZZFeatureMap family, conjectured hard to simulate classically);
        "ring" uses fixed CNOTs.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        entangle: bool = True,
        entangler: str = "iqp",
        scale: float = 1.0,
    ):
        import pennylane as qml

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.entangle = bool(entangle)
        self.entangler = entangler
        self.scale = float(scale)
        self._qml = qml
        self._dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self._dev)
        def _state(x):
            self._feature_map(x)
            return qml.state()

        self._state_fn = _state

    def _feature_map(self, x) -> None:
        qml = self._qml
        nq = self.n_qubits
        for _ in range(self.n_layers):
            for w in range(nq):
                qml.Hadamard(wires=w)
                qml.RZ(self.scale * x[w], wires=w)
            if not self.entangle:
                continue
            if self.entangler == "iqp":
                for w in range(nq - 1):
                    qml.CNOT(wires=[w, w + 1])
                    qml.RZ(
                        self.scale * (np.pi - x[w]) * (np.pi - x[w + 1]),
                        wires=w + 1,
                    )
                    qml.CNOT(wires=[w, w + 1])
            elif self.entangler == "ring":
                for w in range(nq):
                    qml.CNOT(wires=[w, (w + 1) % nq])
            else:
                raise ValueError(f"unknown entangler {self.entangler!r}")

    def states(self, X: np.ndarray) -> np.ndarray:
        """Embedded state vectors, shape (n_samples, 2**n_qubits), complex.

        Uses PennyLane parameter broadcasting to prepare the whole batch in
        one device execution, falling back to a per-sample loop if the device
        rejects the broadcast.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"expected {self.n_qubits} features, got {X.shape[1]}")
        try:
            out = np.asarray(self._state_fn(X.T))
            if out.shape == (len(X), 2 ** self.n_qubits):
                return out
        except Exception:  # noqa: BLE001 - broadcast unsupported, use the loop
            pass
        return np.stack([np.asarray(self._state_fn(x)) for x in X])

    def __call__(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        SA = self.states(A)
        SB = SA if B is None else self.states(B)
        return np.abs(SA.conj() @ SB.T) ** 2
