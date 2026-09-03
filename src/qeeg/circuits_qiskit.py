"""Qiskit rendering of the circuit feature maps, verified against the real code.

The circuit figure in the manuscript must not be a hand-drawn impression of the
circuit. It has to be produced from a circuit that is provably the same one the
experiments executed, otherwise the drawing and the code can drift apart and
nothing catches it.

So this module does two things:

1. `build_feature_map` reconstructs the feature map of `quantum.CircuitKernel`
   gate for gate in Qiskit.
2. `verify_against_pennylane` checks that claim numerically, by comparing the
   full Gram matrices the two implementations produce on random inputs. The
   figure script calls it before drawing and refuses to draw if it fails.

Qiskit and PennyLane order qubits oppositely in the statevector, so the
comparison is made on kernel values, which are basis-order independent, rather
than on raw amplitudes.
"""

from __future__ import annotations

import numpy as np


def build_feature_map(
    x,
    n_qubits: int,
    n_layers: int = 2,
    entangle: bool = True,
    entangler: str = "iqp",
    scale: float = 1.0,
    parameterised: bool = False,
):
    """Qiskit circuit matching `quantum.CircuitKernel._feature_map`.

    With ``parameterised=True`` the rotation angles are Qiskit ``Parameter``
    objects, which is what makes a readable figure; otherwise numeric values
    are bound, which is what makes the numerical check possible.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(n_qubits)
    if parameterised:
        xs = [Parameter(f"x{i}") for i in range(n_qubits)]
    else:
        xs = list(np.asarray(x, dtype=float))

    for _ in range(n_layers):
        for w in range(n_qubits):
            qc.h(w)
            qc.rz(scale * xs[w], w)
        if not entangle:
            continue
        if entangler == "iqp":
            for w in range(n_qubits - 1):
                qc.cx(w, w + 1)
                if parameterised:
                    # Parameter arithmetic cannot express a product of two
                    # Parameters, so the figure shows the coupling symbolically.
                    qc.rz(Parameter(f"z{w}{w+1}"), w + 1)
                else:
                    qc.rz(scale * (np.pi - xs[w]) * (np.pi - xs[w + 1]), w + 1)
                qc.cx(w, w + 1)
        elif entangler == "ring":
            for w in range(n_qubits):
                qc.cx(w, (w + 1) % n_qubits)
        else:
            raise ValueError(f"unknown entangler {entangler!r}")
    return qc


def qiskit_states(X, **kw) -> np.ndarray:
    """Statevectors from the Qiskit construction, one row per sample."""
    from qiskit.quantum_info import Statevector

    return np.stack([
        np.asarray(Statevector(build_feature_map(x, **kw)))
        for x in np.asarray(X, dtype=float)
    ])


def qiskit_kernel(X, **kw) -> np.ndarray:
    """Gram matrix |<phi(x)|phi(z)>|^2 from the Qiskit construction."""
    S = qiskit_states(X, **kw)
    return np.abs(S.conj() @ S.T) ** 2


def verify_against_pennylane(n_qubits: int = 3, n_layers: int = 2,
                             n_samples: int = 12, seed: int = 0,
                             tol: float = 1e-10) -> dict:
    """Compare Qiskit and PennyLane Gram matrices for every configuration.

    Returns the worst absolute deviation per configuration. Raises if any
    exceeds `tol`, so a drifted figure cannot reach the manuscript.
    """
    from .quantum import CircuitKernel

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, np.pi, size=(n_samples, n_qubits))

    configs = [
        ("IQP entangled", dict(entangle=True, entangler="iqp")),
        ("Ring CNOT", dict(entangle=True, entangler="ring")),
        ("Ablation (no entangler)", dict(entangle=False, entangler="iqp")),
    ]

    report = {}
    for label, kw in configs:
        K_pl = CircuitKernel(n_qubits=n_qubits, n_layers=n_layers, **kw)(X)
        K_qk = qiskit_kernel(X, n_qubits=n_qubits, n_layers=n_layers, **kw)
        dev = float(np.max(np.abs(K_pl - K_qk)))
        report[label] = dev
        if dev > tol:
            raise AssertionError(
                f"Qiskit circuit does not match the executed PennyLane circuit "
                f"for {label}: max |dK| = {dev:.3e} > {tol:.1e}"
            )
    return report


if __name__ == "__main__":
    rep = verify_against_pennylane()
    print("Qiskit vs PennyLane, max |Gram difference|:")
    for k, v in rep.items():
        print(f"  {k:26s} {v:.3e}")
    print("All configurations identical to numerical precision.")
