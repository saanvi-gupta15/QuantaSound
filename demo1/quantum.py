"""PRELIMINARY: exact statevector simulation of p=1 QAOA for four binary choices."""
import time
import numpy as np
from scipy.optimize import minimize

SOURCES = ["engine", "typing", "speech", "siren"]


def problem(preferences=None):
    # z_i=1 means request cancellation. Positive preference -> block.
    preference = np.array([1.0, 0.6, -1.0, -1.0] if preferences is None else preferences, dtype=float)
    if preference.shape != (4,) or not np.all(np.isfinite(preference)):
        raise ValueError("Supply four finite preferences in source order")
    states = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(float)
    reward = np.array([3., 2., 4., 5.]) * preference
    # Quadratic toy cost: benefit, pair interactions and exactly-two soft budget.
    energies = (-states @ reward + 0.8 * states[:, 0] * states[:, 1]
                + 2.0 * (states.sum(axis=1) - 2)**2)
    return states, energies


def statevector(angles, energies):
    gamma, beta = angles
    psi = np.exp(-1j * gamma * energies) / np.sqrt(len(energies))
    n = int(np.log2(len(energies)))
    for bit in range(n):
        flipped = np.arange(len(psi)) ^ (1 << bit)
        psi = np.cos(beta) * psi - 1j * np.sin(beta) * psi[flipped]
    return psi


def qiskit_circuit(angles, energies):
    """Same p=1 cost/mixer evolution as statevector(), using Qiskit gates."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    gamma, beta = angles
    n = int(np.log2(len(energies)))
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    circuit.append(DiagonalGate(np.exp(-1j * gamma * energies)), range(n))
    circuit.rx(2 * beta, range(n))
    return circuit


def qiskit_statevector(angles, energies):
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector.from_instruction(qiskit_circuit(angles, energies)).data)


def benchmark(preferences=None, backend="numpy"):
    if backend not in ("numpy", "qiskit"):
        raise ValueError("backend must be numpy or qiskit")
    simulate = statevector if backend == "numpy" else qiskit_statevector
    states, energies = problem(preferences)
    start = time.perf_counter()
    best = int(np.argmin(energies))
    exact_seconds = time.perf_counter() - start
    start = time.perf_counter()
    def expectation(angles):
        return float(np.abs(simulate(angles, energies))**2 @ energies)
    trials = [minimize(expectation, initial, method="Nelder-Mead", options={"maxiter":300})
              for initial in ([0.2, 0.3], [0.8, 0.6], [1.4, 0.9], [2., 0.4])]
    fit = min(trials, key=lambda result: result.fun)
    probability = np.abs(simulate(fit.x, energies))**2
    elapsed = time.perf_counter() - start
    result = {"status":"PRELIMINARY: toy offline simulation; no quantum advantage claim",
              "backend":backend, "angles":fit.x.tolist(),
              "sources":SOURCES, "exact_bits":states[best].astype(int).tolist(),
              "exact_cost":float(energies[best]), "qaoa_expected_cost":float(fit.fun),
              "qaoa_expected_gap":float(fit.fun - energies[best]),
              "optimal_probability":float(probability[np.isclose(energies, energies.min())].sum()),
              "most_likely_bits":states[probability.argmax()].astype(int).tolist(),
              "optimizer_success":bool(fit.success), "objective_evaluations":sum(t.nfev for t in trials),
              "exact_argmin_seconds":exact_seconds, "qaoa_optimization_seconds":elapsed,
              "timing_scope":"Shared cost table excluded; timings are illustrative, not a scaling study"}
    return states, energies, probability, result
