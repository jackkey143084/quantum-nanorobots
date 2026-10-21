# Roadmap

How QNRS maps onto the original four-phase plan, and what is genuinely next.

Legend: ✅ done · 🟡 partial · ⬜ not started

---

## Phase 1 — Simulation & validation (months 1–6)

| Item | Status | Where |
|---|---|---|
| Classical state-vector simulator with nanoscale-relevant gate set | ✅ | `src/qnrs/sim/statevector.py` |
| Exact Schrödinger evolution (eigendecomposition, no series truncation) | ✅ | `core/state.py::QuantumStateManager.evolve` |
| Decoherence modelling (T1/T2, pure dephasing, Kraus channels) | ✅ | `core/state.py::DecoherenceModel`, `noise_kraus` |
| Potential-energy landscapes (van der Waals, electrostatic, molecular binding) | ⬜ | needs a physics module + units policy |
| Noise channels beyond amplitude/phase damping (depolarizing, correlated, 1/f) | 🟡 | only amplitude + phase damping today |
| Validation of control algorithms in simulation | 🟡 | gate/pulse paths validated; no GRAPE/CRAB comparison yet |

**Next concrete step:** add a `physics/` module with trap/electrode potential
models and a depolarizing channel, then a benchmark that compares simulated
nanorobot positioning trajectories against an analytic solution.

---

## Phase 2 — Hardware abstraction layer (months 6–12) ← *this deliverable*

| Item | Status | Where |
|---|---|---|
| Unified device capability model + validation | ✅ | `core/capabilities.py` |
| Discovery with signed advertisements | ✅ | `core/discovery.py`, HMAC-SHA256 + replay window |
| Protocol negotiation (gate/pulse/analog/feedback/hybrid) | ✅ | `core/discovery.py::NegotiationEngine` |
| Hardware-agnostic control protocols | ✅ | adapters expose one contract |
| Middleware translating high-level commands to low-level instructions | ✅ | Silq front-end, circuit IR → adapter, pulse IR → QCoDeS/Labber |
| Qiskit integration | ✅ | `adapters/qiskit_adapter.py` |
| Silq front-end | 🟡 | documented subset, not the official compiler |
| PennyLane ML integration | ✅ | `adapters/pennylane_adapter.py` (variational state prep) |
| Labber integration | 🟡 | documented JSON-RPC bridge + mock; on-host bridge script needed |
| QCoDeS integration | ✅ | `adapters/qcodes_adapter.py` |
| Unified REST/WebSocket API | ✅ | `api/gateway.py` (13 routes) |
| ROS 2 middleware | 🟡 | ament package + tested logic; not built in this sandbox |
| Real-time budgets (RTOS/FPGA-level timing) | 🟡 | software `TimingBudget` only; no RTOS pinning or FPGA path |

**Next concrete steps**
1. Ship the on-host Labber bridge script (Python, using Labber's own API) and an
   integration test against a real workstation.
2. `colcon build` the ROS 2 package in a real workspace and add launch files.
3. Replace the software timing budget with a real-time executor
   (PREEMPT_RT or an FPGA trigger path) and measure jitter.

---

## Phase 3 — Integration & testing (months 12–18)

| Item | Status | Where |
|---|---|---|
| Real hardware platforms (superconducting / trapped ion / photonic) | 🟡 | IBM Runtime path exists behind `QISKIT_IBM_TOKEN`; untested without credentials |
| Automated testing framework for quantum behaviour | ✅ | 83 tests incl. cross-adapter Bell-state agreement |
| Quantum state tomography / weak measurement | ⬜ | only state tracking + fidelity estimation |
| Readout error mitigation (confusion matrix inversion) | ✅ | `core/errors.py::_recover_readout` |
| Quantum error correction | 🟡 | logical-error estimate + checkpoint restore; **no decoder/syndromes** |

**Next concrete steps**
1. Implement a surface-code syndrome/decoder loop (MWPM) behind the existing
   `apply_error_correction` interface.
2. Add tomography endpoints (`/v1/jobs/tomography`) built on Qiskit's
   `quantum_info` and PennyLane's measurement primitives.
3. Record hardware baselines and turn the estimator constants (`A`, `p_th`) into
   calibrated values per backend.

---

## Phase 4 — Autonomous systems (months 18–24)

| Item | Status | Where |
|---|---|---|
| Adaptive control / RL | 🟡 | variational parameter optimization exists; no RL loop |
| Swarm coordination | ⬜ | needs a scheduler across many devices |
| Self-repair / fault tolerance | 🟡 | error strategies + checkpoint restore; no autonomous re-planning |
| Entanglement-based inter-robot communication | ⬜ | not modelled |

**Next concrete steps**
1. Add a swarm scheduler: assign N circuits across devices with a global
   fidelity/deadline objective (extend `QuantumRuntime.plan` to multi-job).
2. Wrap the PennyLane optimizer in a closed-loop controller that re-optimizes
   after each coherence drop reported by `/v1/stream`.
3. Model entanglement distribution and link budgets as a first-class device
   capability (`entanglement_links`).

---

## Engineering backlog (independent of phases)

- Persist the job store (SQLite/Postgres) so restarts do not lose history.
- Authentication on the gateway (bus key exists for discovery, not for HTTP).
- Structured job cancellation and per-device queueing with priorities.
- CI matrix across Python 3.10–3.12 with optional-dependency permutations.
- Property-based tests (Hypothesis) for pulse scheduling and IR round-trips.
- Package/publish to PyPI and build a container image.
