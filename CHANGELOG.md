# Changelog

## 0.2.0 — 2026-09-12

First runnable release of the quantum-instrument API layer (Step 2 of the plan).

### Added
- **Core abstraction layer** (`qnrs.core`): device capability model with
  physics validation (`T2 ≤ 2·T1`), HMAC-signed discovery advertisements with a
  replay window, transparent protocol negotiation scoring, exact unitary
  evolution, Kraus noise channels, surface-code logical-error estimation,
  checkpoints/restore, pulse IR with overlap detection, and microsecond
  `TimingBudget`.
- **Adapters**: Qiskit (gate execution on `StatevectorSampler` / `AerSimulator`,
  optional IBM Runtime), Silq front-end subset, PennyLane (differentiable
  execution + variational optimization), Labber (JSON-RPC bridge + in-process
  mock server), QCoDeS (parameters, snapshots, pulse sweeps), and a
  dependency-free NumPy state-vector baseline.
- **Unified gateway** (`qnrs.api`): FastAPI app with 12 REST routes plus a
  WebSocket telemetry stream, pydantic request models, plan/negotiate endpoints
  and a job store.
- **Runtime facade** (`qnrs.runtime`): discovery, device planning
  (fidelity × coherence headroom), job routing, telemetry.
- **ROS 2 bridge**: `qnrs_ros2` ament package (3 messages, 3 services, `rclpy`
  node) with transport-agnostic bridge logic testable without ROS.
- **CLI** (`qnrs`): `serve`, `discover`, `negotiate`, `plan`, `circuit`, `silq`,
  `pulse`, `optimize`, `demo`.
- **Docs**: README (ru/en), ARCHITECTURE, PROTOCOLS, ROADMAP.
- **Tests**: 86 tests covering state math, pulses, discovery/negotiation,
  every adapter, runtime routing, the HTTP/WS API and the ROS bridge.

### Fixed during development
- Qiskit 2.x measurement fields are named after the classical register
  (`c`), not `meas`; counts extraction is now capability-based.
- Circuits always append terminal measurements on the sampling path, so a
  program carrying an explicit `measure()` no longer collapses to `00…0`.
- `qiskit:aer-noisy` capabilities are registered, so it is executable.
- Labber/QCoDeS instrument names are made unique per adapter instance
  (QCoDeS keeps a process-global registry) while QNRS device ids stay stable.
- Qiskit no longer advertises `pulse_sequence` (Qiskit Pulse was removed in
  Qiskit 1.x); pulse routing prefers Labber/QCoDeS instrument drivers.

### Notes / limitations
- Silq support is a documented subset, not the official ETH Zurich compiler.
- Error correction is a rate model plus checkpoint restore — no MWPM decoder.
- The ROS 2 package has not been built here (no `rclpy`/`colcon`); its logic is
  tested in pure Python.
