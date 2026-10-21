# QNRS — Quantum Nanorobot Runtime

[Русский](../README.md) · **English**

> Step 2 of the quantum-nanorobot software plan: **the API layer that actually
> talks to quantum instruments** — one contract over Qiskit, Silq, PennyLane,
> Labber, QCoDeS and ROS 2.

Not a 200-line sketch: a runnable package with 10 virtual instruments, a unified
REST/WebSocket gateway, a ROS 2 bridge, 86 tests and an end-to-end demo.

```bash
pip install -e ".[all]"
qnrs demo          # full pass across every adapter
qnrs serve         # HTTP + WebSocket gateway on :8080
```

## What it solves

Each instrument stack speaks a different language — Qiskit logical circuits,
Silq a high-level language, PennyLane differentiable circuits, Labber/QCoDeS
pulses and calibration parameters, ROS 2 robot messages. QNRS defines one
contract between them:

* **Device model** — `QuantumDeviceCapabilities` (qubits, native gates,
  protocols, T1/T2, gate times, fidelities) with physics validation (`T2 ≤ 2·T1`).
* **Discovery + negotiation** — HMAC-signed advertisements and a transparent
  protocol scorer (no black box): priority × fidelity × coherence headroom.
* **State tracking** — exact `exp(-iHt)` evolution, Kraus noise channels,
  checkpoints, surface-code logical-error estimates.
* **Pulse IR** — validated envelopes (Gaussian/square/ramp/Blackman/DRAG),
  overlap detection, sample-accurate waveforms.
* **Unified gateway** — REST + WebSocket; submit circuits, pulses, Silq source,
  or variational jobs; plan and rank devices.
* **ROS 2 bridge** — topics/services with the logic decoupled from `rclpy`.

Adapters are simultaneously a discovery `ProtocolNegotiator` and an executor, so
the gateway routes by `device_id` rather than by vendor name.

| Component | QNRS role | Protocols |
|---|---|---|
| Qiskit | gate execution (StatevectorSampler / Aer / IBM Runtime) | `gate_sequence`, `measurement_feedback` |
| Silq | high-level front-end subset → circuit IR | `gate_sequence` |
| PennyLane | differentiable circuits + variational optimization | `gate_sequence` |
| Labber | pulse-level instrument control over a JSON-RPC bridge | `pulse_sequence`, `analog_control` |
| QCoDeS | parameters, calibration, sweeps | `pulse_sequence`, `analog_control` |
| ROS 2 | robot integration (`qnrs_ros2` ament package) | topics + services |

## Quick start

```bash
pip install -e ".[all]"
qnrs discover
qnrs plan --bell
qnrs circuit --bell --statevector
qnrs silq --file examples/nanorobot_scan.silq
qnrs optimize --layers 3 --steps 80
qnrs serve --port 8080        # Swagger UI at /docs
```

```bash
curl -s -X POST localhost:8080/v1/jobs/circuit -H 'content-type: application/json' \
  -d '{"program":{"num_qubits":2,"operations":[{"name":"H","qubits":[0]},{"name":"CNOT","qubits":[0,1]}],"shots":4096}}' \
  | jq '.result.counts'
```

## Layout

```
src/qnrs/{core,adapters,protocols,sim,api,ros2}/   runtime.py   cli.py
ros2_ws/src/qnrs_ros2/                             ament package (msg/, srv/, node)
tests/          86 tests
examples/       Silq program, circuit JSON, pulse sequence
ARCHITECTURE.md PROTOCOLS.md ROADMAP.md
```

## Honest limitations

1. The **Silq adapter is a front-end subset**, not the official ETH Zurich
   compiler (`qfree`, quantum control flow and the full type system are out of scope).
2. **Labber is proprietary**; the adapter speaks a documented JSON-RPC bridge
   (see [PROTOCOLS.md](PROTOCOLS.md)) and ships an in-process mock so the full
   path runs without a workstation.
3. **Qiskit Pulse was removed in Qiskit 1.x**, so the Qiskit adapter no longer
   advertises `pulse_sequence`; it exports a schedule (`status="delegated"`) for
   Labber/QCoDeS to execute.
4. **Error correction is a model, not a decoder.** We estimate logical error
   rates and restore checkpoints; MWPM decoding and syndrome extraction are not
   implemented. (The original Duck.ai sketch multiplied fidelity by an invented
   1.15 factor — that is gone.)
5. The **ROS 2 package was not built here** (no `rclpy`/`colcon` in this sandbox);
   the bridge logic is pure Python and tested, the `rclpy` node builds in a normal
   ROS 2 workspace.
6. Performance numbers are estimates, not hardware benchmarks.

## Publishing to GitHub

```bash
python scripts/upload_to_github.py --repo <owner>/<repo> --dry-run
python scripts/upload_to_github.py --repo <owner>/<repo> --branch main --message "feat: QNRS 0.2.0"
```

Files are sent in batches of 25 through the connector's `github.files.commit`.
The repository must already exist and the connector needs repository access
(a GitHub App installation with that repo selected); check with
`connector-cli provider invoke github.repository.get --input '{"repo":"<owner>/<repo>"}'`.

See [ROADMAP.md](ROADMAP.md) for how this maps onto the 24-month plan.

MIT licensed — see [LICENSE](../LICENSE).
