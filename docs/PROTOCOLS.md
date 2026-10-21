# Protocols and wire formats

Four contracts make the stack interoperable. Each is versioned and testable.

---

## 1. Device capability record

Canonical form (JSON, snake_case). Used by discovery, `/v1/devices`, and the ROS
`DeviceCapabilities` message.

```json
{
  "backend_type": "superconducting_qubit",
  "num_qubits": 6,
  "native_gates": ["id", "rz", "sx", "x", "cx", "measure"],
  "supported_protocols": ["gate_sequence", "pulse_sequence", "measurement_feedback"],
  "max_circuit_depth": 1000,
  "max_shots": 200000,
  "t1_us": 180.0,
  "t2_us": 140.0,
  "gate_time_1q_ns": 35.0,
  "gate_time_2q_ns": 320.0,
  "single_qubit_fidelity": 0.9993,
  "two_qubit_fidelity": 0.989,
  "readout_fidelity": 0.972,
  "supports_mid_circuit_measurement": true,
  "calibration_timestamp": 1789193500.0,
  "metadata": {"engine": "qiskit_aer", "ideal": false}
}
```

Validation (rejected with `ProtocolError`):

* `num_qubits ≥ 1`; fidelities in `[0, 1]`; times `≥ 0`;
* `T2 ≤ 2·T1` (tolerance 5 %) — physical bound;
* `supports_mid_circuit_measurement=true` implies `measurement_feedback`.

`backend_type ∈ {superconducting_qubit, trapped_ion, photonic, neutral_atom,
nv_center, simulator, hybrid}`.

**Control protocols and priority** (lower number wins when scoring ties):

| Protocol | Meaning | Priority |
|---|---|---|
| `measurement_feedback` | mid-circuit readout + conditional branch | 0 |
| `pulse_sequence` | explicit analog waveforms | 1 |
| `hybrid_control` | gates now, pulses for calibration | 2 |
| `gate_sequence` | logical gates, device transpiles | 3 |
| `analog_control` | continuous drive / DC bias | 4 |

---

## 2. Discovery + negotiation

### 2.1 Advertisement

```json
{
  "device_id": "labber:nanorobot-drive-1",
  "device_name": "Nanorobot Drive Unit 1",
  "endpoint": "labber-bridge",
  "nonce": "9f1c2ab73d4e5f60",
  "timestamp": 1789193500.123,
  "capabilities": { ... },
  "signature": "a3f1…"
}
```

Signature = `HMAC-SHA256(bus_key, canonical_json(body))` where `body` excludes
`signature`, keys are sorted and separators are compact. Verification is
constant-time and also enforces a replay window (default: not older than 300 s,
not more than 5 s in the future). With no bus key configured, signatures are
skipped (development mode) and a warning is attached to `metadata`.

### 2.2 Negotiation

`POST /v1/devices/{device_id}/negotiate`

```json
{"required": ["pulse_sequence"], "allow_fallback": true}
```

Score for protocol `p` on device `d`:

```
score(p, d) = (10 − priority(p))
            + 2 · fidelity(p, d)
            + 0.5 · headroom(d)
            + 0.5 · [p ∈ required]
```

where `fidelity` is protocol-specific (readout × 2q for feedback, 2q for pulses,
1q × 2q otherwise) and `headroom = min(1, T_eff / duration(64 layers))`.

Response:

```json
{
  "device_id": "labber:nanorobot-drive-1",
  "protocol": "pulse_sequence",
  "score": 11.47,
  "candidates": [["pulse_sequence", 11.47], ["analog_control", 10.93]],
  "notes": ""
}
```

If none of `required` is supported: `allow_fallback=false` → HTTP 409;
`allow_fallback=true` → best native protocol, with `notes` explaining the fallback.
An unhealthy device (failed `verify`) is rejected regardless.

---

## 3. Circuit IR (`CircuitProgram`)

```json
{
  "num_qubits": 2,
  "operations": [
    {"name": "H", "qubits": [0]},
    {"name": "RZ", "qubits": [0], "params": [0.7853981634]},
    {"name": "CNOT", "qubits": [0, 1]}
  ],
  "shots": 4096,
  "measure_all": true,
  "seed": 7,
  "metadata": {}
}
```

Supported names: `I H X Y Z S SDG T TDG SX RX RY RZ PHASE/P CNOT/CX CZ SWAP`.
Angles are radians. Out-of-range qubits are rejected at construction.

### 3.1 Silq front-end subset

Accepted statements (one per line or `;`-separated):

```
// comment
qubits 3;                 // optional explicit width
H(q0);  X(q1);  Z(q2);
RX(1.5708, q0);  RY(0.5, q1);  RZ(pi/4, q2);
CNOT(q0, q1);  CX(q0, q1);  CZ(q0, q1);  SWAP(q0, q1);
x := measure(q0);         // assignment form accepted
forget(q1);
```

Angle constants: `pi`, `-pi`, `2*pi`, `pi/4`, `3*pi/2`. Qubit indices come from
`q<N>` or plain integers. Unsupported operations raise `SilqSyntaxError` (HTTP 422).
This is a **subset front-end**, not the official Silq compiler.

---

## 4. Pulse IR (`PulseSequence`)

```json
{
  "sequence_id": "demo-x-rotation",
  "repetitions": 2048,
  "gap_ns": 4.0,
  "pulses": [
    {"frequency_ghz": 5.12, "amplitude": 0.35, "duration_ns": 32.0,
     "phase_rad": 0.0, "envelope": "drag", "drag_beta": 0.45, "target": "drive-0"},
    {"frequency_ghz": 6.31, "amplitude": 0.2, "duration_ns": 48.0,
     "phase_rad": 0.0, "envelope": "gaussian", "target": "readout"}
  ]
}
```

* `envelope ∈ {gaussian, square, ramp, drag, blackman}`; `drag` requires `drag_beta ≠ 0`.
* `amplitude ∈ [0, 1]`, `duration_ns > 0`, `frequency_ghz ≥ 0`.
* `start_ns` is optional; omitted means "append after the previous pulse + gap".
* Scheduling rejects overlap **on the same channel** (`TimingError`); two channels
  may be driven simultaneously.

`analog_program()` yields the transport form:

```json
[{"channel": "drive-0", "start_ns": 0.0, "duration_ns": 32.0,
  "iq": [[0.0, 0.0], [0.007, 0.02], "..."],
  "envelope": "drag", "frequency_ghz": 5.12}]
```

---

## 5. Labber bridge (JSON-RPC 2.0, newline-delimited TCP)

Default port `5590`. One JSON object per line; requests and responses are
correlated by `id`.

| Method | Params | Result |
|---|---|---|
| `system.health` | `{}` | `{status, uptime_s, temperature_mk, devices}` |
| `device.list` | `{}` | `{devices: [{id, name, kind, num_qubits}]}` |
| `device.describe` | `{device}` | channels, sample rate, T1/T2, fidelities |
| `pulse.execute` | `{device, program, repetitions, shots}` | `{status, counts, readout, excitation}` |
| `instrument.set` | `{name, value}` | `{name, value}` |
| `instrument.get` | `{name}` | `{name, value}` |
| `mixer.tune` | `{channel}` | `{if_freq_mhz, lo_freq_ghz, phase_offset_rad}` |

Request:

```json
{"jsonrpc": "2.0", "id": "7", "method": "pulse.execute",
 "params": {"device": "labber:nanorobot-drive-1", "program": [...],
            "repetitions": 2048, "shots": 2048}}
```

Errors follow JSON-RPC `{"error": {"code", "message"}}`; the client raises
`LabberRpcError` with the code and data preserved. `MockLabberServer` implements
this exact surface in-process (used by the demo and CI).

Because Labber's own API is proprietary, a small bridge script on the Labber host
(using its Python API) is required to expose these methods — the wire format above
is the contract.

---

## 6. ROS 2 interface (`qnrs_ros2`)

### Topics

| Topic | Type | Content |
|---|---|---|
| `/qnrs/telemetry` | `std_msgs/String` | runtime telemetry JSON |
| `/qnrs/state` | `qnrs_ros2/QuantumState` | fidelity, coherence level, T1/T2, age |
| `/qnrs/capabilities` | `qnrs_ros2/DeviceCapabilities` | full capability record |

### Services

| Service | Type | Request → Response |
|---|---|---|
| `/qnrs/negotiate` | `NegotiateProtocol` | `device_id, required_protocols[], allow_fallback` → `success, protocol, score, message` |
| `/qnrs/execute_pulse` | `ExecutePulseSequence` | `device_id, sequence_json` → `success, job_id, counts_json, excitation, message` |
| `/qnrs/submit_circuit` | `SubmitCircuit` | `program_json, device_id` → `success, job_id, counts_json, fidelity, message` |

`QuantumBridge` (in `qnrs.ros2.bridge_logic`) implements the handlers and is
tested without ROS; the `rclpy` node is a thin adapter over it.

---

## 7. Gateway endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness, enabled adapters |
| GET | `/v1/telemetry` | uptime, jobs, coherence, error stats |
| GET | `/v1/devices` | discovery (refreshes caches) |
| GET | `/v1/devices/{id}` | one device's capabilities |
| POST | `/v1/devices/{id}/negotiate` | pick a control protocol |
| POST | `/v1/plan` | rank devices for a circuit |
| POST | `/v1/jobs/circuit` | submit a circuit program |
| POST | `/v1/jobs/pulse` | submit a pulse sequence |
| POST | `/v1/jobs/silq` | compile + run Silq-subset source |
| POST | `/v1/jobs/variational` | PennyLane variational optimization |
| GET | `/v1/jobs` | recent jobs |
| GET | `/v1/jobs/{id}` | one job (+ result) |
| WS | `/v1/stream` | telemetry stream; send `{"interval": 2.0}` to change period |

Error semantics: `422` malformed/invalid payload, `404` unknown device or job,
`409` protocol/route conflict, `503` adapter disabled.
