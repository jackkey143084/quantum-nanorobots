"""Command-line entry point: ``qnrs <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Optional

from qnrs.adapters.base import CircuitProgram, GateOperation
from qnrs.core.capabilities import ControlProtocol
from qnrs.core.pulses import Envelope, PulseParameters, PulseSequence
from qnrs.runtime import QuantumRuntime, RuntimeConfig

BELL = CircuitProgram(
    num_qubits=2,
    operations=[
        GateOperation("H", [0]),
        GateOperation("CNOT", [0, 1]),
    ],
    shots=4096,
    seed=7,
)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _demo_pulse() -> PulseSequence:
    sequence = PulseSequence(sequence_id="demo-x-rotation", repetitions=2048, gap_ns=4.0)
    sequence.add(
        PulseParameters(
            frequency_ghz=5.12,
            amplitude=0.35,
            duration_ns=32.0,
            envelope=Envelope.DRAG,
            drag_beta=0.45,
            target="drive-0",
        )
    )
    sequence.add(
        PulseParameters(
            frequency_ghz=6.31,
            amplitude=0.2,
            duration_ns=48.0,
            envelope=Envelope.GAUSSIAN,
            target="readout",
        )
    )
    return sequence


async def _cmd_discover(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    await runtime.discover(refresh=True)
    devices = [
        {
            "device_id": m.device_id,
            "name": m.device_name,
            "adapter": runtime._device_adapter.get(m.device_id),
            "backend_type": m.capabilities.backend_type.value,
            "num_qubits": m.capabilities.num_qubits,
            "protocols": [p.value for p in m.capabilities.supported_protocols],
            "fidelity_2q": m.capabilities.two_qubit_fidelity,
            "t1_us": m.capabilities.t1_us,
            "t2_us": m.capabilities.t2_us,
        }
        for m in runtime.devices.values()
    ]
    if args.json:
        _print({"devices": devices})
        return 0
    print(f"{len(devices)} device(s):")
    for d in devices:
        print(f"  - {d['device_id']:<28} [{d['adapter']}] {d['backend_type']}, "
              f"{d['num_qubits']}q, 2q-fid={d['fidelity_2q']}, protocols={','.join(d['protocols'])}")
    return 0


async def _cmd_negotiate(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    required = [ControlProtocol(p) for p in args.required]
    try:
        result = await runtime.negotiate(args.device, required=required, allow_fallback=not args.strict)
    except Exception as exc:
        print(f"negotiation failed: {exc}", file=sys.stderr)
        return 2
    _print(result.to_dict())
    return 0


async def _cmd_plan(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    program = BELL if args.bell else CircuitProgram.from_dict(json.loads(args.program))
    await runtime.discover(refresh=False)
    _print(runtime.plan(program, required=[ControlProtocol(p) for p in args.required]))
    return 0


async def _cmd_circuit(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    program = BELL if args.bell else CircuitProgram.from_dict(json.loads(args.program))
    record = await runtime.submit_circuit(program, device_id=args.device)
    _print(record.to_dict(include_statevector=args.statevector))
    return 0 if record.status == "completed" else 1


async def _cmd_silq(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as handle:
            source = handle.read()
    else:
        source = args.source or ""
    if not source:
        print("provide --source or --file", file=sys.stderr)
        return 2
    silq = runtime.adapters["silq"]
    compiled = silq.compile(source, shots=args.shots)
    record = await runtime.submit_circuit(compiled.program, device_id=args.device)
    payload = record.to_dict(include_statevector=args.statevector)
    payload["frontend"] = {"qubit_names": compiled.qubit_names, "measurements": compiled.measurements}
    _print(payload)
    return 0 if record.status == "completed" else 1


async def _cmd_pulse(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    sequence = _demo_pulse()
    record = await runtime.submit_pulse(sequence, device_id=args.device)
    _print(record.to_dict())
    return 0 if record.status == "completed" else 1


async def _cmd_optimize(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    from qnrs.adapters.pennylane_adapter import PennyLaneAdapter, VariationalTemplate

    adapter = next((a for a in runtime.adapters.values() if isinstance(a, PennyLaneAdapter)), None)
    if adapter is None:
        print("PennyLane adapter unavailable", file=sys.stderr)
        return 2
    target: List[complex] = [1 / 2 ** 0.5, 0.0, 0.0, 1 / 2 ** 0.5]  # Bell state
    template = VariationalTemplate(num_qubits=2, layers=args.layers)
    result = await adapter.optimize(target, template=template, steps=args.steps,
                                    learning_rate=args.lr, seed=args.seed)
    _print(result.to_dict())
    return 0


async def _cmd_demo(runtime: QuantumRuntime, _args: argparse.Namespace) -> int:
    await runtime.discover(refresh=True)
    print("== discovered devices ==")
    for device_id, message in runtime.devices.items():
        caps = message.capabilities
        print(f"  {device_id:<28} {caps.backend_type.value:<22} {caps.num_qubits:>3}q")

    print("\n== plan for Bell circuit ==")
    _print(runtime.plan(BELL))

    print("\n== Bell circuit on NumPy simulator ==")
    record = await runtime.submit_circuit(BELL, device_id="numpy:simulator")
    _print({k: v for k, v in record.to_dict(include_statevector=False).items() if k != "request"})

    qiskit_ids = [d for d in runtime.devices if d.startswith("qiskit:")]
    if qiskit_ids:
        print(f"\n== Bell circuit on {qiskit_ids[0]} ==")
        record = await runtime.submit_circuit(BELL, device_id=qiskit_ids[0])
        _print({k: v for k, v in record.to_dict(include_statevector=False).items() if k != "request"})

    print("\n== Silq subset -> circuit ==")
    silq = runtime.adapters.get("silq")
    if silq is not None:
        compiled = silq.compile(
            """
            // prepare a Bell pair, then rotate q0
            H(q0);
            CNOT(q0, q1);
            RZ(pi/4, q0);
            x = measure(q0);
            """,
            shots=1024,
        )
        print(f"  compiled {len(compiled.program.operations)} gates, "
              f"{compiled.program.num_qubits} qubits, measurements={len(compiled.measurements)}")
        record = await runtime.submit_circuit(compiled.program, device_id="numpy:simulator")
        print(f"  counts: {record.result.counts if record.result else 'n/a'}")

    print("\n== pulse sequence on Labber (mock bridge) ==")
    record = await runtime.submit_pulse(_demo_pulse())
    if record.result:
        print(f"  device={record.device_id} status={record.status} "
              f"excitation={record.result.metadata.get('excitation')}")
        print(f"  readout channels={[r['channel'] for r in record.result.readout]}")

    print("\n== PennyLane variational state prep (Bell target) ==")
    try:
        from qnrs.adapters.pennylane_adapter import PennyLaneAdapter, VariationalTemplate

        adapter = next(a for a in runtime.adapters.values() if isinstance(a, PennyLaneAdapter))
        target = [1 / 2 ** 0.5, 0.0, 0.0, 1 / 2 ** 0.5]
        result = await adapter.optimize(
            target, template=VariationalTemplate(num_qubits=2, layers=2), steps=40, seed=3
        )
        print(f"  fidelity={result.final_fidelity:.6f} best_cost={result.best_cost:.6f} "
              f"engine={result.engine}")
    except StopIteration:
        print("  (PennyLane adapter disabled)")

    print("\n== telemetry ==")
    _print(runtime.telemetry())
    return 0


async def _cmd_serve(runtime: QuantumRuntime, args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required for `qnrs serve` (pip install 'qnrs[api]')", file=sys.stderr)
        return 2
    from qnrs.api.gateway import create_app

    app = create_app(runtime)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    await server.serve()
    return 0


COMMANDS = {
    "serve": _cmd_serve,
    "discover": _cmd_discover,
    "negotiate": _cmd_negotiate,
    "plan": _cmd_plan,
    "circuit": _cmd_circuit,
    "silq": _cmd_silq,
    "pulse": _cmd_pulse,
    "optimize": _cmd_optimize,
    "demo": _cmd_demo,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qnrs", description="Quantum Nanorobot Runtime control plane")
    parser.add_argument("--bus-key", default="", help="Shared HMAC key for discovery messages")
    parser.add_argument("--no-qiskit", action="store_true")
    parser.add_argument("--no-pennylane", action="store_true")
    parser.add_argument("--no-labber", action="store_true")
    parser.add_argument("--no-qcodes", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the HTTP/WebSocket gateway")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--log-level", default="info")

    p = sub.add_parser("discover", help="list discovered devices")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("negotiate", help="pick a control protocol for a device")
    p.add_argument("device")
    p.add_argument("--required", nargs="*", default=[])
    p.add_argument("--strict", action="store_true", help="fail instead of falling back")

    p = sub.add_parser("plan", help="rank devices for a circuit")
    p.add_argument("--program", help="JSON CircuitProgram payload")
    p.add_argument("--bell", action="store_true", help="use the built-in Bell circuit")
    p.add_argument("--required", nargs="*", default=[])

    p = sub.add_parser("circuit", help="submit a circuit program")
    p.add_argument("--program", help="JSON CircuitProgram payload")
    p.add_argument("--bell", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--statevector", action="store_true")

    p = sub.add_parser("silq", help="compile and run Silq-subset source")
    p.add_argument("--source", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.add_argument("--statevector", action="store_true")

    p = sub.add_parser("pulse", help="submit the demo pulse sequence")
    p.add_argument("--device", default=None)

    p = sub.add_parser("optimize", help="PennyLane variational state preparation")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    sub.add_parser("demo", help="end-to-end demo across every adapter")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RuntimeConfig(
        bus_key=args.bus_key,
        enable_qiskit=not args.no_qiskit,
        enable_pennylane=not args.no_pennylane,
        enable_labber=not args.no_labber,
        enable_qcodes=not args.no_qcodes,
    )
    runtime = QuantumRuntime(config)
    try:
        return asyncio.run(COMMANDS[args.command](runtime, args))
    except KeyboardInterrupt:
        return 130
    finally:
        with __import__("contextlib").suppress(Exception):
            asyncio.run(runtime.close())


if __name__ == "__main__":
    raise SystemExit(main())
