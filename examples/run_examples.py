"""End-to-end example: run every example artifact through the runtime.

    PYTHONPATH=src python examples/run_examples.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qnrs.adapters.base import CircuitProgram  # noqa: E402
from qnrs.core.pulses import PulseSequence  # noqa: E402
from qnrs.runtime import QuantumRuntime, RuntimeConfig  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent


async def main() -> None:
    runtime = QuantumRuntime(RuntimeConfig())
    try:
        await runtime.discover()
        print(f"discovered {len(runtime.devices)} devices")

        # 1. Silq front-end subset -> circuit -> execution
        source = (EXAMPLES / "nanorobot_scan.silq").read_text(encoding="utf-8")
        silq = runtime.adapters["silq"]
        compiled = silq.compile(source, shots=4096)
        record = await runtime.submit_circuit(compiled.program, device_id="qiskit:statevector")
        print("\n[silq] qubits:", compiled.program.num_qubits,
              "gates:", len(compiled.program.operations),
              "\n[silq] counts:", record.result.counts if record.result else record.error)

        # 2. Circuit JSON, run on two backends and compare
        program = CircuitProgram.from_dict(
            json.loads((EXAMPLES / "ghz_circuit.json").read_text(encoding="utf-8"))
        )
        for device in ("numpy:simulator", "qiskit:statevector"):
            outcome = await runtime.submit_circuit(program, device_id=device)
            top = sorted(outcome.result.counts.items(), key=lambda kv: -kv[1])[:3]
            print(f"\n[circuit] {device} top counts: {top}")

        # 3. Pulse sequence JSON -> Labber bridge
        sequence = PulseSequence.from_dict(
            json.loads((EXAMPLES / "pulse_sequence.json").read_text(encoding="utf-8"))
        )
        pulse_record = await runtime.submit_pulse(sequence)
        print(f"\n[pulse] device={pulse_record.device_id} status={pulse_record.status} "
              f"excitation={pulse_record.result.metadata.get('excitation') if pulse_record.result else None}")

        # 4. Plan for the circuit (ranked, auditable)
        plan = runtime.plan(program)
        print(f"\n[plan] recommended={plan['recommended']}")
        for candidate in plan["candidates"][:3]:
            print(f"       {candidate['device_id']:<28} score={candidate['score']}")

        print("\n[telemetry]", json.dumps(runtime.telemetry()["jobs"]))
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
