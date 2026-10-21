"""Silq front-end adapter.

Scope note (important): this is a **front-end subset** of Silq's surface syntax
that lowers to the QNRS circuit IR. It is not, and does not claim to be, the
official Silq compiler from ETH Zurich. The subset exists so a nanorobot control
program can be written in a high-level, ``qfree``-style language and executed on
any QNRS backend.

Supported statements::

    // comment
    H(q0);
    X(q1);  Y(q0);  Z(q2);  S(q0);  T(q1);
    RX(1.5708, q0);  RY(0.5, q1);  RZ(3.1416, q2);
    CNOT(q0, q1);  CX(q0, q1);  CZ(q0, q1);  SWAP(q0, q1);
    x = measure(q0);
    forget(q0);

Qubits are named ``q0, q1, ...``; indices are derived from the highest index
seen (or from an explicit ``qubits N;`` pragma).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from qnrs.adapters.base import (
    AdapterError,
    CircuitProgram,
    ExecutionResult,
    GateOperation,
    QuantumBackendAdapter,
)
from qnrs.core.capabilities import (
    ControlProtocol,
    DiscoveryMessage,
    QuantumBackendType,
    QuantumDeviceCapabilities,
)

_SINGLE_QUBIT = {"H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "SX"}
_PARAMETRISED = {"RX", "RY", "RZ", "PHASE", "P"}
_TWO_QUBIT = {"CNOT", "CX", "CZ", "SWAP"}
_CALL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*;?\s*$")
_QUBIT = re.compile(r"^q(\d+)$", re.IGNORECASE)


class SilqSyntaxError(AdapterError):
    """Raised on a statement the front-end subset cannot parse."""


@dataclass
class SilqCompileResult:
    program: CircuitProgram
    qubit_names: Dict[str, int] = field(default_factory=dict)
    measurements: List[Tuple[str, int]] = field(default_factory=list)
    source_lines: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "program": self.program.to_dict(),
            "qubit_names": self.qubit_names,
            "measurements": self.measurements,
            "source_lines": self.source_lines,
        }


class SilqCompiler:
    """Lowers the supported Silq subset into :class:`CircuitProgram`."""

    def __init__(self, default_shots: int = 1024) -> None:
        self.default_shots = default_shots

    def compile(self, source: str, shots: Optional[int] = None) -> SilqCompileResult:
        qubit_names: Dict[str, int] = {}
        next_index = 0
        operations: List[GateOperation] = []
        measurements: List[Tuple[str, int]] = []
        declared_qubits: Optional[int] = None

        def resolve(name: str) -> int:
            nonlocal next_index
            name = name.strip()
            match = _QUBIT.match(name)
            if match:
                index = int(match.group(1))
                qubit_names[name.lower()] = index
                next_index = max(next_index, index + 1)
                return index
            if name.isdigit():
                index = int(name)
                next_index = max(next_index, index + 1)
                return index
            if name in qubit_names:
                return qubit_names[name]
            raise SilqSyntaxError(f"unknown qubit {name!r}")

        lines = source.splitlines()

        def parse_statement(statement: str, lineno: int, raw: str) -> None:
            nonlocal declared_qubits
            statement = statement.strip()
            if not statement or statement in ("{", "}"):
                return

            pragma = re.match(r"^qubits\s+(\d+)$", statement)
            if pragma:
                declared_qubits = int(pragma.group(1))
                return

            call = _CALL.match(statement)
            if not call:
                # Accept Silq-style declarations / assignments whose RHS is a
                # call, e.g. ``x := measure(q0);`` or ``r = measure(q0);``.
                assignment = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*(?::=|=)\s*(.+)$", statement)
                if assignment:
                    rhs = assignment.group(1).strip()
                    call = _CALL.match(rhs)
                    if call is None:
                        return  # plain declaration such as ``x := 0:𝔹;``
                    statement = rhs
                elif statement.endswith("{"):
                    return  # a def/if block header; body lines are parsed independently
                else:
                    raise SilqSyntaxError(f"line {lineno}: cannot parse {raw!r}")

            func, arglist = call.group(1), call.group(2)
            args = [a.strip() for a in arglist.split(",") if a.strip()]
            upper = func.upper()

            if upper == "MEASURE":
                if len(args) != 1:
                    raise SilqSyntaxError(f"line {lineno}: measure() takes one qubit")
                measurements.append((args[0], resolve(args[0])))
            elif upper == "FORGET":
                if len(args) != 1:
                    raise SilqSyntaxError(f"line {lineno}: forget() takes one qubit")
                resolve(args[0])
            elif upper in _SINGLE_QUBIT:
                if len(args) != 1:
                    raise SilqSyntaxError(f"line {lineno}: {func}() takes one qubit")
                operations.append(GateOperation(name=upper, qubits=[resolve(args[0])]))
            elif upper in _PARAMETRISED:
                if len(args) != 2:
                    raise SilqSyntaxError(f"line {lineno}: {func}() takes (angle, qubit)")
                angle = _parse_angle(args[0], lineno)
                operations.append(GateOperation(name=upper, qubits=[resolve(args[1])], params=[angle]))
            elif upper in _TWO_QUBIT:
                if len(args) != 2:
                    raise SilqSyntaxError(f"line {lineno}: {func}() takes two qubits")
                operations.append(GateOperation(name=upper, qubits=[resolve(args[0]), resolve(args[1])]))
            else:
                raise SilqSyntaxError(f"line {lineno}: unsupported operation {func!r}")

        for lineno, raw in enumerate(lines, start=1):
            line = raw.split("//", 1)[0].strip()
            if not line or line in ("{", "}"):
                continue
            # Allow several statements per line: ``RZ(pi/4, q0); RX(-pi, q1);``
            statements = [s for s in line.split(";") if s.strip()] or [line]
            for statement in statements:
                parse_statement(statement, lineno, raw)

        num_qubits = declared_qubits or max(next_index, 1)
        if any(op.qubits and max(op.qubits) >= num_qubits for op in operations):
            raise SilqSyntaxError(
                f"operation targets a qubit >= declared qubit count ({num_qubits})"
            )
        program = CircuitProgram(
            num_qubits=num_qubits,
            operations=operations,
            shots=shots or self.default_shots,
            measure_all=not measurements,
            metadata={"frontend": "silq-subset", "measurements": len(measurements)},
        )
        return SilqCompileResult(
            program=program,
            qubit_names=qubit_names,
            measurements=measurements,
            source_lines=len(lines),
        )


def _parse_angle(token: str, lineno: int) -> float:
    """Parse ``1.57``, ``pi``, ``-pi``, ``2*pi``, ``pi/4`` or ``3*pi/2``."""
    token = token.strip()
    pi_forms = {"pi": 1.0, "-pi": -1.0, "π": 1.0, "+pi": 1.0}
    if token in pi_forms:
        return pi_forms[token] * 3.141592653589793
    match = re.match(r"^([+-]?[\d.]*)\s*\*?\s*(?:pi|π)\s*(?:/\s*([\d.]+))?$", token)
    if match:
        coefficient = match.group(1)
        if coefficient in ("", "+"):
            coefficient = "1"
        elif coefficient == "-":
            coefficient = "-1"
        divisor = match.group(2)
        value = float(coefficient) * 3.141592653589793
        if divisor:
            value /= float(divisor)
        return value
    try:
        return float(token)
    except ValueError as exc:
        raise SilqSyntaxError(f"line {lineno}: cannot parse angle {token!r}") from exc


class SilqAdapter(QuantumBackendAdapter):
    """Compiles the Silq subset and delegates execution to a backend adapter."""

    name = "silq"

    def __init__(self, bus_key: str = "", delegate: Optional[QuantumBackendAdapter] = None) -> None:
        super().__init__(bus_key=bus_key)
        self.compiler = SilqCompiler()
        self._delegate = delegate
        self._caps = QuantumDeviceCapabilities(
            backend_type=QuantumBackendType.SIMULATOR,
            num_qubits=20,
            native_gates=sorted(_SINGLE_QUBIT | _PARAMETRISED | _TWO_QUBIT),
            supported_protocols=[ControlProtocol.GATE_SEQUENCE],
            max_circuit_depth=10_000,
            max_shots=10_000_000,
            gate_time_1q_ns=20.0,
            gate_time_2q_ns=120.0,
            metadata={"engine": "silq-frontend -> delegate", "frontend": True},
        )

    def set_delegate(self, delegate: QuantumBackendAdapter) -> None:
        self._delegate = delegate

    def compile(self, source: str, shots: Optional[int] = None) -> SilqCompileResult:
        return self.compiler.compile(source, shots=shots)

    async def discover(self) -> List[DiscoveryMessage]:
        message = self._advertise("silq:compiler", "Silq front-end (QNRS subset)", self._caps)
        return self._remember([message])

    async def verify(self, device_id: str) -> bool:
        return device_id == "silq:compiler" and self._delegate is not None

    async def capabilities(self, device_id: str) -> QuantumDeviceCapabilities:
        if device_id != "silq:compiler":
            raise AdapterError(f"unknown Silq device {device_id!r}")
        return self._caps

    async def execute_circuit(self, device_id: str, program: CircuitProgram) -> ExecutionResult:
        if self._delegate is None:
            raise AdapterError("Silq front-end has no execution delegate configured")
        return await self._delegate.execute_circuit("numpy:simulator", program)

    async def execute_source(self, source: str, shots: Optional[int] = None,
                             delegate_device: str = "numpy:simulator",
                             delegate: Optional[QuantumBackendAdapter] = None) -> ExecutionResult:
        """Compile Silq source and run it on ``delegate`` (default: numpy sim)."""
        target = delegate or self._delegate
        if target is None:
            raise AdapterError("no execution delegate configured")
        compiled = self.compile(source, shots=shots)
        result = await target.execute_circuit(delegate_device, compiled.program)
        result.metadata.setdefault("frontend", "silq-subset")
        result.metadata.setdefault("source_lines", compiled.source_lines)
        return result
