"""QNRS — Quantum Nanorobot Runtime.

A hardware-agnostic control plane for quantum instruments. See ARCHITECTURE.md.

Layers
------
core      : device model, discovery/negotiation, state tracking, pulse IR, timing.
adapters  : concrete drivers for Qiskit, Silq front-end, PennyLane ML, Labber, QCoDeS.
api       : unified REST/WebSocket gateway (the "API interface" layer).
ros2      : ROS 2 bridge package (messages, services, node logic).
"""

from qnrs.version import __version__

__all__ = ["__version__"]
