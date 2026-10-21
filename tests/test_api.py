import json

import pytest
from fastapi.testclient import TestClient

from qnrs.api.gateway import create_app
from qnrs.runtime import QuantumRuntime, RuntimeConfig

BELL = {
    "num_qubits": 2,
    "operations": [
        {"name": "H", "qubits": [0]},
        {"name": "CNOT", "qubits": [0, 1]},
    ],
    "shots": 512,
    "seed": 3,
}


@pytest.fixture()
def client():
    runtime = QuantumRuntime(RuntimeConfig())
    app = create_app(runtime)
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "numpy" in response.json()["adapters"]


def test_list_devices(client):
    response = client.get("/v1/devices")
    assert response.status_code == 200
    device_ids = {d["device_id"] for d in response.json()}
    assert "numpy:simulator" in device_ids


def test_get_device_and_404(client):
    assert client.get("/v1/devices/numpy:simulator").status_code == 200
    assert client.get("/v1/devices/nope:nope").status_code == 404


def test_negotiate_endpoint(client):
    response = client.post(
        "/v1/devices/numpy:simulator/negotiate",
        json={"required": ["gate_sequence"], "allow_fallback": True},
    )
    assert response.status_code == 200
    assert response.json()["protocol"] == "gate_sequence"


def test_negotiate_rejects_bad_protocol_name(client):
    response = client.post(
        "/v1/devices/numpy:simulator/negotiate",
        json={"required": ["teleportation"], "allow_fallback": True},
    )
    assert response.status_code == 422


def test_plan_endpoint(client):
    response = client.post("/v1/plan", json={"program": BELL})
    assert response.status_code == 200
    assert response.json()["recommended"]


def test_submit_circuit_job(client):
    response = client.post("/v1/jobs/circuit", json={"program": BELL, "device_id": "numpy:simulator"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert sum(payload["result"]["counts"].values()) == 512
    job_id = payload["job_id"]

    fetched = client.get(f"/v1/jobs/{job_id}")
    assert fetched.status_code == 200
    assert fetched.json()["job_id"] == job_id


def test_submit_invalid_circuit_returns_422(client):
    bad = {"program": {"num_qubits": 1, "operations": [{"name": "H", "qubits": [5]}]}}
    response = client.post("/v1/jobs/circuit", json=bad)
    assert response.status_code == 422


def test_submit_pulse_job(client):
    sequence = {
        "sequence_id": "api-pulse",
        "repetitions": 128,
        "gap_ns": 2.0,
        "pulses": [
            {
                "frequency_ghz": 5.0,
                "amplitude": 0.5,
                "duration_ns": 24.0,
                "envelope": "gaussian",
                "target": "drive-0",
            }
        ],
    }
    response = client.post("/v1/jobs/pulse", json={"sequence": sequence})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"


def test_silq_endpoint(client):
    response = client.post(
        "/v1/jobs/silq",
        json={"source": "H(q0); CNOT(q0, q1);", "shots": 256, "device_id": "numpy:simulator"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["frontend"]["qubit_names"] == {"q0": 0, "q1": 1}


def test_silq_syntax_error_returns_422(client):
    response = client.post("/v1/jobs/silq", json={"source": "NOTAGATE(q0);"})
    assert response.status_code == 422


def test_variational_endpoint(client):
    response = client.post(
        "/v1/jobs/variational",
        json={"target_state": {"real": [0.0, 1.0]}, "layers": 1, "steps": 40, "seed": 2},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["variational"]["final_fidelity"] > 0.9


def test_jobs_listing(client):
    client.post("/v1/jobs/circuit", json={"program": BELL, "device_id": "numpy:simulator"})
    response = client.get("/v1/jobs")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_telemetry_endpoint(client):
    response = client.get("/v1/telemetry")
    assert response.status_code == 200
    body = response.json()
    assert "adapters" in body and "devices" in body


def test_websocket_stream(client):
    with client.websocket_connect("/v1/stream") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "telemetry"
        assert "payload" in message
