# Examples

| File | Run it |
|---|---|
| `nanorobot_scan.silq` | `qnrs silq --file examples/nanorobot_scan.silq --shots 4096` |
| `ghz_circuit.json` | `qnrs circuit --program "$(cat examples/ghz_circuit.json)" --device qiskit:statevector` |
| `pulse_sequence.json` | `curl -X POST localhost:8080/v1/jobs/pulse -H 'content-type: application/json' -d "{\"sequence\": $(cat examples/pulse_sequence.json)}"` |

Or run the whole pipeline in Python:

```bash
PYTHONPATH=src python examples/run_examples.py
```
