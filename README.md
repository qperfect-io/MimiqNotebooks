# Quantum Routing at the Strasbourg Christmas Markets
### Quantum in Practice — QPerfect

This repository contains the notebook and companion code from the **Quantum in Practice** live session.

---

## Main Notebook

**`strasbourg_quantum_routing.ipynb`** — the notebook walked through during the session.

End-to-end quantum workflow on a real-world combinatorial problem: routing visitors and bike couriers through the Christmas-market sites of central Strasbourg, from problem formulation to a **49-qubit QAOA simulation** on MIMIQ's MPS engine.

Topics covered:
- Problem setup: TSP and VRP on the Strasbourg market map
- QUBO formulation
- QAOA circuit construction
- Running on MIMIQ (cloud MPS simulator)
- Result analysis and visualization

---

## Derivative Examples

The `examples/` folder contains focused notebooks that each isolate one aspect of the main demo:

| Notebook | Topic |
|---|---|
| `strasbourg_module.ipynb` | Tour of the `strasbourg_markets_demo` helper package |
| `qubo_mappings.ipynb` | QUBO formulations for TSP and VRP |
| `qaoa_demo.ipynb` | QAOA circuit construction and parameter optimization |
| `qaoa_server_side.ipynb` | Running QAOA jobs on the MIMIQ cloud |
| `tsp_classical.ipynb` | Classical TSP solver (OR-Tools) for comparison |
| `vrp_classical.ipynb` | Classical VRP solver (OR-Tools) for comparison |
| `fidelity_sweep.ipynb` | Bond-dimension fidelity sweep on MPS simulation |

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. MIMIQ credentials

The quantum simulation cells require a MIMIQ account. Sign up at [mimiq.qperfect.io](https://mimiq.qperfect.io) and authenticate:

```python
import mimiqcircuits as mc
conn = mc.MimiqConnection()
conn.connect(email="your@email.com", password="your_password")
```

### 3. Launch JupyterLab

```bash
uv run jupyter lab
```

Open `strasbourg_quantum_routing.ipynb` to start.

---

## Dependencies

Python ≥ 3.12. Key packages (installed automatically with the companion package):

- `mimiqcircuits` — QPerfect MIMIQ SDK
- `numpy`, `matplotlib`, `scipy`
- `networkx` — graph utilities
- `ortools` — classical solver (Google OR-Tools)
- `folium`, `geopy` — map visualization
- `jupyterlab`, `ipywidgets`

---

*QPerfect — [qperfect.io](https://qperfect.io)*
