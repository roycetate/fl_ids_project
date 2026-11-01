# FL-IDS: Closed-Loop Explainable Federated Learning for Cyber Threat Detection

A research prototype for **cross-enterprise network intrusion detection**. It trains a Graph Attention Network (GAT) on CICIDS2017 flow features, either in one place (centralized) or across simulated organizations with Flower FedAvg (federated), then serves predictions from a local web dashboard.

The full title of the work is *A Closed-Loop Explainable Federated Learning Framework for Real-Time Cross-Enterprise Cyber Threat Detection and Autonomous Incident Response.*

## What this project does

Raw CICIDS2017 CSVs are cleaned, scaled, and split into per-organization shards. Each shard is turned into **k-NN similarity graphs** (a flow is a node; edges connect behaviorally similar flows). A GAT classifies each flow into one of 13 CICIDS2017 classes (BENIGN plus attack types such as DoS Hulk, PortScan, DDoS, Bot, web attacks).

Federated training keeps each “organization” on its own graphs and only shares model weights. A FastAPI app lets you inspect metrics, simulate FedAvg rounds, and score uploaded flow CSVs.

On the full CICIDS2017 MachineLearningCSV set (CPU runs in `results/full/`):

| Setting | Accuracy | Weighted-F1 | Macro-F1 |
|---|---:|---:|---:|
| Centralized GAT | 0.9734 | 0.9747 | 0.6979 |
| Federated IID | 0.9593 | 0.9613 | 0.5835 |
| Federated non-IID (α = 0.5) | 0.9647 | 0.9660 | 0.6215 |

Macro-F1 is lower because rare classes (Heartbleed, SQL injection, XSS) remain hard. Common classes (BENIGN, DDoS, DoS Hulk, PortScan) score much higher. Details are in `results/experiment_summary.md`.

## Repo layout

```
fl_ids_project/
├── config.py                 # paths, FL clients, GAT hyperparameters
├── requirements.txt          # training stack (pandas, torch, Flower, …)
├── app/
│   ├── run_app.py            # launch dashboard + API
│   ├── backend/              # FastAPI inference and metrics
│   └── frontend/             # single-page UI
├── src/
│   ├── data_preprocessing/   # load, clean, scale, FL partition
│   ├── graph_construction/   # k-NN graphs for GAT
│   ├── models/gat.py         # GAT classifier
│   └── training/             # centralized, federated, diagnose, ablation
├── data/
│   ├── raw/                  # put CICIDS2017 CSVs here (not in git)
│   └── processed_iid/        # small sample shards + graphs in this repo
├── results/                  # saved metrics and model checkpoints
└── tests/make_dummy_data.py  # synthetic CSVs for a pipeline smoke test
```

Large CICIDS files (`data/raw/`, `data/processed/`, `data/processed_iid_full/`) are **not** on GitHub. Download them locally as described below.

## Requirements

- Python 3.10+ recommended
- CPU works; a GPU is faster for full-data GAT training
- `torch` + `torch-geometric` must match your platform (see [PyG install](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if `pip install -r requirements.txt` fails on those two)

## Setup

```bash
git clone https://github.com/roycetate/fl_ids_project.git
cd fl_ids_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r app/requirements_app.txt
```

## Quick start (no real dataset)

Smoke-test preprocessing with fake CICIDS-shaped CSVs:

```bash
python tests/make_dummy_data.py
python -m src.data_preprocessing.pipeline
python -m src.graph_construction.build_graphs
```

Then open the dashboard (uses an untrained GAT unless you load a checkpoint):

```bash
python app/run_app.py
```

- UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health

Stop with Ctrl+C.

**Delete `data/raw/*.csv` before using the real dataset** so dummy rows are not mixed in.

## Full pipeline (CICIDS2017)

### 1. Dataset

Download the **CICIDS2017 MachineLearningCSV** release (already flow-featurized):

- Official: https://www.unb.ca/cic/datasets/ids-2017.html
- Kaggle mirrors of the same official CSVs also work

Put the `.csv` files in `data/raw/`. Expected names (any subset is enough to start; missing files are skipped with a warning):

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

### 2. Preprocess

```bash
python -m src.data_preprocessing.pipeline
```

Useful flags:

```bash
python -m src.data_preprocessing.pipeline \
    --num-clients 5 \
    --strategy non_iid \
    --alpha 0.5 \
    --test-size 0.2
```

| Flag | Default | Meaning |
|---|---|---|
| `--num-clients` | 5 | simulated organizations |
| `--strategy` | `non_iid` | `iid` even split, or Dirichlet-skewed `non_iid` |
| `--alpha` | 0.5 | lower = more label skew across clients |
| `--test-size` | 0.2 | held-out centralized test set |
| `--seed` | 42 | reproducibility |

This writes `data/processed/artifacts/` (scaler, label encoder, test set), `data/processed/clients/client_*.npz`, and related files.

### 3. Build graphs

```bash
python -m src.graph_construction.build_graphs
```

```bash
python -m src.graph_construction.build_graphs --k 10 --metric euclidean --batch-size 2000
```

Each flow is a node. Edges are k nearest neighbors in scaled feature space (IPs/ports were dropped as leakage). Large shards are split into `--batch-size` node graphs so they fit in memory.

Outputs go to `data/processed/graphs/` as `client_*_batch_*.npz` and `test_batch_*.npz`.

### 4. Train

Centralized upper bound (all client graphs together):

```bash
python -m src.training.train_centralized --loss ce --dropout 0.3 --lr 0.01
```

Federated FedAvg (IID vs non-IID is determined by how you preprocessed; `--strategy` is a label on the run):

```bash
python -m src.training.train_federated --strategy non_iid --rounds 10 --local-epochs 1
```

Per-class diagnostics and confusion matrices:

```bash
python -m src.training.diagnose
python -m src.training.diagnose --checkpoint results/full/centralized_ce_d03_lr001/centralized_model.pt --no-plot
```

Loss / hyperparameter ablation:

```bash
python -m src.training.run_ablation --losses ce weighted_ce focal
```

Checkpoints and JSON metrics land under `results/`. Paths, epochs, learning rate, dropout, and FL rounds can also be edited in `config.py`.

## Web application

```bash
python app/run_app.py
```

The UI talks to:

| Endpoint | Purpose |
|---|---|
| `GET /` | dashboard |
| `GET /api/health` | engine + device |
| `GET /api/metrics` | reported model and class metrics |
| `POST /api/predict/flow` | score one flow (70-D vector or slider fields) |
| `POST /api/predict/csv` | upload a numeric flow CSV (first 100 rows) |
| `GET /api/federated/clients` | simulated client mix |
| `POST /api/federated/simulate?round_number=1` | one FedAvg round demo |

## Configuration

Edit `config.py` for data paths, number of FL clients, Dirichlet α, k-NN `k`, GAT width/heads/dropout, loss (`ce`, `weighted_ce`, `focal`), epochs, and Flower rounds.

## License / data

CICIDS2017 is distributed by the Canadian Institute for Cybersecurity. This repo does not redistribute the original captures; obtain them from UNB and keep them in `data/raw/` locally.
