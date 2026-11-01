# FL-IDS: Closed-Loop Explainable Federated Learning for Cyber Threat Detection

Data preprocessing pipeline (Phase 1) for the project *"A Closed-Loop
Explainable Federated Learning Framework for Real-Time Cross-Enterprise
Cyber Threat Detection and Autonomous Incident Response."*

This phase takes raw CICIDS2017 traffic captures and produces clean,
scaled, per-organization data shards ready to feed into the Federated
Learning + GAT training stage (Phase 2).

## 1. Project layout

```
fl_ids_project/
├── config.py                     # all paths & settings live here
├── requirements.txt
├── data/
│   ├── raw/                      # <-- put the downloaded CICIDS2017 CSVs here
│   └── processed/
│       ├── artifacts/            # scaler.pkl, label_encoder.pkl, test_set.npz
│       ├── clients/               # client_0.npz ... client_N.npz (FL shards)
│       └── graphs/                # client_i_batch_j.npz, test_batch_j.npz (GAT-ready graphs)
├── src/
│   ├── data_preprocessing/
│   │   ├── load_data.py          # find + load raw CSVs
│   │   ├── clean_data.py         # drop leakage cols, fix inf/NaN, dedupe
│   │   ├── feature_engineering.py# label encode, split, scale
│   │   ├── fl_partition.py       # IID / non-IID split across FL clients
│   │   └── pipeline.py           # orchestrates the full run (entry point)
│   └── graph_construction/
│       ├── knn_graph.py          # k-NN feature-similarity graph builder
│       ├── build_graphs.py       # batches shards + builds/saves graphs (entry point)
│       └── pyg_loader.py         # loads saved graphs into PyTorch-Geometric Data objects
└── tests/
    └── make_dummy_data.py        # generates fake CICIDS2017-shaped CSVs for a smoke test
```

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(`requirements.txt` also includes packages for later phases — FL training,
GAT, SHAP, dashboard — so you only need to install once.)

## 3. Get the dataset

Download **CICIDS2017** (the `MachineLearningCSV` release — already
flow-featurized, no need to re-run CICFlowMeter):

- Official: https://www.unb.ca/cic/datasets/ids-2017.html
- Mirror: search "CICIDS2017" on Kaggle (multiple public mirrors of the
  same official CSVs)

Place the `.csv` files directly in `data/raw/`. You don't need every day's
file to run the pipeline — `config.CICIDS2017_FILES` lists all 8 expected
files, but `load_data.py` will just skip and warn about any that are
missing, so you can start with 1-2 days while waiting on the rest.

### No dataset yet? Smoke-test with synthetic data

```bash
python tests/make_dummy_data.py
python -m src.data_preprocessing.pipeline
```

This generates small fake CSVs that mimic CICIDS2017's shape and quirks
(leading-space columns, occasional inf/NaN, duplicate rows, class
imbalance) so you can verify the whole pipeline runs before the real
~500MB+ dataset is in place. **Delete `data/raw/*.csv` before using real
data** so the two don't mix.

## 4. Run the pipeline

```bash
python -m src.data_preprocessing.pipeline
```

Optional flags:

```bash
python -m src.data_preprocessing.pipeline \
    --num-clients 8 \
    --strategy non_iid \
    --alpha 0.5 \
    --test-size 0.2
```

| Flag            | Default (config.py) | Meaning                                                             |
|-----------------|----------------------|-----------------------------------------------------------------------|
| `--raw-dir`     | `data/raw`           | where to look for CICIDS2017 CSVs                                    |
| `--num-clients` | `5`                  | number of simulated organizations for FL                             |
| `--strategy`    | `non_iid`            | `iid` (even split) or `non_iid` (Dirichlet-skewed, more realistic)   |
| `--alpha`       | `0.5`                | Dirichlet concentration; lower = more skewed across clients          |
| `--test-size`   | `0.2`                | fraction held out as a centralized test set for evaluating the global model |
| `--seed`        | `42`                 | random seed, for reproducibility                                     |

## 5. What the pipeline does

1. **Load** — finds and concatenates whichever CICIDS2017 CSVs are present,
   normalizing whitespace-inconsistent column names and handling the
   dataset's occasional latin-1 encoding.
2. **Clean** — drops identifier/leakage columns (IPs, ports, timestamps,
   flow IDs), replaces the known `Flow Bytes/s` / `Flow Packets/s`
   infinities with NaN, drops rows with missing values, removes exact
   duplicates, drops zero-variance columns.
3. **Encode** — multi-class label encoding (keeps every attack type
   distinct — `BENIGN`, `DoS Hulk`, `PortScan`, `DDoS`, `FTP-Patator`,
   `Bot`, etc. — rather than collapsing to binary attack/benign), so the
   GAT can do multi-class detection and the severity-assessment stage can
   key off the specific attack type.
4. **Split** — stratified train/test split so rare attack classes appear
   in both.
5. **Scale** — `StandardScaler` fit on train only, applied to both splits
   (prevents test-set leakage into the scaler).
6. **Partition** — splits the training set across `--num-clients`
   simulated organizations. `non_iid` uses a Dirichlet distribution per
   class (standard in FL literature) so different "organizations" see
   different, realistic mixes of attacks — some see mostly port scans,
   others mostly DDoS, etc. `iid` gives an even random split, useful as a
   sanity-check baseline before testing under heterogeneity.

## 6. Outputs

- `data/processed/artifacts/scaler.pkl` — fitted `StandardScaler`, reuse
  for any new data at inference time.
- `data/processed/artifacts/label_encoder.pkl` — fitted `LabelEncoder`,
  maps encoded ints back to attack-type strings (needed for SHAP/severity
  stages later).
- `data/processed/artifacts/feature_names.txt` — ordered feature column
  names matching the columns in every `X` array.
- `data/processed/artifacts/test_set.npz` — centralized `X`, `y` test set
  for evaluating the global FL model after aggregation.
- `data/processed/clients/client_0.npz ... client_N.npz` — each
  organization's local `X`, `y` training shard, load with
  `np.load("client_0.npz")` → `data["X"]`, `data["y"]`.

## 7. Graph construction (Phase 2)

Turns each client's flat feature shards into graphs for the GAT.

```bash
python -m src.graph_construction.build_graphs
```

Optional flags:

```bash
python -m src.graph_construction.build_graphs \
    --k 10 \
    --metric euclidean \
    --batch-size 2000
```

Class-imbalance variants for ablations:

```bash
python -m src.graph_construction.build_graphs \
    --out-dir data/processed/graphs_oversampled \
    --oversample-minority

python -m src.graph_construction.build_graphs \
    --out-dir data/processed/graphs_stratified \
    --stratified-batches
```

| Flag             | Default (config.py) | Meaning                                                        |
|------------------|----------------------|------------------------------------------------------------------|
| `--k`            | `10`                 | neighbors per node in the similarity graph                       |
| `--metric`       | `euclidean`          | `euclidean` or `cosine` distance over scaled features            |
| `--batch-size`   | `2000`               | nodes per graph batch (keeps each graph GPU-sized)                |
| `--oversample-minority` | `False`       | random-duplicate minority training rows within each client before graphing |
| `--stratified-batches` | `False`        | interleave classes before chunking so rare labels are spread across graph batches |

**How it works:** each flow is a node. Edges connect a node to its k
nearest neighbors in scaled-feature space (not by IP/host — those were
already dropped as leakage-prone during preprocessing, and behavioral
similarity generalizes better across organizations and unseen attacks).
The graph is made undirected and deduplicated, with edge weights derived
from distance (closer = higher weight, in `(0, 1]`).

Because a full client's flows can be very large (hundreds of thousands of
rows on the real dataset), each client's data is first shuffled and split
into `--batch-size`-node chunks, and a separate graph is built per chunk.
This keeps every saved graph a fixed, GPU-friendly size and maps directly
onto a PyTorch-Geometric `DataLoader` — one `Data` object per batch file —
rather than requiring one enormous graph in memory.

**Outputs**, in `data/processed/graphs/`:
- `client_0_batch_0.npz`, `client_0_batch_1.npz`, ... — one file per batch
  per client, each containing `x` (node features), `y` (labels),
  `edge_index` (2×E, PyG convention), `edge_weight` (E,).
- `test_batch_0.npz`, ... — same, for the centralized test set.

**Loading for training** (needs `torch` + `torch-geometric`, not required
for graph construction itself):

```python
from src.graph_construction.pyg_loader import load_client_graphs
from torch_geometric.loader import DataLoader

graphs = load_client_graphs("data/processed/graphs", "client_0")
loader = DataLoader(graphs, batch_size=4, shuffle=True)
```

> **Note:** this repo's sandbox has no `torch`/`torch-geometric` installed
> (and no network access to add them), so graph construction was verified
> with structural checks — symmetry, no self-loops, no duplicate edges,
> valid node indices, valid edge weights, sane degree distribution — but
> an actual GAT forward pass hasn't been run yet. Install `torch` +
> `torch-geometric` in your own (GPU) environment for Phase 3.

## 8. Next steps (Phase 3+)

- Define the GAT model architecture (`GATConv` layers) and train it on the
  batched graphs above, using `x`, `edge_index`, `edge_weight`.
- Wire the per-client graph batches into Flower `NumPyClient` objects for
  the federated training loop (FedAvg across `client_0` ... `client_N`).
- SHAP explainability, severity scoring, and the response engine plug in
  after a trained global model exists.
