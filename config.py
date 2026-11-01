"""
Central configuration for the data preprocessing pipeline.
Edit paths / settings here rather than inside the pipeline modules.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(PROCESSED_DATA_DIR, "artifacts")  # scaler.pkl, label_encoder.pkl
CLIENTS_DIR = os.path.join(PROCESSED_DATA_DIR, "clients")      # per-organization FL shards

# ---------------------------------------------------------------------------
# CICIDS2017 raw files (MachineLearningCSV variant, from UNB / Kaggle).
# Only files that exist in RAW_DATA_DIR at runtime are actually loaded, so
# you don't have to download all of them to test the pipeline.
# ---------------------------------------------------------------------------
CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# Column in the raw CSVs that holds the class label (has a leading space
# in the official release: " Label"). load_data.py normalizes this away.
LABEL_COLUMN = "Label"

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
DROP_DUPLICATES = True
# Columns that are constant / identifiers in CICIDS2017 and carry no
# predictive signal for a per-flow classifier.
DROP_COLUMNS = ["Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
                 "Source Port", "Src Port", "Timestamp"]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Federated partitioning
# ---------------------------------------------------------------------------
NUM_CLIENTS = 5              # simulated organizations
PARTITION_STRATEGY = "non_iid"  # "iid" or "non_iid"
DIRICHLET_ALPHA = 0.5        # lower = more non-IID (skewed) across clients

# ---------------------------------------------------------------------------
# Graph construction (flows -> graph, for the GAT stage)
# ---------------------------------------------------------------------------
GRAPHS_DIR = os.path.join(PROCESSED_DATA_DIR, "graphs")
KNN_K = 10                   # neighbors per node when building the similarity graph
KNN_METRIC = "euclidean"     # "euclidean" or "cosine"
GRAPH_BATCH_SIZE = 2000      # nodes per graph "batch" -- keeps each graph GPU-sized
                               # and maps 1:1 onto PyG DataLoader batches later
GRAPH_SEED = 42
OVERSAMPLE_MINORITY = False   # random oversampling within each train shard before graphing
STRATIFIED_GRAPH_BATCHES = False  # spread rare classes across graph batches when chunking

# ---------------------------------------------------------------------------
# GAT model + training (Phase 3)
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
HIDDEN_CHANNELS = 128
NUM_HEADS = 4
DROPOUT = 0.1                 # optimized dropout to prevent feature crippling
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
EPOCHS = 15                   # optimized epoch count
TRAIN_BATCH_SIZE = 4          # PyG DataLoader: number of graph batches per step

# Loss: "ce" (baseline), "weighted_ce", "focal"
LOSS_TYPE = "weighted_ce"
FOCAL_GAMMA = 2.0

# Federated learning
FL_ROUNDS = 10
LOCAL_EPOCHS = 1
