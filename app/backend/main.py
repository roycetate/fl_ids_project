"""
FastAPI Backend Application for Federated GAT Intrusion Detection System.
Provides REST API endpoints for model predictions, CSV threat auditing,
performance metrics retrieval, and federated learning simulation.
"""

import sys
import os
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.backend.inference_engine import GATInferenceEngine, ATTACK_CLASSES

app = FastAPI(
    title="Federated GAT-IDS REST API",
    description="Cross-Enterprise Network Intrusion Detection API powered by Graph Attention Networks",
    version="2.0.0"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine instance
engine = GATInferenceEngine(in_channels=70, num_classes=13, k=10)


class FlowInput(BaseModel):
    duration: float = 1240.0
    packet_count: int = 12
    packet_len_mean: float = 142.0
    iat_mean: float = 45.2
    features_70d: Optional[List[float]] = None


class BatchFlowInput(BaseModel):
    flows: List[FlowInput]


# Model Metrics Data (Table I, II, III)
MODEL_PARAMETERS = {
    "dataset": "CICIDS2017",
    "train_test_split": "80/20",
    "num_clients": 5,
    "input_features": 70,
    "classes": 13,
    "graph_type": "Behavioral k-NN (k=10)",
    "distance_metric": "Euclidean",
    "gat_layers": 2,
    "hidden_channels": 128,
    "attention_heads": "4 (Layer 1), 2 (Layer 2)",
    "residual_skip": "Linear (70 -> 256)",
    "dropout": 0.1,
    "activation": "GELU",
    "optimizer": "AdamW (lr=0.005, wd=1e-4)",
    "loss_function": "Sublinear Power-Scaled Cross-Entropy",
    "local_epochs": 2,
    "fedavg_rounds": 10,
    "non_iid_dirichlet_alpha": 0.5
}

TABLE_II_CLASS_METRICS = [
    {"class": "BENIGN", "prec": 0.998, "rec": 0.995, "f1_cent": 0.996, "f1_iid": 0.994, "f1_noniid": 0.992, "status": "Optimal"},
    {"class": "DoS Hulk", "prec": 0.991, "rec": 0.988, "f1_cent": 0.989, "f1_iid": 0.985, "f1_noniid": 0.981, "status": "Optimal"},
    {"class": "PortScan", "prec": 0.985, "rec": 0.979, "f1_cent": 0.982, "f1_iid": 0.978, "f1_noniid": 0.971, "status": "Optimal"},
    {"class": "DDoS", "prec": 0.982, "rec": 0.975, "f1_cent": 0.978, "f1_iid": 0.971, "f1_noniid": 0.965, "status": "Optimal"},
    {"class": "DoS GoldenEye", "prec": 0.932, "rec": 0.921, "f1_cent": 0.912, "f1_iid": 0.815, "f1_noniid": 0.848, "status": "Strong"},
    {"class": "FTP-Patator", "prec": 0.905, "rec": 0.892, "f1_cent": 0.884, "f1_iid": 0.762, "f1_noniid": 0.805, "status": "Strong"},
    {"class": "SSH-Patator", "prec": 0.874, "rec": 0.860, "f1_cent": 0.852, "f1_iid": 0.720, "f1_noniid": 0.772, "status": "Strong"},
    {"class": "DoS slowloris", "prec": 0.821, "rec": 0.812, "f1_cent": 0.795, "f1_iid": 0.612, "f1_noniid": 0.665, "status": "Strong"},
    {"class": "DoS Slowhttptest", "prec": 0.765, "rec": 0.748, "f1_cent": 0.721, "f1_iid": 0.495, "f1_noniid": 0.552, "status": "Moderate"},
    {"class": "Bot", "prec": 0.542, "rec": 0.654, "f1_cent": 0.465, "f1_iid": 0.182, "f1_noniid": 0.282, "status": "Minority"},
    {"class": "Web Attack – Brute Force", "prec": 0.385, "rec": 0.521, "f1_cent": 0.284, "f1_iid": 0.050, "f1_noniid": 0.145, "status": "Minority"},
    {"class": "Web Attack – XSS", "prec": 0.224, "rec": 0.428, "f1_cent": 0.142, "f1_iid": 0.015, "f1_noniid": 0.080, "status": "Rare"},
    {"class": "Infiltration / Sql Inject", "prec": 0.0795, "rec": 0.2487, "f1_cent": 0.0727, "f1_iid": 0.0065, "f1_noniid": 0.0215, "status": "Extreme Rare"}
]

TABLE_III_LITERATURE = [
    {"model": "CICIDS2017 Study", "accuracy": 96.90, "f1_score": 95.80, "type": "Baseline"},
    {"model": "EdgeDetect", "accuracy": 98.00, "f1_score": 97.90, "type": "Baseline"},
    {"model": "Proposed Centralized GAT", "accuracy": 97.34, "f1_score": 97.47, "type": "Proposed"},
    {"model": "Proposed Federated IID", "accuracy": 95.93, "f1_score": 96.13, "type": "Proposed"},
    {"model": "Proposed Federated Non-IID (alpha=0.5)", "accuracy": 96.47, "f1_score": 96.60, "type": "Proposed"}
]


@app.get("/api/health")
def get_health():
    """Healthcheck endpoint verifying model & hardware engine."""
    return {
        "status": "online",
        "device": str(engine.device),
        "in_channels": engine.in_channels,
        "num_classes": engine.num_classes,
        "model_architecture": "GATClassifier (Residual Skip + BatchNorm + GELU)"
    }


@app.get("/api/metrics")
def get_metrics():
    """Returns Table I parameters, Table II class metrics, and Table III literature benchmarks."""
    return {
        "parameters": MODEL_PARAMETERS,
        "class_wise_metrics": TABLE_II_CLASS_METRICS,
        "macro_averages": {
            "precision": 0.7295,
            "recall": 0.7709,
            "f1_centralized": 0.6979,
            "f1_iid": 0.5835,
            "f1_noniid": 0.6215
        },
        "literature_comparison": TABLE_III_LITERATURE
    }


@app.post("/api/predict/flow")
def predict_flow(data: FlowInput):
    """Predicts attack class for a single network flow input."""
    if data.features_70d is not None and len(data.features_70d) == 70:
        feat_vec = np.array(data.features_70d, dtype=np.float32).reshape(1, 70)
    else:
        # Construct synthetic normalized 70-D vector from key sliders
        feat_vec = np.zeros((1, 70), dtype=np.float32)
        feat_vec[0, 0] = (data.duration - 1000.0) / 50000.0
        feat_vec[0, 1] = (data.packet_count - 10.0) / 100.0
        feat_vec[0, 2] = (data.packet_len_mean - 100.0) / 500.0
        feat_vec[0, 3] = (data.iat_mean - 20.0) / 100.0

    predictions, edge_index, edge_weight = engine.predict_flow_batch(feat_vec)
    return {
        "prediction": predictions[0],
        "graph_topology": {
            "num_nodes": 1,
            "num_edges": len(edge_weight),
            "edge_weights": edge_weight
        }
    }


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Uploads CSV of network flows, preprocesses features, builds k-NN graph, and returns threat audit."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV format")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)
        if numeric_df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no numeric flow features")

        # Standardize features
        data_matrix = numeric_df.values.astype(np.float32)
        mean = np.mean(data_matrix, axis=0, keepdims=True)
        std = np.std(data_matrix, axis=0, keepdims=True) + 1e-6
        norm_matrix = (data_matrix - mean) / std

        # Cap batch size to max 100 flows for rapid inference
        if norm_matrix.shape[0] > 100:
            norm_matrix = norm_matrix[:100]

        predictions, edge_index, edge_weight = engine.predict_flow_batch(norm_matrix)

        # Compute summary audit metrics
        class_counts = {}
        malicious_count = 0

        for p in predictions:
            cls_name = p["predicted_class"]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            if p["is_malicious"]:
                malicious_count += 1

        total_flows = len(predictions)
        benign_count = total_flows - malicious_count

        return {
            "filename": file.filename,
            "total_flows_audited": total_flows,
            "benign_flows": benign_count,
            "malicious_flows": malicious_count,
            "malicious_percentage": round((malicious_count / total_flows) * 100, 2),
            "class_distribution": class_counts,
            "flow_predictions": predictions[:20]  # Return top 20 sample details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@app.get("/api/federated/clients")
def get_federated_clients():
    """Returns client distributions across 5 simulated enterprise clients."""
    return {
        "num_clients": 5,
        "dirichlet_alpha": 0.5,
        "clients": [
            {"id": "Client_1", "name": "Org A (Enterprise HQ)", "samples": 257918, "primary_attack": "BENIGN (75%)"},
            {"id": "Client_2", "name": "Org B (Data Center)", "samples": 257918, "primary_attack": "DoS Hulk (50%)"},
            {"id": "Client_3", "name": "Org C (Branch Office)", "samples": 257918, "primary_attack": "PortScan (55%)"},
            {"id": "Client_4", "name": "Org D (Cloud Subnet)", "samples": 257918, "primary_attack": "DDoS (60%)"},
            {"id": "Client_5", "name": "Org E (DMZ Gateway)", "samples": 257918, "primary_attack": "SSH/FTP-Patator (25%)"}
        ]
    }


@app.post("/api/federated/simulate")
def simulate_fedavg_round(round_number: int = Query(1, ge=1, le=10)):
    """Simulates 1 round of FedAvg client aggregation."""
    # Compute simulated convergence metrics
    base_acc = 90.0 + (round_number * 0.647)
    base_loss = 0.45 - (round_number * 0.038)
    base_wf1 = 90.5 + (round_number * 0.61)

    return {
        "round": round_number,
        "total_rounds": 10,
        "participating_clients": 5,
        "aggregation_algorithm": "Weighted FedAvg",
        "global_accuracy": round(min(base_acc, 96.47), 2),
        "global_weighted_f1": round(min(base_wf1, 96.60), 2),
        "global_loss": round(max(base_loss, 0.07), 4),
        "status": "FedAvg aggregation complete for round " + str(round_number)
    }


# Mount Frontend UI static route
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        html_file = FRONTEND_DIR / "index.html"
        if html_file.exists():
            return html_file.read_text(encoding="utf-8")
        return "<h1>Frontend index.html not found</h1>"
