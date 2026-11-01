"""
Unit tests for FastAPI REST API endpoints and GAT Inference Engine.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.backend.main import app
from app.backend.inference_engine import GATInferenceEngine


class TestGATWebAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.engine = GATInferenceEngine(in_channels=70, num_classes=13, k=10)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["num_classes"], 13)

    def test_metrics_endpoint(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("parameters", data)
        self.assertIn("class_wise_metrics", data)
        self.assertEqual(len(data["class_wise_metrics"]), 13)

    def test_predict_flow_endpoint(self):
        payload = {
            "duration": 1240.0,
            "packet_count": 12,
            "packet_len_mean": 142.0,
            "iat_mean": 45.2
        }
        response = self.client.post("/api/predict/flow", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        pred = data["prediction"]
        self.assertIn("predicted_class", pred)
        self.assertIn("confidence", pred)
        self.assertIn("is_malicious", pred)

    def test_inference_engine_batch(self):
        dummy_features = np.random.randn(5, 70).astype(np.float32)
        results, edges, weights = self.engine.predict_flow_batch(dummy_features)
        self.assertEqual(len(results), 5)
        self.assertGreaterEqual(len(edges[0]), 5)

    def test_federated_clients_endpoint(self):
        response = self.client.get("/api/federated/clients")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["num_clients"], 5)
        self.assertEqual(len(data["clients"]), 5)

    def test_simulate_fedavg_endpoint(self):
        response = self.client.post("/api/federated/simulate?round_number=3")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["round"], 3)
        self.assertIn("global_accuracy", data)

    def test_predict_csv_endpoint(self):
        # Create a sample dummy CSV file in memory
        csv_content = "duration,packets,bytes,iat\n1240,12,142,45\n950000,480,1400,2\n120,2,40,1\n"
        files = {"file": ("test_flows.csv", csv_content, "text/csv")}
        response = self.client.post("/api/predict/csv", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test_flows.csv")
        self.assertEqual(data["total_flows_audited"], 3)
        self.assertIn("malicious_percentage", data)



if __name__ == "__main__":
    unittest.main()
