"""Load test for /api/predict.

    locust -f locust/locustfile.py --host http://localhost:8080
"""
from __future__ import annotations

import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLES = sorted((Path(__file__).resolve().parents[1] / "data" / "samples").glob("*.wav"))
if not SAMPLES:
    raise SystemExit("no clips in data/samples/ - run the notebook first")
CLIPS = [(p.name, p.read_bytes()) for p in SAMPLES]


class PumpTechnician(HttpUser):
    # each simulated user sends a 10 s clip about once a second
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(CLIPS)
        with self.client.post("/api/predict", files={"file": (name, data, "audio/wav")},
                              name="POST /api/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.json().get("label") not in ("normal", "abnormal"):
                resp.failure("malformed prediction")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
