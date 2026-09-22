"""
FastAPI application exposing a single /score endpoint for the Home Credit
scorecard. Loads the frozen pipeline once at startup, not per request.
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI

sys.path.append(str(Path(__file__).parent.parent / "src"))
from scorecard_utils import score_client

from schemas import ClientFeatures, ScoreResponse

app = FastAPI(
    title="Home Credit Scorecard API",
    description="Scores a single credit applicant using a 17-variable WoE/IV logistic regression scorecard.",
    version="1.0",
)

PIPELINE_PATH = Path(__file__).parent.parent / "models" / "scorecard_pipeline.json"
with open(PIPELINE_PATH) as f:
    PIPELINE = json.load(f)


@app.get("/")
def root():
    return {"status": "ok", "model": PIPELINE["metadata"]["model_name"], "version": PIPELINE["metadata"]["version"]}


@app.post("/score", response_model=ScoreResponse)
def score(client: ClientFeatures):
    result = score_client(client.model_dump(), PIPELINE)
    return result