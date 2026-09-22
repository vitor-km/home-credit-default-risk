# Home Credit Scorecard API

A minimal FastAPI service that scores a client against the frozen 17-variable
WoE/IV scorecard built in this repository (see `notebooks/04_scorecard_woe.ipynb`
and `notebooks/06_model_packaging.ipynb`).

The service loads a self-contained artifact, `models/scorecard_pipeline.json`,
which holds everything needed to score a client: bin edges, WoE tables,
logistic regression coefficients, and the Factor/Offset point scale. No
notebook or training-time state is required at runtime.

## What it returns

`POST /score` accepts the 17 raw client features and returns:

- `score`: the client's scorecard points (Factor/Offset scale, centered at 500)
- `probability_default`: model-estimated probability of default
- `breakdown`: per-variable WoE and points, for full auditability
- `unmapped_variables`: any categorical fields whose value was not seen during
  training
- `review_flag`: `true` when 3 or more variables were unmapped for this client

### Design note: unmapped categorical values

The scorecard never blocks a scoring request because a single categorical
value is unseen (for example, an `ORGANIZATION_TYPE_BINNED` value outside the
training vocabulary). Blocking an entire client over one field out of 17 is
disproportionate. Instead, an unmapped value is scored with WoE = 0
(statistically neutral, no implicit bias), the variable is listed in
`unmapped_variables`, and `review_flag` is set once 3 or more variables are
unmapped for the same client. This threshold is a reasonable starting point,
not statistically tuned, refining it is left for the future PSI-based
monitoring step.

## Running locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is then available at `http://127.0.0.1:8000`, with interactive docs
at `http://127.0.0.1:8000/docs`.

## Running with Docker

Build and run from the repository root (the `Dockerfile` copies `api/`,
`src/`, and `models/scorecard_pipeline.json`):

```bash
docker build -t home-credit-scorecard .
docker run -p 8000:8000 home-credit-scorecard
```

The API is then available at `http://127.0.0.1:8000`, identical behavior to
the local run above. This was validated end to end: the same client
(`SK_ID_CURR` 251294) produces the identical score (670.81) and
`probability_default` (0.0357) whether scored directly from the notebook,
from a local `uvicorn` process, or from the Docker container, confirming the
pipeline is deterministic and environment-independent.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_1": 0.75,
    "EXT_SOURCE_2": 0.68,
    "EXT_SOURCE_3": 0.71,
    "NAME_EDUCATION_TYPE_BINNED": "Higher education",
    "AMT_CREDIT": 450000,
    "CODE_GENDER": "F",
    "YEARS_EMPLOYED": 15.2,
    "ORGANIZATION_TYPE_BINNED": "Business Entity Type 3",
    "FLOORSMAX_AVG": 0.25,
    "BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT": 1,
    "BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN": 5000,
    "OCCUPATION_TYPE_BINNED": "Core staff",
    "REGION_POPULATION_RELATIVE": 0.02,
    "DAYS_ID_PUBLISH": -3000,
    "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": 0,
    "DAYS_LAST_PHONE_CHANGE": -500,
    "BUREAU_DAYS_CREDIT_MEAN": -900
  }'
```

### Example response

```json
{
  "score": 670.81,
  "probability_default": 0.0357,
  "breakdown": {
    "EXT_SOURCE_1": {"raw_value": 0.75, "woe": 0.91, "points": 55.85},
    "...": "..."
  },
  "unmapped_variables": [],
  "review_flag": false
}
```

## Files

- `main.py`: FastAPI app, loads `models/scorecard_pipeline.json` once at
  startup, exposes `GET /` (health check) and `POST /score`.
- `schemas.py`: Pydantic request/response models. Categorical fields are open
  `str`/`Optional[str]`, not `Literal`, by design (see the note above on
  unmapped values).
- `requirements.txt`: `fastapi`, `uvicorn[standard]`, `pandas`, `numpy`,
  `pydantic`.

The scoring logic itself (`score_client`) lives in `src/scorecard_utils.py`,
shared with the notebooks so the API and the notebooks never diverge.