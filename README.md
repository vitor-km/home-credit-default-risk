# Home Credit Default Risk: From XGBoost to a Deployed, Monitored Scorecard

A probability-of-default (PD) modeling project built on the [Home Credit
Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle
dataset, taken from exploratory analysis to a scoring API. Two models
answer the same question with different trade-offs: an **XGBoost model**
that measures the performance ceiling, and a **logistic regression
scorecard** (Weight of Evidence / Information Value) that is auditable
coefficient by coefficient. The scorecard is then frozen into a portable
artifact, served through a FastAPI + Docker endpoint, checked for
population stability (PSI), and tracked with MLflow.

The data scope is deliberately narrow (see
[Scope and limitations](#scope-and-limitations)): `application_train.csv`
combined with aggregated `bureau.csv` history. The goal was to validate a
complete pipeline, end to end, before expanding to the remaining tables.

## Headline results

| | XGBoost (baseline, 154 features) | XGBoost (15 features) | Scorecard (17 variables) |
|---|---|---|---|
| AUC, 5-fold mean | 0.7588 | 0.7525 | n/a |
| AUC, holdout split | 0.7614 | 0.7538 | 0.7406 |
| KS, holdout | n/a | n/a | 0.3621 |
| Gini, holdout | n/a | n/a | 0.4812 |

Holdout figures for all three models come from the same stratified 80/20
split (`random_state=42`).

- Rejecting the riskiest 30% of applicants cuts the default rate among
  approved clients from 8.07% to 4.03% (XGBoost baseline, out-of-fold
  predictions over all 307,511 clients) and to 4.31% (scorecard, on the
  61,503 holdout clients).
- The scorecard retains roughly **90-93%** of the XGBoost baseline's
  default-rate reduction at the 10%, 20% and 30% rejection thresholds,
  and every one of its 17 coefficients has a sign consistent with
  business logic. The two gain tables are built on different bases (see
  above), so the ratio is indicative rather than exact.
- The 15-feature XGBoost variant loses only 0.0063 AUC against the
  154-feature baseline (5-fold means), showing that the predictive signal
  is concentrated in a small set of variables.
- On `application_test.csv`, used as a proxy for a new population, the
  final score and all 17 characteristics are stable (PSI below 0.10; score
  PSI 0.0007).

## The business problem

Home Credit extends consumer credit to clients who often lack a formal
credit history with traditional financial institutions. Underwriting has
to rely on alternative signals: income, employment, housing, and whatever
limited external bureau history is available. This project builds a PD
model on that basis, with a historical default rate of 8.07% across
307,511 applications.

## Approach

Two models, two different jobs:

- **XGBoost** measures how much predictive signal the data contains,
  including variable interactions a linear model would miss.
- **Logistic regression on WoE-transformed variables** produces a
  **scorecard**: a table of points per characteristic that sums to a final
  score, auditable line by line. This is the format credit-risk regulation
  typically expects in underwriting.

The scorecard then goes through the steps a model needs before and after
it is used in production:

1. **Packaging**: bins, WoE values, coefficients and the point scale are
   frozen into `models/scorecard_pipeline.json`. A rebuild from scratch in
   a separate notebook reproduces the original model at reported
   precision (AUC, KS, and the score of a known client).
2. **Serving**: a FastAPI endpoint loads the JSON once and scores a client
   with a per-characteristic point breakdown. The same client returns the
   same score in the notebook, in a local `uvicorn` server and inside the
   Docker container.
3. **Monitoring**: PSI on the final score and on each characteristic,
   using the frozen training bins.
4. **Tracking**: MLflow records parameters, metrics and artifacts for the
   final scorecard and XGBoost runs, logged in the same execution that
   produces each model.

## Repository structure

| Path | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory analysis: correlation with TARGET, sentinel-value treatment (`DAYS_EMPLOYED`), decile analysis, missing-value patterns |
| `notebooks/02_feature_engineering.ipynb` | Column-by-column aggregation of `bureau.csv` into 29 client-level features, merged with `application_train.csv` |
| `notebooks/03_modeling.ipynb` | XGBoost baseline and 15-feature variant, stratified k-fold validation, SHAP explainability, business gain curve, MLflow run |
| `notebooks/04_scorecard_woe.ipynb` | WoE/IV calculation, systematic variable selection (154 to 17), scorecard, points table, KS/Gini validation |
| `notebooks/05_presentation.ipynb` | Supporting charts and tables used in the business presentation |
| `notebooks/06_model_packaging.ipynb` | Rebuilds the scorecard from scratch, verifies it reproduces notebook 04, writes the frozen JSON pipeline, MLflow run |
| `notebooks/07_psi_monitoring.ipynb` | Population Stability Index on the score and the 17 characteristics, train vs. validation and train vs. `application_test.csv` |
| `src/scorecard_utils.py` | WoE/IV and binning functions, and `score_client`, the production scoring function |
| `models/scorecard_pipeline.json` | Frozen scorecard: bin edges, WoE tables, category mappings, coefficients, point scale |
| `api/` | FastAPI application, Pydantic schemas and API documentation ([api/README.md](api/README.md)) |
| `dockerfile` | Container image for the scoring API |
| `reports/figures/` | Figures, including PSI by characteristic |
| `requirements.txt` | Pinned environment (Python 3.13, pandas, XGBoost, scikit-learn, statsmodels, SHAP, MLflow) |

Notebooks are meant to be read in order; each one loads the processed
output of an earlier step rather than recomputing it.

## How to reproduce

```bash
git clone <repo-url>
cd home-credit-default-risk
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download `application_train.csv`, `application_test.csv`, `bureau.csv` and
`HomeCredit_columns_description.csv` from the
[Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data)
into `data/raw/` (not included in this repository, see the competition
rules). Run notebooks 01 through 04, then 06 and 07. Notebook 02 writes the
processed files that 03, 04, 06 and 07 read.

**Scoring API.** Build and run the container from the repository root, then
open `http://localhost:8000/docs`:

```bash
docker build -t home-credit-scorecard .
docker run -p 8000:8000 home-credit-scorecard
```

See [api/README.md](api/README.md) for the request format and a real
example response.

**Experiment tracking.** MLflow runs are stored locally and are not
committed. Running notebooks 03 and 06 creates them; to browse them:

```bash
cd notebooks
mlflow ui
```

## Key technical decisions

- **Sentinel value handling**: `DAYS_EMPLOYED = 365243` (a placeholder for
  "not applicable", mostly pensioners) was flagged and converted to null
  rather than left in as a numeric outlier. Fixing it nearly doubled the
  variable's correlation with the target (0.045 to 0.075).
- **Class imbalance**: addressed via `scale_pos_weight` (XGBoost) and
  `class_weight='balanced'` (logistic regression), given an 8.07% base
  rate.
- **Feature redundancy caught two different ways**: an exact duplicate
  (`DAYS_EMPLOYED` vs. `YEARS_EMPLOYED`, a linear transform of each other)
  was removed from the XGBoost feature set, improving AUC. A high but
  non-exact correlation (`AMT_GOODS_PRICE` vs. `AMT_CREDIT`, r = 0.99) was
  tested empirically and kept, since removing it measurably hurt AUC.
- **Systematic WoE/IV variable selection**: all ~154 candidate variables
  were scanned for Information Value, deduplicated by concept (regex
  grouping of `_AVG`/`_MODE`/`_MEDI` variants and `DAYS_X`/`YEARS_X`
  pairs), and narrowed from 30 to 17 through correlation-based pruning,
  variance inflation factor checks, and a grouped diagnosis by information
  type (Siddiqi). The grouping showed the external bureau scores reversing
  the sign of `AGE_YEARS`; the final pruning cost only 0.0011 AUC (0.7417
  to 0.7406) and removed every coefficient sign inconsistency.
- **No data leakage in the scorecard**: WoE tables and bin edges are
  computed strictly on the training split and applied, never recalculated,
  on validation, the way a production pipeline treats a new applicant.
- **Reproducibility checked per variable and per client, not only by
  AUC**: rebuilding the pipeline exposed three silent divergences from the
  original model, one of which moved AUC by only 0.0001. They were fixed
  and the artifact was versioned (v1.1) rather than overwritten.
- **Graceful handling of unseen categories**: a category never seen in
  training receives a neutral WoE of 0 and is reported in the API response
  for monitoring, instead of blocking the whole application.
- **PSI on frozen bins**: the score is binned on training-set deciles with
  open outer bins, and characteristics use the model's own bins. Bins
  recalculated on each new population would show near-zero PSI by
  construction.
- **Explainability without a black box**: SHAP explains the XGBoost model;
  the scorecard's point table explains any individual client's score
  directly, characteristic by characteristic.

## Stability monitoring (PSI)

| Comparison | Score PSI | Highest characteristic PSI |
|---|---|---|
| Train vs. validation (noise floor) | 0.0003 | 0.0004 |
| Train vs. `application_test.csv` | 0.0007 | 0.0843 (`EXT_SOURCE_1`) |

- The largest shift, `EXT_SOURCE_1`, is a change in coverage (56.4% missing
  in train vs. 42.1% in test), not in the distribution of observed values.
- Shifts in different characteristics offset each other in the score: the
  mean score moves only 1.25 points, and a per-characteristic decomposition
  reproduces that difference exactly.
- `DAYS_ID_PUBLISH` and `DAYS_LAST_PHONE_CHANGE` show bin-level shifts far
  beyond sampling noise whose cause could not be established with this
  dataset. Their effect on the score is small, and they are documented as
  open findings.

## Scope and limitations

- This slice of the pipeline uses only `application_train.csv` and
  aggregated `bureau.csv`. The remaining related tables
  (`previous_application.csv`, monthly balance tables) are the main lever
  identified for closing the gap to the competition's top scores.
- **Predicted probabilities are not calibrated.** Training with
  `class_weight='balanced'` preserves the ranking (AUC, KS, score) but
  shifts absolute probabilities upward relative to the 8.07% observed
  default rate. The API's `probability_default` should be read as a
  ranking signal, not as a calibrated PD.
- `application_test.csv` is a proxy for a new population, not a later time
  period: the dataset has no calendar dates and the test set has no
  target, so PSI measures population shift, not performance decay.
- The frozen JSON contains everything needed to score the 17 final
  variables, but not the preprocessing that produces them (sentinel
  handling, year conversions, bureau aggregation). That logic lives in
  notebooks 02 and 06, and the API expects already-prepared inputs.
- Three of the variables removed in the final pruning had sign
  inversions with no isolable statistical cause (correlation and VIF
  checks ruled out multicollinearity). They were removed and documented
  rather than explained away.
- XGBoost hyperparameters were fixed, not tuned. The 15-feature k-fold
  AUC is slightly optimistic, because the features were selected using
  part of the data that later appears in the validation folds; the
  holdout AUC is the clean comparison.

## References

- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
- Siddiqi, N. *Intelligent Credit Scoring: Building and Implementing Better
  Credit Risk Scorecards* (2nd edition, Wiley)

## Author

Vitor Krolikowski Massuchetti

[vitor_km@hotmail.com](mailto:vitor_km@hotmail.com) · [LinkedIn](https://www.linkedin.com/in/vitorkrolikowski/)
