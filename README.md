# Home Credit Default Risk: From XGBoost to a Fully Auditable Scorecard

A probability-of-default (PD) modeling project built on the [Home Credit
Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle
dataset. Two parallel deliverables answer the same question with different
trade-offs: an **XGBoost model** for maximum predictive performance, and a
**logistic regression scorecard** (Weight of Evidence / Information Value)
for full, coefficient-by-coefficient interpretability.

The project is scoped to a specific slice of the data on purpose (see
[Scope and limitations](#scope-and-limitations)): `application_train.csv`
combined with aggregated `bureau.csv` history. This validates a complete
pipeline, end to end, before expanding to the remaining related tables.

## Headline results

| | XGBoost (baseline) | XGBoost (15 features) | Scorecard (17 variables) |
|---|---|---|---|
| AUC (5-fold mean) | 0.7588 | 0.7525 | — |
| AUC (holdout split) | 0.7614 | 0.7538 | 0.7406 |
| KS | — | — | 0.3621 |
| Gini | — | — | 0.4812 |

- Rejecting the riskiest 30% of applicants by model score cuts the
  portfolio's default rate roughly in half (8.07% -> 4.03%, XGBoost
  baseline; -> 4.31%, scorecard).
- The scorecard, with 17 auditable variables, retains **90-93%** of the
  XGBoost baseline's business gain at every rejection threshold tested,
  and every coefficient has a sign consistent with business logic, no
  exceptions.
- A 15-feature XGBoost variant performs within 0.6% of the 154-feature
  baseline, showing the predictive signal is concentrated in a small set
  of variables.

## The business problem

Home Credit extends consumer credit to clients who typically lack a
formal credit history with traditional financial institutions. Underwriting
decisions have to rely on alternative signals: income, employment,
housing, and whatever limited external bureau history is available. This
project builds a PD model on that basis, historical default rate 8.07%
across 307,511 applications.

## Approach

Two models, two different jobs:

- **XGBoost** measures the performance ceiling: how much predictive signal
  the data actually contains, including variable interactions a linear
  model would miss on its own.
- **Logistic regression on WoE-transformed variables** produces a
  **scorecard**: a table of points per characteristic that sums to a final
  score, auditable line by line. This is the format credit-risk
  regulation typically expects, and the one actually used in underwriting
  decisions in regulated markets.

Both are evaluated on the same held-out validation set, so their AUC,
KS, Gini and gain curves are directly comparable.

## Repository structure

| File | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory analysis: correlation with TARGET, sentinel-value treatment (`DAYS_EMPLOYED`), decile analysis, missing-value patterns |
| `notebooks/02_feature_engineering.ipynb` | Column-by-column aggregation of `bureau.csv` into 29 client-level features, merged with `application_train.csv` |
| `notebooks/03_modeling.ipynb` | XGBoost baseline and 15-feature variant, stratified k-fold validation, SHAP explainability, business gain curve |
| `notebooks/04_scorecard_woe.ipynb` | WoE/IV calculation, systematic variable selection (154 -> 17), Weight-of-Evidence scorecard, points table, KS/Gini validation |
| `notebooks/05_presentation.ipynb` | Supporting charts and tables used in the business presentation |
| `requirements.txt` | Pinned environment (Python 3.13, pandas, XGBoost, scikit-learn, statsmodels, SHAP) |

Notebooks are meant to be read in order; each one loads the processed
output of the previous step rather than recomputing it.

## How to reproduce

```bash
git clone <repo-url>
cd home-credit-default-risk
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Download `application_train.csv`, `bureau.csv`, and
`HomeCredit_columns_description.csv` from the
[Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data)
into `data/raw/` (not included in this repository, see the competition
rules). Run the notebooks in order, 01 through 04.

## Key technical decisions

- **Sentinel value handling**: `DAYS_EMPLOYED = 365243` (a placeholder for
  "not applicable", mostly pensioners) was flagged and converted to null
  rather than left in as a numeric outlier. Fixing it nearly doubled the
  variable's correlation with the target (0.045 -> 0.075).
- **Class imbalance**: addressed via `scale_pos_weight` (XGBoost) and
  `class_weight='balanced'` (logistic regression), given an 8.07% base
  rate.
- **Feature redundancy caught two different ways**: an exact duplicate
  (`DAYS_EMPLOYED` vs. `YEARS_EMPLOYED`, a linear transform of each other)
  was removed from the XGBoost feature set, improving AUC. A high but
  non-exact correlation (`AMT_GOODS_PRICE` vs. `AMT_CREDIT`, r = 0.99) was
  tested empirically and kept, since removal measurably hurt AUC, high
  correlation alone was not treated as sufficient grounds for removal.
- **Systematic WoE/IV variable selection**: all ~154 candidate variables
  were scanned for Information Value, deduplicated by concept (regex
  grouping of `_AVG`/`_MODE`/`_MEDI` variants and `DAYS_X`/`YEARS_X`
  pairs), and narrowed from 30 to 17 through correlation-based pruning,
  variance inflation factor checks, and a grouped stepwise diagnosis
  (Siddiqi's information-type grouping) that identified the external
  bureau score as the source of sign instability in weaker variables.
  The final pruning cost only 0.0011 AUC (0.7417 -> 0.7406) to eliminate
  every coefficient sign inconsistency.
- **No data leakage in the scorecard**: WoE tables and bin edges are
  computed strictly on the training split and applied, never recalculated,
  on validation, mirroring how a production scoring pipeline would behave
  with a new applicant.
- **Explainability without a black box**: SHAP for the XGBoost model
  explains aggregate behavior; the scorecard's point table explains any
  individual client's score directly, characteristic by characteristic.

## Scope and limitations

- This slice of the pipeline uses only `application_train.csv` and
  aggregated `bureau.csv`. The remaining related tables
  (`previous_application.csv`, monthly balance tables) are the main lever
  identified for closing the gap to the competition's top scores, and are
  a planned next step.
- One scorecard variable (`BUREAU_DAYS_CREDIT_UPDATE_MAX`) retains a
  coefficient sign inconsistent with its individual WoE pattern.
  Correlation and variance inflation factor checks did not isolate a
  single cause; documented as an open item rather than forced to fit.
- The scoring pipeline (frozen bins, WoE tables, coefficients, point
  scale) has not yet been packaged as a standalone production artifact.

## References

- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)
- Siddiqi, N. *Intelligent Credit Scoring: Building and Implementing Better
  Credit Risk Scorecards* (2nd edition, Wiley)

## Author

Vitor Krolikowski Massuchetti

[vitor_km@hotmail.com](mailto:vitor_km@hotmail.com) · [LinkedIn](https://www.linkedin.com/in/vitorkrolikowski/)
