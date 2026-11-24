# Bank Account Fraud: time-split modelling, rule distillation and fairness

[中文说明](README.zh-CN.md)

An end-to-end, reproducible analysis of all **1,000,000 applications** in the Base variant of Feedzai's [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) dataset (NeurIPS 2022). The data is synthetic, generated from anonymised real bank account-opening applications; fraud is 1.10% of applications and the distribution drifts month to month.

The question is the one a bank fraud team actually faces: **under a false-positive budget, which model catches the most fraud, what does it cost to turn it into rules a rule engine can run, and who bears the false positives?**

**Full report:** [`outputs/BAF_分析报告.html`](outputs/BAF_分析报告.html) — self-contained HTML (Chinese), figures embedded. Download it and open it in a browser, or publish it with GitHub Pages.

## Key results (test month, scored once after everything was frozen)

| Model | Recall at frozen threshold | FPR | Precision | Average precision | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression (class-weighted) | 45.1% | 3.42% | 16.5% | 0.185 | 0.887 |
| **LightGBM** (selected) | **53.3%** | **4.38%** | 15.4% | **0.210** | **0.893** |
| MLP (class-weighted) | 54.6% | 5.31% | 13.3% | 0.189 | 0.886 |
| Isolation Forest (unsupervised check) | 8.8% | 6.28% | 2.1% | 0.019 | 0.574 |

Test month: 96,843 applications, 1,428 frauds, so a random model's average precision is about 0.015. The MLP's higher recall comes from overshooting the 5% FPR limit on the test month; LightGBM won on the validation month, which is the only place selection happened.

- **Rules cost about 20 percentage points of recall.** A 5-leaf regression tree distilled from LightGBM's log-odds, given the same review budget (5,463 alerts, 5.64% of applications), catches 34.4% of fraud versus 54.9% for LightGBM: 293 fewer frauds caught.
- **Age disparity.** At the same global threshold, applicants aged 50+ have **2.9×** the false-positive rate of younger applicants (gap 6.6 pp). Dropping the age feature narrows the gap but does not close it, and isn't free in recall.
- **Robustness.** Over 4 expanding-window backtests (added after seeing the original result, so exploratory), baseline recall ranged 45.3%–56.8% at FPR 3.2%–6.1%, and 49.9%–53.3% at a fixed 5% review budget.
- **Unsupervised doesn't work here.** Isolation Forest is close to random: this fraud is not a simple outlier problem.

## Method

| Stage | Raw `month` | Used for |
| --- | --- | --- |
| Train | 0–5 | fitting models and every preprocessing step |
| Validation | 6 | hyper-parameters, early stopping, threshold, rule-tree configuration |
| Test | 7 | scored once, after a frozen manifest was written |

1. **Data audit.** The data has no explicit nulls, but the datasheet documents `-1` as missing in five fields and negative `intended_balcon_amount` as missing — about 1.7 million cells. They are converted to missing with indicator columns; legitimate negatives (`velocity_6h`, `credit_risk_score`) are kept.
2. **No leakage by time.** Time-based split; `month` is excluded as an input; imputation, scaling and category vocabularies are fit on training months only.
3. **Model selection under a business constraint.** Models are ranked by **recall at FPR ≤ 5% on the validation month** (AP as tie-breaker), not by accuracy or AUC. The threshold is chosen on validation and frozen; test FPR is measured, not forced to 5%.
4. **Explain, then distil.** LightGBM's exact TreeSHAP (built in, no `shap` dependency) explains the model; a depth-3, ≤5-leaf regression tree is fit to the teacher's log-odds and compared at an **equal review budget**, with label-independent tie-breaking.
5. **Fairness and drift.** FPR by age group with Wilson 95% intervals and a sensitivity check on the age cut-off; monthly PSI and a `velocity_*` drift ablation.
6. **Exploratory extensions**, clearly separated from the pre-registered protocol in [`protocol.json`](protocol.json): WOE/IV and PSI feature screening, derived features, rule-complexity sweep, SMOTE vs class weights, random vs time split, rolling backtests.

Class-weighted scores are **not calibrated probabilities**; SHAP values and leaf scores are on the log-odds scale.

## Reproduce

Python 3.12.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py              # downloads Base.csv from Kaggle if missing, runs every stage, verifies, builds the report
python run_analysis.py --retrain    # retrain from scratch
python -m pytest -q
```

Stages can also be run one by one: `src.eda → src.train → src.evaluate → src.explain → src.extensions → src.rolling_backtest → src.coursework → src.feature_study → src.report --require-complete → src.verify`. `src.verify` checks every saved number against the protocol and the frozen manifest.

## Repository layout

```
src/                 analysis stages (eda, train, evaluate, explain, extensions, rolling_backtest, feature_study, report, verify, …)
outputs/             report, JSON results, CSV tables, PNG figures, frozen manifest
notebooks/           executed analysis notebook
docs/                model card, field dictionary, step-by-step write-up (Chinese)
protocol.json        pre-registered analysis protocol
scripts/             data download with provenance hashes
tests/               tests for the analysis code
```

Not included: the dataset (CC BY-NC-ND 4.0 — download it with `scripts/download_data.py`), fitted model binaries and prediction files (regenerated by the pipeline), and the original paper and datasheet (linked below).

## Limitations

One final time-based hold-out plus four overlapping, after-the-fact backtest windows — not independent confirmations. Synthetic data has no investigation cost, loss amount, entity links or intra-month event order, so no monetary savings or real-time velocity windows can be derived. SHAP is a model explanation, not causal evidence; removing age from the inputs does not by itself guarantee fairness.

## Data and citation

- Dataset: [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) · [Feedzai repository](https://github.com/feedzai/bank-account-fraud) · [datasheet](https://raw.githubusercontent.com/feedzai/bank-account-fraud/main/documents/datasheet.pdf)
- Jesus, S. et al. (2022). *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation.* NeurIPS 2022 Datasets and Benchmarks.

The analysis protocol differs from the paper's, so this is an independent analysis, not a reproduction of the paper's numbers.
