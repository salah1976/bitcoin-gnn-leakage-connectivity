# Results

This directory contains the tables and figures reported in the paper:

**"When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation of Graph Neural Networks for Bitcoin Fraud Detection"**

The results were generated using the leakage-aware GriffinGNN evaluation framework on the BITFRAUD benchmark under a strict forward temporal protocol.

## Tables

### `table_connectivity_spectrum.csv`

Performance of GriffinGNN across the four graph-construction protocols spanning the connectivity spectrum.

Reported metrics include:

* Test-to-test leakage rate ($\lambda$)
* AUC-PR
* AUC-ROC
* F1-score

The table quantifies how predictive performance changes as graph connectivity is progressively restricted.

---

### `table_exposure_distribution.csv`

Distribution of fraudulent transactions across structural exposure categories.

The table reports:

* Number of transactions
* Number of fraud cases
* Fraud prevalence

for both connected and structurally isolated test transactions.

---

### `table_connected_vs_isolated.csv`

Performance decomposition of GriffinGNN according to structural exposure.

Metrics include:

* AUC-PR
* AUC-ROC
* F1-score

reported separately for connected and isolated fraud transactions.

This analysis identifies the graph regions responsible for the majority of predictive performance.

---

### `table_gnn_vs_lightgbm.csv`

Comparison between GriffinGNN and the LightGBM baseline across all evaluated connectivity regimes.

Reported metrics include:

* AUC-PR
* AUC-ROC
* F1-score
* Structural leakage level ($\lambda$)

The table evaluates whether graph-based learning provides benefits beyond a strong tabular baseline under progressively more restrictive graph-construction protocols.

---

## Figures

### `figure_connectivity_spectrum`

Overview of the four connectivity-aware evaluation protocols used throughout the study.

The figure illustrates the progressive reduction of test-to-test structural leakage from standard transductive graph construction to near-zero-leakage evaluation.

---

### `figure_connectivity_vs_ap`

Relationship between test-to-test leakage ($\lambda$) and GriffinGNN performance.

The figure shows how AUC-PR evolves across the connectivity spectrum and compares GriffinGNN against the LightGBM baseline.

The observed performance collapse under low-connectivity conditions constitutes the central empirical finding of the paper.

---

## Experimental Protocol

All results follow the same chronological evaluation procedure:

* Training snapshots: D1--D3
* Validation snapshot: D4
* Test snapshots: D5--D6

No future transaction, future label, or future external-intelligence information is used during training or model selection.

## Reproducibility

The results contained in this directory can be reproduced using the experimental scripts provided in:

```text
experiments/
```

The underlying benchmark dataset, feature-engineering pipeline, and fraud-label generation framework are available in the companion repository:

```text
bitcoin-fraud-benchmark
```

