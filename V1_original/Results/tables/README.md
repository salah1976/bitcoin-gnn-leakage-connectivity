# Results

This directory contains the experimental outputs reported in the paper:

**"When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation of Graph Neural Networks for Bitcoin Fraud Detection"**

The results are generated using the leakage-aware GriffinGNN evaluation framework on the BITFRAUD benchmark.

## Files

### `table_baselines_v26.csv`

Summary of the primary experimental results.

The table compares:

* LightGBM baseline
* GriffinGNN under standard evaluation
* GriffinGNN under progressively restricted connectivity protocols

For each model, the table reports:

* AUC-PR
* AUC-ROC
* F1-score
* Estimated structural leakage level

This table supports the central finding of the paper that GNN performance strongly depends on graph connectivity and structural exposure.

---

### `table_connected_vs_isolated.csv`

Performance decomposition across graph regions.

The table reports separate results for:

* Connected fraud transactions
* Structurally isolated fraud transactions

Metrics include:

* AUC-PR
* AUC-ROC
* F1-score
* Precision
* Recall

The analysis demonstrates that predictive performance is concentrated in fraud-exposed graph regions, while structurally isolated fraud remains substantially more difficult to detect.

---

### `table_bootstrap_ci_v26.csv`

Bootstrap-based statistical validation of the reported results.

The table contains:

* Mean metric values
* 95% confidence intervals
* Confidence interval widths

Metrics include:

* AUC-ROC
* AUC-PR
* F1-score
* Precision
* Recall

Confidence intervals are estimated using 1,000 bootstrap resamples of the held-out test set.

---

### `table_threshold_sweep_v26.csv`

Threshold sensitivity analysis.

The table reports model performance across multiple classification thresholds and includes:

* AUC-ROC
* AUC-PR
* F1-score
* Precision
* Recall
* Accuracy

The threshold selected in the paper is determined exclusively on the validation period (D4) and evaluated on the fully held-out D5–D6 test period.

### `table comparison_bitfraud_elliptic.csv


Comparison of connectivity sensitivity between the proposed Bitcoin Fraud benchmark and Elliptic++. The table reports the AUC-PR obtained under the standard evaluation protocol and the reduced-connectivity protocol, together with the absolute and relative performance degradation.

**Columns**

- `dataset` – Benchmark name (Bitcoin Fraud or Elliptic++).
- `reduced_AUC_PR` – AUC-PR obtained under the reduced-connectivity evaluation protocol.
- `standard_AUC_PR` – AUC-PR obtained under the standard evaluation protocol.
- `delta_AUC_PR` – Absolute difference in AUC-PR between the standard and reduced protocols.
- `relative_drop_pct` – Relative percentage decrease in AUC-PR from the standard to the reduced protocol.



---

## Evaluation Protocol

All results follow the strict forward temporal protocol:

* Training: D1–D3
* Validation: D4
* Testing: D5–D6

No future transaction information, future labels, future address history, or future external-intelligence reports are available during training or validation.

## Reproducibility

The tables contained in this directory can be reproduced using the scripts provided in:
```text
experiments/run_leakage_aware_griffin_gnn.py
```

and the processed BITFRAUD benchmark dataset distributed through the companion repository:

```text
bitcoin-fraud-benchmark
```

