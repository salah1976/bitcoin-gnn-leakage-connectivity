# Data

This directory contains benchmark metadata, feature definitions, label statistics, and documentation required to reproduce the leakage-aware Graph Neural Network evaluation framework.

## Files

* `feature_definitions.csv`: definitions of the 15 leakage-controlled learning features used in the transaction-level models.
* `label_statistics.csv`: transaction and fraud-label counts for each chronological snapshot.
* `snapshot_ranges.json`: chronological snapshot metadata and temporal split definition.
* `samples/`: optional small sample dataset for demonstration purposes.

## Dataset Origin

This repository reuses the BITFRAUD benchmark dataset introduced in the companion repository:

**bitcoin-fraud-benchmark**

The benchmark consists of six chronologically ordered Bitcoin blockchain snapshots (D1–D6) covering January 2025 to September 2025.

## Full Dataset Availability

The complete processed benchmark dataset is not stored directly in this GitHub repository because of file-size constraints and redistribution considerations associated with external intelligence sources.

Dataset access information:

[Dataset link to be added upon publication]

The full benchmark dataset will be deposited separately in a public data repository upon publication.

The expected processed release will contain:

* `tx_hash`
* `snapshot_id`
* `block_height`
* `timestamp`
* the 15 learning features listed in `feature_definitions.csv`
* `label_final`

Raw external-intelligence records are not redistributed directly. Fraud labels are derived exclusively from temporally valid external-intelligence reports available before each transaction timestamp.

## Temporal Protocol

The benchmark follows a strict forward temporal protocol:

* Training snapshots: D1, D2, D3
* Validation snapshot: D4
* Test snapshots: D5, D6

No future transaction, future label, or future external-intelligence information is used during training or validation.

## Graph Construction

The experiments reported in this repository construct heterogeneous transaction-address graphs from the benchmark data. Graph connectivity is progressively restricted under multiple evaluation protocols to quantify the influence of structural exposure on Graph Neural Network performance.

The underlying transaction features and fraud labels remain identical to those provided in the original BITFRAUD benchmark.
