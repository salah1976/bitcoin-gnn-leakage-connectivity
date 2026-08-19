# Experiments

This directory contains the scripts used to reproduce the leakage-aware Graph Neural Network experiments.

The experiments start from the processed BITFRAUD benchmark dataset and focus on graph construction, connectivity-aware evaluation, and structural exposure analysis.

## Scripts

### `run_leakage_aware_griffin_gnn.py`

Main experimental pipeline used throughout the paper.

The script performs:

* Loading the processed BITFRAUD benchmark dataset
* Construction of heterogeneous transaction-address graphs
* Generation of address-level structural features
* GriffinGNN training under a strict chronological protocol
* Validation-based threshold selection
* Evaluation on the fully held-out D5–D6 test period
* Connectivity-aware graph construction
* Connected-versus-isolated fraud analysis
* Structural exposure evaluation
* Bootstrap confidence interval estimation
* Comparison with previously reported benchmark baselines
* ### Targeted λ sweep

The pipeline additionally performs a targeted connectivity sweep to quantify how GriffinGNN performance changes as structural exposure (λ) is progressively reduced.

For each target λ, the Top-K connectivity parameter is identified using a binary search, exploiting the monotonic relationship between K and the resulting λ. The sweep uses a fine resolution of 0.5 percentage points in the critical 15–30% range, with coarser targets outside this transition region.

Each configuration is evaluated using AUC-PR, AUC-ROC, F1-score, precision, and recall. Results are cached to avoid redundant evaluations and continuously checkpointed in JSON format, allowing the sweep to be resumed after interruption.

The script reproduces the main results reported in the paper.

## Elliptic++ Cross-Benchmark Validation

This repository includes the external validation experiments conducted on the public **Elliptic++** benchmark to assess whether the connectivity-dependent behavior observed on the proposed Bitcoin benchmark generalizes to a graph with fundamentally different structural properties.

### Script

#### `ellipticpp_gnn_external_validation.py`

This script reproduces the cross-benchmark experiments reported in the paper. It performs a chronological evaluation of a GraphSAGE-based model on Elliptic++ under progressively constrained graph connectivity.

The following evaluation protocols are implemented:

1. **MLP baseline** (feature-only model without graph information)
2. **Standard graph**
3. **No test-test edges**
4. **No history-test edges**
5. **Fully isolated test graph**

The script automatically:

* loads the Elliptic++ dataset;
* constructs the graph for each connectivity protocol;
* trains the GraphSAGE model under the official chronological split;
* selects the decision threshold on the validation period;
* evaluates AUC-ROC, AUC-PR, Precision, Recall, and F1-score;
* exports the complete experimental results as CSV files.

### Purpose

These experiments provide an external validation of the connectivity-aware evaluation framework proposed in the paper.

Unlike the proposed heterogeneous Bitcoin benchmark, Elliptic++ is represented as a homogeneous transaction graph containing only transaction-to-transaction edges. Consequently, the benchmark exhibits fundamentally different structural connectivity characteristics.

The comparison demonstrates that the influence of graph connectivity is benchmark-dependent rather than a universal property of Graph Neural Networks. While predictive performance on the proposed benchmark is highly sensitive to structural exposure, Elliptic++ exhibits a markedly different response to graph isolation, highlighting the importance of explicitly measuring connectivity conditions when evaluating graph-based fraud detection systems.


These experiments support the paper's central claim that graph
connectivity should be treated as an explicit evaluation variable rather
than an implicit property of the benchmark.

## Experimental Protocol

Training snapshots:

* D1
* D2
* D3

Validation snapshot:

* D4

Test snapshots:

* D5
* D6

The protocol follows strict forward temporal evaluation.

No future transaction, future label, future address history, or future external-intelligence information is used during training or validation.

## Connectivity Analysis

The experiments evaluate Graph Neural Networks under progressively restricted graph-construction protocols in order to quantify the influence of structural connectivity on predictive performance.

The analysis includes:

* Connected fraud transactions
* Structurally isolated fraud transactions
* Fraud-exposed graph regions
* Structural exposure measurement
* Connectivity-aware performance decomposition

## Dataset Dependency

This repository reuses the processed BITFRAUD benchmark dataset introduced in the companion repository:

**bitcoin-fraud-benchmark**

Dataset construction, external-intelligence collection, feature engineering, and label generation are documented and implemented in the benchmark repository.

## Companion Repository

This work builds upon the BITFRAUD benchmark introduced in:

https://github.com/salah1976/bitcoin-fraud-benchmark

The companion repository contains:

- benchmark construction scripts
- blockchain data collection
- external-intelligence collection
- feature engineering
- fraud-label generation
- tabular baseline models

## Reproducibility

The script contained in this directory reproduces the complete leakage-aware GNN evaluation pipeline reported in the paper, including graph construction, model training, structural exposure analysis, connected-versus-isolated evaluation, and bootstrap-based statistical validation.

