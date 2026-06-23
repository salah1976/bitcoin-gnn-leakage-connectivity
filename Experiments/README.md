# Experiments

This directory contains the scripts used to reproduce the leakage-aware Graph Neural Network experiments reported in the paper:

**"When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation of Graph Neural Networks for Bitcoin Fraud Detection"**

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

The script reproduces the main results reported in the paper.

## Elliptic++ Cross-Benchmark Validation

This repository includes additional experiments conducted on the public
Elliptic++ benchmark to validate whether the connectivity-dependent
behavior observed on the proposed Bitcoin benchmark generalizes across
datasets with different graph structures.

### Scripts

#### `elliptic_validation_lgbm.py`

Performs a structural analysis of the Elliptic++ graph and verifies that
all transaction edges are intra-timestep. The script computes:

- edge connectivity statistics;
- train-test connectivity diagnostics;
- test-to-test connectivity analysis;
- LightGBM performance under temporal evaluation protocols.

The analysis confirms that Elliptic++ eliminates train-test graph
connectivity by construction, providing a useful contrast to the
proposed benchmark.

#### `elliptic_gnn_external_validation.py`

Evaluates GraphSAGE under multiple graph-isolation protocols:

1. Standard graph
2. No test-test edges
3. No history-test edges
4. Fully isolated test graph

A feature-only MLP baseline is also included.

The objective is not to optimize performance on Elliptic++, but to
analyze how progressively removing graph connectivity affects predictive
performance under a benchmark with fundamentally different structural
properties.

### Purpose

The Elliptic++ experiments serve as an external validation of the
connectivity-aware evaluation framework proposed in the paper.

The results demonstrate that connectivity-driven performance gains are
benchmark-dependent rather than universal properties of graph neural
networks. While the proposed benchmark exhibits strong sensitivity to
structural exposure, Elliptic++ remains largely stable under graph
isolation due to its intra-timestep graph design and highly informative
transaction features.

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

