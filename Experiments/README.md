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

## Reproducibility

The script contained in this directory reproduces the complete leakage-aware GNN evaluation pipeline reported in the paper, including graph construction, model training, structural exposure analysis, connected-versus-isolated evaluation, and bootstrap-based statistical validation.

