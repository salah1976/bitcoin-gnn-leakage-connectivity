# Bitcoin GNN Leakage Evaluation

Leakage-aware evaluation framework for Graph Neural Networks in Bitcoin fraud detection.

## Overview

This repository accompanies the paper:

**"When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation of Graph Neural Networks for Bitcoin Fraud Detection"**

The study investigates how graph connectivity influences the reported performance of Graph Neural Networks (GNNs) in Bitcoin fraud detection.

Using the BITFRAUD benchmark, which contains more than 1.68 million transactions collected from six chronologically ordered Bitcoin blockchain snapshots spanning January to September 2025, the repository provides a systematic framework for evaluating the relationship between structural graph exposure and predictive performance.

Unlike conventional GNN evaluations, this work progressively restricts graph connectivity between training and testing regions in order to quantify the extent to which performance gains arise from genuine behavioural generalisation versus structural exposure.

The repository contains:

* Leakage-aware graph construction protocols
* Heterogeneous Bitcoin transaction graph generation
* GriffinGNN evaluation framework
* Connectivity-restricted evaluation pipelines
* Structural exposure measurement procedures
* Connected-versus-isolated fraud analysis
* Bootstrap confidence interval estimation
* Reproducibility scripts and experimental configurations

The benchmark dataset is shared with the companion repository:

**bitcoin-fraud-benchmark**

A small demonstration subset is provided for reproducibility purposes.

The complete dataset will be released upon publication.

---

## Main Contributions

* Leakage-aware evaluation framework for GNN-based fraud detection
* Heterogeneous Bitcoin transaction graph construction
* Strict chronological train-validation-test protocol
* Quantification of structural graph exposure
* Connectivity-restricted evaluation scenarios
* Connected versus isolated fraud analysis
* Fraud-exposed region characterization
* Bootstrap-based performance validation
* Reproducible GriffinGNN evaluation pipeline

---

## Experimental Protocol

Training period:

* D1–D3

Validation period:

* D4

Fully held-out test period:

* D5–D6

All evaluations follow strict forward temporal ordering.

No future information is used during training.

---

## Main Findings

The study demonstrates that graph connectivity is a major determinant of reported GNN performance.

Key observations include:

* Performance is highly concentrated in fraud-exposed graph regions.
* Structurally isolated fraud cases remain difficult to detect.
* Restricting connectivity causes substantial reductions in predictive performance.
* Aggregate metrics can mask strong dependence on graph continuity.
* Structural exposure must be explicitly quantified when evaluating graph-based fraud detection systems.

---

## Repository Structure

data/            Dataset metadata and benchmark references

experiments/     Leakage-aware GriffinGNN evaluation scripts

results/         Experimental outputs, tables, and figures

src/             Graph construction and evaluation utilities

---

## Reproducibility

The repository provides the code, graph-construction procedures, evaluation protocols, model configurations, and analysis scripts required to reproduce all experiments reported in the paper.

---

## Related Repository

The underlying benchmark dataset and tabular baseline models are available in:

**bitcoin-fraud-benchmark**

---

## License

MIT License.
