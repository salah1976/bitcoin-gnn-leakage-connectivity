# Figures

This directory contains the main figures supporting the analyses reported in the paper:

**"When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation of Graph Neural Networks for Bitcoin Fraud Detection"**

## `figure_connectivity_spectrum`

Connectivity spectrum used throughout the leakage-aware evaluation framework.

The figure summarizes the four graph-construction protocols investigated in the study and their corresponding levels of test-to-test structural leakage:

* Standard graph construction ($\lambda = 94.99%$)
* Known-address construction ($\lambda = 24.27%$)
* Top-$K$ controlled construction ($\lambda = 4.87%$)
* Zero-leakage construction ($\lambda \approx 0%$)

This figure provides an overview of the experimental methodology and illustrates how graph connectivity is progressively restricted to quantify its impact on GNN performance.

---

## `figure_connectivity_vs_ap`

Relationship between test-to-test structural leakage ($\lambda$) and GriffinGNN performance.

The figure reports AUC-PR across the four connectivity regimes and compares the resulting performance to the LightGBM baseline.

The results reveal a sharp performance transition between moderate-connectivity and low-connectivity regimes. GriffinGNN remains competitive with the LightGBM baseline when structural connectivity is preserved ($\lambda \geq 24.27%$), but performance collapses when test-to-test connectivity is largely removed ($\lambda \leq 4.87%$).

Together, these figures summarize the central finding of the paper: graph connectivity and structural exposure constitute major determinants of the predictive performance reported by Graph Neural Networks for Bitcoin fraud detection.

