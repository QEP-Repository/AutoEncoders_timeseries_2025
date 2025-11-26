# Causality Autoencoder (CausAE)

## Reference
This repository contains the code for the Physics-Informed Code AutoEncoder (PICAE). 
If you use this code in research, please cite

> R. Rossi et al.,  
> *On the Use of Autoencoders to Study the Dynamics and the Causality Relations of Complex Systems with Applications to Nuclear Fusion*,  
> Submitted to **Computer Physics Communications**, 2025.

--- 

This repository provides the MATLAB implementation of the **Causality AutoEncoder (CausAE)**, a deep learning method developed to infer **causal relationships** and **dynamical dependencies** from multivariate time series.

The approach compares prediction performance:
- Using **all variables** (full model),
- **Excluding a candidate driver**, and observing degraded prediction accuracy.

This provides evidence of causal influence in nonlinear dynamic systems.
---

## Method Overview

CausAE:
- Builds **time-lagged embedded inputs** from data,
- Learns a **latent-state autoencoder**, and
- Predicts the next time step of system evolution.

A statistically robust **ensemble analysis** quantifies causal effects.


## Repository Structure

├── Numerical Data/  
│   ├── AR_C0.mat # Example autoregressive dataset - no coupling
│   ├── AR_C04.mat # Example autoregressive dataset - coupling
│   ├── Generate_AR.m
│   └── Generate_HenonHenon.m   
│  
├── Functions/ # Core algorithms
│   ├── CausAE_ModelTraining.m
│   ├── CausAE_Buffer.m
│   ├── CausAE_TrainingTestSet.m
│   ├── CausAE_ErrorAnalysis_Ensemble.m
│   └── (other utilities)
│
├── CausAE_main.m # Main execution script
└── README.md