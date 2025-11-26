# Physics-Informed Code Autoencoder (PIC-AE)

## Reference
This repository contains the code for the Physics-Informed Code AutoEncoder (PICAE). 
If you use this code in research, please cite

> R. Rossi et al.,  
> *On the Use of Autoencoders to Study the Dynamics and the Causality Relations of Complex Systems with Applications to Nuclear Fusion*,  
> Submitted to **Computer Physics Communications**, 2025.

--- 

This repository contains the implementation of the **Physics-Informed Code Autoencoder (PIC-AE)**, a method for discovering latent dynamics of nonlinear systems from **noisy proxy measurements** and optionally sparse direct observations.
The method is demonstrated using the **Lotka–Volterra** predator–prey system, but it is designed to be applicable to a broad class of nonlinear dynamical systems.

---

## Key Features

- Learns **latent governing equations** directly from data  
- Works with **proxy variables** (indirect system measurements)
- Can also incorporate **sparse observable constraints**
- Supports **Adaptive Physics–Data loss balancing**
- Trains using **GPU or CPU**
- Fully implemented in **MATLAB Deep Learning Toolbox**

---

## Method Overview

The PIC-AE consists of:

- An **encoder** that maps proxy measurements into a **latent state**
- A **decoder** that reconstructs the proxies from latent predictions
- A **physics model** that enforces consistency in latent dynamics

If sparse system observables exist, they are enforced as latent constraints.

---

## Repository Structure

├── Dataset/  
│   └── LotkaVolterra.mat    # Example dataset  
│  
├── Functions/  
│   ├── AE_network_LV.m      # Encoder/decoder architecture  
│   ├── ModelGradient_LV.m   # Gradient for fixed-parameter mode  
│   ├── ModelGradient_LV_SM.m# Gradient for self-modeling mode  
│   └── ModelGradient_LV_SM_withObservable.m  
│
├── PIC_AE_example_LotkaVolterra.m                     
└── README.md