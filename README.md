# GAN-based Deep Transfer Learning Preconditioner for 3D Frequency-Domain Acoustic Wave Equation

This repository provides a **demonstration implementation** of a GAN-based preconditioning framework for accelerating iterative solvers in large-scale 3D frequency-domain wave equation problems.

---

## 🔍 Overview

Solving large-scale linear systems is a central challenge in applications such as **Reverse Time Migration (RTM)** and **Full waveform inversion（FFWI）**.

This project explores the use of **Generative Adversarial Networks (GANs)** to learn an implicit approximation of the inverse impedance matrix, which is then used as a preconditioner within Krylov subspace methods (e.g., BiCGSTAB).

In addition, the framework is designed to support transferability across different frequency settings, enabling improved efficiency in multi-frequency scenarios.

---

## 🧠 Key Idea

* Learn a mapping that approximates the inverse of the impedance matrix
* Use the learned model as a preconditioner
* Accelerate the convergence of iterative solvers
* Support transferability across frequency settings 

---
## 📂 Project Structure

```
GAN_preconditioned_3D/
├── generator_3D.py        # Generator network
├── discriminator_3D.py    # Discriminator network
├── bicgstab_3D.py         # Iterative solver (BiCGSTAB)
├── ofd_matrix_4_3D.py     # Helmholtz operator construction
├── GAN_fd_3D.py           # GAN-based preconditioner

GANtrain_3D/
├── Homogeneous_example.ipynb                   # Training scripts (simplified)

GANtrain_TL_3D/
├── TL_21Hz_Homogeneous_example                    # Transfer learning scripts (simplified)
```
---

## ⚠️ Important Notes

This repository is intended for **research demonstration purposes only**:

* The full training pipeline is **not provided in a reproducible form**
* Some implementation details are **simplified or omitted**
* The provided scripts are intended to illustrate the core idea rather than reproduce full experimental results

---

## 📊 Reproducibility

Due to the complexity of the full workflow and data dependencies, this repository **does not aim to fully reproduce** the results reported in the paper.
