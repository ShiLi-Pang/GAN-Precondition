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
src/
├── models/            # GAN architectures
├── solvers/           # Iterative solvers (interface)
├── physics/           # Helmholtz operator (simplified)
├── preconditioners/   # Learned preconditioner interface
└── utils/             # Visualization tools

scripts/
└── demo_inference.py  # Demo script
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python scripts/demo_inference.py
```

> Note: This demo uses a simplified setup and does not reproduce full experimental results.

---

## ⚠️ Important Notes

This repository is intended for **research demonstration purposes only**:

* The full training pipeline is **not included**
* Some implementation details are intentionally omitted
* The original datasets are **not publicly available**

---

## 📊 Reproducibility

Due to the complexity of the full system and data dependencies, this repository **does not aim to fully reproduce** the experimental results reported in the paper.

---

```
(Your paper citation here)
```
