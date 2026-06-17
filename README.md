# EERLoss: A Novel Loss Function for Training Deep Biometric Models

Official repository for the paper: **"EERLoss: A Novel Loss Function for Training Deep Biometric Models. A Case Study in Keystroke Dynamics"**, submitted to *IEEE Transactions on Information Forensics and Security (TIFS)*.

<h5 align="center"><a href="https://scholar.google.com/citations?user=fXVZRNsAAAAJ&hl=en&oi=ao">Nahuel Gonzalez</a> &emsp;
<a href="https://scholar.google.com/citations?user=3rjxsSYAAAAJ&hl=en">Marta Robledo Moreno</a> &emsp;
<a href="https://scholar.google.com/citations?user=4ulTK3wAAAAJ&hl=en&oi=ao">Ivan de Andres Tame</a> &emsp;
<a href="https://scholar.google.com/citations?user=KYMQ0tsAAAAJ&hl=es">Ruben Vera Rodriguez</a> &emsp;
<a href="https://rubentolosana.github.io/">Ruben Tolosana Moranchel</a> &emsp;<br></h5>

[![Status](https://img.shields.io/badge/Status-Under_Revision-orange)](https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security/ieee-transactions)
[![arXiv](https://img.shields.io/badge/arXiv-ID_AQUÍ-B31B1B.svg)](https://arxiv.org/abs/ID_AQUÍ)

---

## Table of Contents
- [About the Study](#about-the-study)
  - [Key results](#key-results)
- [1. Motivation](#1-motivation)
- [2. Problem Statement & Definitions](#2-problem-statement--definitions)
- [3. EERLoss: Proposed Loss Function](#3-eerloss-proposed-loss-function)
  - [3.1 Smooth FAR / FRR](#31-smooth-far--frr)
  - [3.2 A Smooth Implementation of Binary Search](#32-a-smooth-implementation-of-binary-search)
  - [3.3 Direct EER Loss](#33-direct-eer-loss)
  - [3.4 Area-Based Formulation of the Loss Function](#34-area-based-formulation-of-the-loss-function)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation](#5-installation)
- [6. Usage](#6-usage)
  - [PyTorch](#pytorch)
  - [TensorFlow / Keras](#tensorflow--keras)
- [8. Citation](#8-citation)
- [9. Acknowledgment](#9-acknowledgment)

---

## About the Study

Deep learning approaches to biometric verification are commonly trained by optimizing indirect objectives, creating a misalignment between the optimization process and the primary evaluation metric, typically the Equal Error Rate (EER). This paper introduces EERLoss: a subdifferentiable, arbitrarily accurate approximation to EER for training deep biometric models. Furthermore, this framework has the potential to be adapted to optimize any specific operating point on the DET curve, enhancing its generalizability. 

To validate this approach, EERLoss is evaluated on a particularly demanding behavioral biometric modality: keystroke dynamics verification. This task is characterized by its high intra-class and low inter-class variability. Experiments are conducted on the large-scale KVC-onGoing benchmark, incorporating data from over 185,000 subjects across different scenarios. 

### Key results

* A new state-of-the-art result on the large-scale KVC-onGoing benchmark, where EERLoss significantly outperforms the previous best method, achieving a relative EER reduction of up to 30.7%. 

* The proposed method converges substantially faster compared to other losses, reducing the SoTA's overall training cost from 92 hours and 47 minutes to only 6 hours and 52 minutes in the final experiment. 

* A comprehensive ablation study initially demonstrates the superiority of EERLoss in comparison to existing state-of-the-art loss functions. 

* Geometrically, they assume that clusters have comparable radii, that they are approximately equidistant, and that inter-cluster overlap is negligible, allowing a separating margin. 

---

## 1. Motivation

Biometric verification systems decide whether two samples belong to the same identity while trading off security and usability. In this context, the Equal Error Rate (EER) denotes performance at the operating point where the False Acceptance Rate (FAR) is equal to the False Rejection Rate (FRR). 

Deep biometric models are commonly trained to optimize for embedding separability, creating a fundamental misalignment with the common evaluation standard, typically the EER. This inherent mismatch between the training objective (separability) and the evaluation metric (EER) can result in sub-optimal performance, particularly when the decision boundary is critical to the security and usability trade-off. Unlike conventional losses, EERLoss aligns the training objective directly with the evaluation metric, optimizing the actual operational performance of a biometric system. 

## 2. Problem Statement & Definitions

For each training batch, we define L as the set of distances between embeddings of samples that share the same label. The set I will consist of the distances between the embeddings of samples with different labels:

$$FRR(L,d)=1-\frac{|\{d^{L}\in L : d^{L} > d\}|}{|L|}$$

$$FAR(I,d)=\frac{|\{d^{I}\in I : d^{I} < d\}|}{|I|}$$

Problem statement: Find a subdifferentiable function $\mathcal{L}(L,I)$ that, given the sets L and I containing the distances between samples of the same user and those of different users, respectively, approximates $EER(L,I)$ arbitrarily well. 

## 3. EERLoss: Proposed Loss Function

### 3.1 Smooth FAR / FRR

Our points of departure are the following smoothings of the equations for FAR and FRR:

$$FRR_{S}(L,d)=1-\frac{1}{|L|}\sum_{d_{i}^{L}\in L}M_{R}(d,d_{i}^{L})$$

$$FAR_{S}(I,d)=\frac{1}{|I|}\sum_{d_{j}^{I}\in I}M_{L}(d,d_{j}^{I})$$

Any continuous, monotonously increasing function f such that $f(0)=0$ and $f(x)\to 1$ fast enough when $x\to\infty$ should suffice for the purpose. In particular, tanh was chosen for behaving well when training neural networks, and having fast tensorial implementations in modern GPUs.

### 3.2 A Smooth Implementation of Binary Search

We now want to use the proposed FAR and FRR approximations to perform a binary search for $d_{EER}$. Note that the initial endpoints are based on the average of the distance values in L and I. Thus, the 0.5 and 1.5 constants are used to make sure that $d_{L}<d_{R}$ in the initial search interval. 

### 3.3 Direct EER Loss

We simply estimate $d_{EER}$ with Alg. 4, compute the FRR and FAR at $d_{EER}$ using Algorithms 1 and 2, take the EER to be its average, and use it as the loss value. Although the above algorithm encodes the EER objective directly, it is not optimal for training a neural network. 

### 3.4 Area-Based Formulation of the Loss Function

To overcome the limitation discussed in the previous subsection, we propose a loss function that measures the area of overlap between the FAR and FRR curves:

$$\mathcal{L}_{AREA}(L, I) = \frac{A(L, I)}{d_{EER}}$$

For this purpose, we introduce a constant $\beta$ such that $0<\beta<2$, to delinearize the contribution of points further from $d_{EER}$.

## 4. Repository Structure

```text
EERLoss/
├── pytorch/
│   ├── eer_torch.py        # Direct EER loss (Alg. 5) — PyTorch
│   └── eerArea_torch.py    # Area-based EERLoss (Alg. 7, β) — PyTorch
├── tensorflow/
│   ├── eer_keras.py        # Direct EER loss (Alg. 5) — TensorFlow / Keras
│   └── eerArea_keras.py    # Area-based EERLoss (Alg. 7, β) — TensorFlow / Keras
├── assets/                 # Figures used in this README
└── README.md
```

## 5. Installation

```bash
# PyTorch implementation
pip install torch torchvision

# TensorFlow implementation
pip install tensorflow tensorflow-addons

```

## 6. Usage

### PyTorch

```python
import torch
from pytorch.eerArea_torch import get_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SETS_PER_BATCH       = 40
SAMPLES_PER_SET      = 8
BINARY_SEARCH_STEPS  = 16
BETA                 = 0.85

criterion = get_loss(device, BETA, SETS_PER_BATCH, SAMPLES_PER_SET, BINARY_SEARCH_STEPS)

loss = criterion(embeddings)
loss.backward()

```

### TensorFlow / Keras

```python
import tensorflow as tf
from tensorflow.eerArea_keras import get_loss

SETS_PER_BATCH      = 40
SAMPLES_PER_SET      = 8
BINARY_SEARCH_STEPS  = 16
BETA                 = 0.85

eer_loss = get_loss(BETA, SETS_PER_BATCH, SAMPLES_PER_SET, BINARY_SEARCH_STEPS)

loss_value = eer_loss(y_true=None, y_pred=embeddings)

```

## 8. Citation

```bibtex
@article{gonzalez2026eerloss,
  title   = {EERLoss: A Novel Loss Function for Training Deep Biometric Models. A Case Study in Keystroke Dynamics},
  author  = {Gonzalez, Nahuel and Robledo-Moreno, Marta and DeAndres-Tame, Ivan and Vera-Rodriguez, Ruben and Tolosana, Ruben},
  journal = {arxiv},
  year    = {2026},
  note    = {Under revision TIFS}
}
```

## 9. Acknowledgment

This project has been supported by Cátedra ENIA UAM-VERIDAS en IA Responsable (NextGenerationEU PRTR TSI100927-2023-2) and TRUST-ID (PID2025-173396OB-100 MICIU/AEI and the EU). Robledo-Moreno is supported by a FPI Fellowship (FPI-UAM-2025).