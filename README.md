# EECS 567 Final Project (Winter 2026) - Team 3

This repository contains the implementation and evaluation of **goal-conditioned reinforcement learning (GCRL)** methods in both **online** and **offline** settings.

We benchmark modern algorithms across diverse continuous control environments, focusing on **generalization, stability, and sample efficiency**.

---

## Projects

### Project 2: Online Goal-Conditioned Reinforcement Learning
- **Benchmark:** JaxGCRL  
- **Environments:** Brax (GPU-based MuJoCo)
  - Locomotion tasks
  - Manipulation tasks  

**Baselines:**
- Contrastive Reinforcement Learning (CRL)
- Goal-conditioned SAC
- Goal-conditioned PPO
- Goal-conditioned TD3

---

### Project 3: Offline Goal-Conditioned Reinforcement Learning
- **Benchmark:** OGBench  
- **Environments:**
  - Locomotion maze environments
  - Manipulation environments
  - Powderworld

**Baselines:**
- Offline Contrastive RL (CRL)
- HIQL (Hierarchical Implicit Q-Learning)
- QRL (Query-based RL)
- Implicit Q-Learning / V-Learning

---

## Problem Formulation

We study **goal-conditioned policies** of the form:

$$
\pi(a \mid s, g)
$$

where:
- $s$ = state  
- $g$ = goal  
- $a$ = action  

The objective is to learn policies that:
- Generalize across goals
- Achieve high reward efficiently
- Remain stable in both online and offline regimes

---

## Experimental Setup

We evaluate algorithms across:
- Multiple environments (locomotion + manipulation)
- Online vs offline data regimes
- Different goal distributions

**Key evaluation metrics:**
- Success rate
- Episode return
- Generalization to unseen goals

---
