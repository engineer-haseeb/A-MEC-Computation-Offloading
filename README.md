# Learning Automata for Dynamic Multi-User Task Offloading in MEC

This repository contains the official high-fidelity discrete-event simulation engine developed for the research project **"Learning Automata for Dynamic Multi-User Task Offloading"**. The framework implements a decentralized multi-agent Learning Automata (LA-MEC) model to optimize computational latency and terminal energy consumption under dense cellular interference constraints.

## 🚀 Key Features
* **Decentralized Multi-Agent Learning:** Employs a linear reward-penalty ($L_{R-P}$) reinforcement paradigm.
* **Chaotic Interference Modeling:** Simulates path loss coupled with rapid, uncorrelated Rayleigh fast fading.
* **Comparative Benchmarking:** Direct validation against traditional centralized setups and uncoordinated Static Game-Theoretic models.

## 📊 Empirical Simulation Results

### 1. Baseline Framework Performance Bottlenecks
The uncoordinated Static Game baseline suffers heavily from chaotic strategy updates under scaling multi-user loads, locking the network into a permanent 100% channel collision saturation.
<p align="center">
  <img src="baseline_trends.png" width="95%" alt="Baseline Bottlenecks">
</p>

### 2. Proposed Learning Automata Adaptations
By adjusting decision profiles via environmental feedback, the proposed LA engine mitigates collisions, dropping network saturation systematically down to controlled operational levels.
<p align="center">
  <img src="proposed_la_trends.png" width="95%" alt="Proposed LA Trends">
</p>

### 3. Direct Optimization Benchmarks (at N = 30)
Under standard validation loads, the proposed model provides a **31.01% latency reduction** (1762.4 ms down to 1215.8 ms) and saves up to **61.5% terminal battery power** (0.13 J down to 0.05 J) compared to the state-of-the-art potential game.
<p align="center">
  <img src="comparison_suite.png" width="95%" alt="Comparison Benchmarks">
</p>

## 🛠️ Execution & Replication
To run the evaluation framework and regenerate the production-quality assets, execute the core script using standard Python 3.x deployment stacks:

```bash
pip install numpy matplotlib
python simulation.py
