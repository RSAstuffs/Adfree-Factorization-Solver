Quantum-Inspired Annealing Factorization Solver

A classical simulation of quantum annealing for integer factorization, featuring triangle qubit encoding, machine learning-enhanced search, and incremental solving with state logging.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Quantum-Inspired-purple" alt="Quantum Inspired"/>
</p>

---

## 🌟 Overview

This project implements a **simulated quantum annealer** that mimics the behavior of real quantum computers (like D-Wave systems) to solve the integer factorization problem. Given a semiprime `N = p × q`, it finds the prime factors `p` and `q`.

### How It Mimics Quantum Behavior

| Quantum Effect | Classical Simulation |
|----------------|---------------------|
| **Quantum Tunneling** | Metropolis acceptance criterion at finite temperature |
| **Entanglement** | Triangle qubit pairs with equality constraints |
| **Adiabatic Evolution** | Exponential temperature cooling + incremental qubit activation |
| **Ising Hamiltonian** | QUBO energy function: `(p×q - N)²` |
| **Superposition** | Probabilistic bit flips with learned biases |
| **Ground State Finding** | Energy minimization via simulated annealing |

---

## ✨ Features

### 🧊 Triangle Qubit Architecture
- Each logical bit is represented by a **source-triangle pair** for error correction
- Constraint: `triangle_state == source_state` (entanglement-like coupling)
- Heavy penalty for constraint violations guides the search

### 🌡️ Temperature-Controlled Annealing
- **Exponential cooling schedule**: `T(t) = T₀ × (T_f/T₀)^(t/t_max)`
- **Metropolis acceptance**: `P(accept) = exp(-ΔE/T)` for uphill moves
- Allows escaping local minima at high temperatures

### 🧠 Machine Learning Enhancement
- **Neural clause learner** predicts configuration quality
- **Policy network** for intelligent bit selection
- **Q-learning** for action-value estimation
- **Trap detection** for √N proximity avoidance

### 📚 Learning from Experience
- **Clause learning**: Remembers bad configurations (SAT-solver inspired)
- **Bit correlations**: Learns which bit combinations work well
- **Elite population**: Maintains diverse good solutions
- **Tabu/Nogood lists**: Avoids revisiting failed regions

### 🔄 Incremental Solving
- Progressive qubit activation (adiabatic-style)
- Constraint propagation from mathematical properties
- State logging for Z3-style incremental solving
- Checkpoint/resume support for long runs

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-annealing-factorizer.git
cd quantum-annealing-factorizer

# Install dependencies
pip install numpy

# Optional: Install PyTorch for neural policy network
pip install torch
```

### Basic Usage

```python
from incremental_annealing_with_logging import IncrementalQuantumAnnealing

# Create annealer for factoring N
N = 143  # = 11 × 13
annealer = IncrementalQuantumAnnealing(
    N=N,
    num_triangle_pairs=20,
    initial_temp=1000.0,
    final_temp=0.01
)

# Run the solver
best_config, best_energy = annealer.incremental_solve(
    num_steps=100,
    num_reads_per_step=10
)

# Extract factors
p, q = annealer.extract_factors_from_best_config(best_config, N)
print(f"Factors: {p} × {q} = {N}")
```

### GUI Mode

```bash
python annealing_gui_v2.py
```

Launches a modern Tkinter GUI with:
- Real-time progress visualization
- Parameter tuning controls
- Elite solution tracking
- Learning statistics dashboard

---

## 🎛️ Configuration Options

### Annealer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | Required | The semiprime to factor |
| `num_triangle_pairs` | 20 | Number of qubit pairs (auto-adjusts if too low) |
| `initial_temp` | 1000.0 | Starting temperature for annealing |
| `final_temp` | 0.01 | Final temperature (near-zero for exploitation) |
| `log_file` | `"triangle_qubit_states.log"` | State logging file (None to disable) |

### Solving Methods

```python
# Single run with restarts
best_config, best_energy = annealer.solve_with_restarts(
    num_restarts=10,
    num_steps=100
)

# Parallel solving (uses all CPU cores)
best_config, best_energy = annealer.solve_parallel(
    num_workers=None,  # Auto-detect
    num_steps=100,
    share_learning=True  # Share learned clauses between workers
)

# Run until convergence (with state persistence)
annealer.solve_until_convergence(
    state_file="annealing_state.json",
    max_restarts=None,  # Run indefinitely
    save_interval=5     # Save state every 5 restarts
)
```

---

## 🔬 How It Works

### 1. QUBO Encoding

The factorization problem is encoded as a **Quadratic Unconstrained Binary Optimization** (QUBO):

```
Energy(config) = (decode_p(config) × decode_q(config) - N)²
```

Where:
- First half of qubit pairs encode factor `p`
- Second half encode factor `q`
- Triangle qubits provide redundancy

### 2. Annealing Schedule

```
Temperature: T(t) = T_initial × (T_final / T_initial)^(t / t_max)

Acceptance: P(ΔE > 0) = exp(-ΔE / T)
```

At high T → explore broadly (quantum tunneling analog)
At low T → exploit best solutions (ground state convergence)

### 3. Constraint Propagation

Mathematical constraints are derived automatically:
- If N is odd → both factors must be odd (LSB = 1)
- Low bits of `p × q` must match low bits of N
- Modular arithmetic constraints

### 4. √N Trap Avoidance

For RSA semiprimes, both factors should be **far from √N**. The solver:
- Detects when `p ≈ q ≈ √N` (the "trap")
- Applies massive energy penalties
- Learns to diverge toward asymmetric factors

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `triangle_qubit_states.log` | JSON log of all qubit state transitions |
| `annealing_state.json` | Checkpoint for resume (config, best solution, learned clauses) |
| `z3_constraints.smt2` | Exported constraints for Z3 SMT solver |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 IncrementalQuantumAnnealing                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Triangle   │  │  Triangle   │  │  Triangle   │  ...    │
│  │  Qubit Pair │  │  Qubit Pair │  │  Qubit Pair │         │
│  │  (p bit 0)  │  │  (p bit 1)  │  │  (q bit 0)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MLClauseLearner (Neural Net)           │   │
│  │  • Predicts configuration quality                   │   │
│  │  • Trap detection (√N proximity)                    │   │
│  │  • Suggests escape flips                            │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Learned    │  │    Elite     │  │    Tabu      │      │
│  │   Clauses    │  │  Population  │  │    List      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  Temperature Schedule │ Metropolis │ Constraint Propagation │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Examples

### Factor Small Numbers

```python
# Factor 15 = 3 × 5
annealer = IncrementalQuantumAnnealing(N=15, num_triangle_pairs=10)
config, energy = annealer.incremental_solve(num_steps=50)
```

### Factor Larger Semiprimes

```python
# Factor 2021 = 43 × 47
annealer = IncrementalQuantumAnnealing(
    N=2021,
    num_triangle_pairs=30,
    initial_temp=5000.0
)
config, energy = annealer.solve_with_restarts(num_restarts=20, num_steps=200)
```

### Resume from Checkpoint

```python
annealer = IncrementalQuantumAnnealing(N=143)
checkpoint = annealer.load_checkpoint("annealing_state.json")
# Continue solving from saved state...
```

---

## 📈 Performance Tips

1. **More triangle pairs** = more precision but slower
2. **Higher initial temperature** = better exploration, needs more steps
3. **Use `solve_parallel()`** for multi-core speedup
4. **Enable `share_learning=True`** to share discoveries between workers
5. **For large N**, increase `num_steps` and `num_restarts`

---

## 🔗 Related Work

- [D-Wave Quantum Annealing](https://www.dwavesys.com/)
- [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [QUBO Formulation](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)
- [Adiabatic Quantum Computing](https://en.wikipedia.org/wiki/Adiabatic_quantum_computation)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Performance optimizations
- New learning heuristics
- Better trap escape strategies
- GPU acceleration


