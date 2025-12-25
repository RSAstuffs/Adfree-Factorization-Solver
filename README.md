Quantum Annealing Factorization with Neural Networks

A hybrid quantum-classical approach to integer factorization combining **simulated annealing**, **neural network learning**, and **transformer attention mechanisms**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Required-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This tool attempts to factorize large semiprimes (N = P × Q) using a novel combination of:

- **Simulated Annealing** with adaptive temperature control
- **Hourglass Neural Network** for bit pattern learning
- **Transformer Attention** (LLM-style) for capturing long-range bit dependencies
- **Factorization-Aware Learning** that directly optimizes for P×Q = N

The system learns from every factorization attempt, building up knowledge about which bit patterns lead closer to the solution.

## ✨ Features

### 🧠 Dual Neural Architecture
- **Hourglass Network**: Diamond-shaped architecture that compresses and expands bit patterns to learn local correlations
- **BitTransformer**: LLM-style multi-head self-attention that captures global bit dependencies

### 🎛️ Configurable Model Presets
| Preset | Parameters | Best For |
|--------|------------|----------|
| Light | ~2M | Quick iterations, testing |
| Medium | ~8M | Balanced speed/accuracy |
| Heavy | ~33M | Better learning |
| Ultra | ~76M | Maximum accuracy |

### 🔥 Adaptive Annealing
- Auto-scaling temperature based on problem size
- Automatic reheating when stuck in local minima
- Metropolis acceptance with configurable leniency

### 💾 State Persistence
- Save/resume long-running factorization attempts
- Neural network weights preserved between sessions
- Transformer context memory saved

### 🎯 Factorization-Aware Learning
- Learns which bits belong to P vs Q
- Predicts product error direction (too high/too low)
- Tracks bit-N correlations for smarter suggestions

## 🚀 Usage

### GUI Mode (Recommended)

```bash
python annealing_gui_v2.py
```

The GUI provides:
- Real-time progress visualization
- Learning statistics dashboard
- Model preset selection
- Bit selection strategy controls
- State save/load functionality

### CLI Mode

```python
from incremental_annealing_with_logging import IncrementalQuantumAnnealing

# Target number to factorize
N = 15  # = 3 × 5

# Create annealer
annealer = IncrementalQuantumAnnealing(
    N=N,
    num_pairs=8,  # Number of triangle qubit pairs
    log_file="factorization.log",
    initial_temp=None,  # Auto-scale
    final_temp=None,
    state_file="state.json"
)

# Run until convergence
result = annealer.solve_until_convergence(
    state_file="state.json",
    num_steps=1000,
    num_reads_per_step=10,
    max_restarts=100
)

if result:
    p, q = result
    print(f"Found factors: {p} × {q} = {p*q}")
```

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Bit Configuration           │
                    │    [1,0,1,1,0,0,1,0,1,1,0,1,...]   │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Hourglass     │     │  BitTransformer │     │     Random      │
    │    Network      │     │   (Attention)   │     │   Exploration   │
    │                 │     │                 │     │                 │
    │  Input → Expand │     │  Multi-Head     │     │   Uniform       │
    │  → Bottleneck   │     │  Self-Attention │     │   Sampling      │
    │  → Contract     │     │  + RoPE + MoE   │     │                 │
    │  → Output       │     │  + Context Mem  │     │                 │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────┐
                    │      Weighted Bit Selection         │
                    │   (Configurable % per strategy)     │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Simulated Annealing Step        │
                    │   Flip bits → Evaluate energy       │
                    │   Metropolis accept/reject          │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Learn from Attempt             │
                    │   Update NN weights, correlations   │
                    │   Track best solutions              │
                    └─────────────────────────────────────┘
```

## ⚙️ Configuration

### Bit Selection Strategy
Control how bits are selected for flipping:

```python
# In GUI or code:
strategy = {
    'transformer_pct': 50,   # % using transformer attention
    'hourglass_pct': 35,     # % using hourglass network
    'random_pct': 15         # % random exploration
}
```

### Model Presets
```python
# Light: Fast iteration
model_settings = {'d_model': 128, 'num_layers': 2, 'num_heads': 4, 'num_experts': 2}

# Medium: Balanced
model_settings = {'d_model': 256, 'num_layers': 4, 'num_heads': 8, 'num_experts': 4}

# Heavy: Better accuracy
model_settings = {'d_model': 512, 'num_layers': 6, 'num_heads': 16, 'num_experts': 4}

# Ultra: Maximum power
model_settings = {'d_model': 768, 'num_layers': 8, 'num_heads': 16, 'num_experts': 8}
```

### Metropolis Acceptance
```python
# Strict: Faster convergence, less exploration
min_accept_prob = 0.01

# Normal: Balanced
min_accept_prob = 0.05

# Lenient: More exploration, better ML learning
min_accept_prob = 0.15
```

## 📊 How It Works

### 1. Triangle Qubit Encoding
The factors P and Q are encoded as binary strings using "triangle qubits" that enforce multiplication constraints.

### 2. Energy Function
The energy function combines:
- **Constraint violations**: Penalties for invalid qubit states
- **Factorization error**: |P×Q - N| normalized by magnitude
- **Symmetry penalty**: Avoids the √N trap where P ≈ Q

### 3. Learning Loop
```
For each annealing step:
    1. Neural networks suggest promising bit flips
    2. Apply flip, calculate new energy
    3. Metropolis accept/reject based on ΔE and temperature
    4. Learn from attempt (update weights, correlations)
    5. Track best solutions found
```

### 4. Avoiding the √N Trap
A critical challenge is avoiding solutions where P ≈ Q ≈ √N. The system:
- Penalizes symmetric solutions in energy function
- Learns "escape patterns" that break symmetry
- Uses trap-aware bit selection

## 🔧 Key Classes

### `BitTransformer`
LLM-style transformer for bit sequence modeling:
- Multi-head self-attention with RoPE encoding
- Mixture of Experts (MoE) routing
- Context memory for historical patterns
- Factorization-aware output heads

### `MLClauseLearner`
Hourglass neural network for pattern learning:
- Diamond architecture (expand → bottleneck → contract)
- Skip connections for gradient flow
- Trap awareness and escape learning
- Bit correlation tracking

### `IncrementalQuantumAnnealing`
Main annealing engine:
- Adaptive temperature schedule
- Incremental pair activation
- State persistence
- Multi-strategy bit selection

## 📈 Performance Tips

1. **Start with Medium preset** for new problems
2. **Use Light preset** for quick exploration/testing
3. **Increase Transformer %** for problems with complex bit dependencies
4. **Enable lenient Metropolis** when ML learning seems stuck
5. **Save state frequently** for long-running attempts

## ⚠️ Limitations

- This is a **heuristic approach** - not guaranteed to find factors
- Large semiprimes (1000+ bits) require significant computation
- Memory usage scales with model size and problem size
- The √N trap can still be challenging for balanced semiprimes

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improved energy functions
- Better trap escape strategies
- More efficient attention mechanisms
- Parallelization improvements

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by quantum annealing approaches to optimization
- Transformer architecture based on "Attention Is All You Need"
- Hourglass design inspired by pose estimation networks

---

**Note**: This tool is for research and educational purposes. Integer factorization of large semiprimes remains computationally hard, and this tool provides a novel hybrid approach rather than a cryptographic attack.
