ML-Solver: Quantum-Inspired Factorization with Machine Learning

A hybrid quantum-classical factorization solver that combines **simulated quantum annealing** with **transformer-based machine learning** to factor large integers. Features GPU acceleration via Intel iGPU/CUDA for enhanced performance.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🧠 Transformer-Based Learning**: LLM-style BitTransformer with multi-head attention, Mixture of Experts (MoE), RoPE embeddings, and GQA
- **⚛️ Quantum-Inspired Annealing**: Simulated annealing with triangle qubit pairs and carry-aware bit propagation
- **🚀 GPU Acceleration**: Intel iGPU (IPEX), NVIDIA CUDA, Apple MPS, and OpenCL support
- **📊 Real-Time GUI**: Tkinter-based interface with live statistics, progress bars, and training metrics
- **🎯 Coppersmith-Ready**: Optimized for finding Most Significant Bits (MSB) for lattice attacks
- **💾 Checkpoint System**: Save/resume long-running factorization attempts
- **🔄 Adaptive Learning**: Online learning from factorization attempts with experience replay

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ML-Solver Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Input N   │───▶│  Triangle    │───▶│  Annealing   │   │
│  │  (to factor)│    │  Qubit Pairs │    │    Engine    │   │
│  └─────────────┘    └──────────────┘    └──────┬───────┘   │
│                                                 │           │
│                     ┌───────────────────────────┘           │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              BitTransformer (LLM-style)              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │  RoPE   │ │  GQA    │ │  MoE    │ │ SwiGLU  │   │   │
│  │  │Embedding│ │Attention│ │ Experts │ │   FFN   │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                       │
│                     ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Output Heads                        │   │
│  │  • Flip Scores    • Value Estimation                │   │
│  │  • Confidence     • Factor Attribution              │   │
│  │  • Pair Affinity  • Product Direction               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- Python 3.10+
- NumPy
- Tkinter (usually included with Python)

### Optional (for GPU acceleration):
- **Intel iGPU**: `intel-extension-for-pytorch`
- **NVIDIA GPU**: PyTorch with CUDA
- **Apple Silicon**: PyTorch with MPS

## 🚀 Installation

### 1. Clone the repository

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install numpy

# For Intel iGPU acceleration (recommended for Intel CPUs):
pip install torch==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Or for NVIDIA CUDA:
pip install torch

# Or CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Intel GPU drivers (for Intel iGPU only)
```bash
# Add Intel repository
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/gpu/ubuntu noble unified" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu.list

# Install drivers
sudo apt update
sudo apt install -y libze1 libze-intel-gpu1 intel-opencl-icd
```

## 💻 Usage

### GUI Mode (Recommended)
```bash
cd policy/annealing/hourglass
python annealing_gui_v2.py
```

The GUI provides:
- **Control Panel**: Start/stop annealing, adjust parameters
- **Statistics Tab**: Real-time metrics, bit accuracy, MSB progress
- **Training Tab**: Transformer architecture details, GPU status, learning progress
- **Visualization**: Energy landscape, factor convergence

### CLI Mode
```python
from incremental_annealing_with_logging import IncrementalQuantumAnnealing

# Factor a number
N = 15 * 17  # 255
solver = IncrementalQuantumAnnealing(N=N)

# Run solver
best_config, best_energy = solver.solve_with_restarts(
    num_restarts=10,
    num_steps=100,
    num_reads_per_step=10
)

# Get factors
p, q = solver._decode_factors(best_config)
print(f"Factors: {p} × {q} = {p*q}")
```

## ⚙️ Configuration

### Model Presets

| Preset | d_model | Layers | Heads | Experts | Parameters |
|--------|---------|--------|-------|---------|------------|
| Micro  | 64      | 1      | 2     | 1       | ~100K      |
| Turbo  | 128     | 2      | 4     | 2       | ~500K      |
| Lean   | 192     | 3      | 6     | 2       | ~1.2M      |
| Medium | 256     | 4      | 8     | 4       | ~2.5M      |
| Heavy  | 512     | 6      | 8     | 8       | ~10M       |

### Annealing Parameters

```python
solver = IncrementalQuantumAnnealing(
    N=your_number,
    initial_temp=1000.0,    # Starting temperature
    final_temp=0.1,         # Final temperature
    cooling_rate=0.99,      # Exponential cooling
)
```

## 🎯 Strategies

### For Quick Factorization (small N < 64 bits)
- Use "Micro" or "Turbo" preset
- Lower iteration count
- Disable transformer learning

### For Large N (2048+ bits) - Coppersmith Attack
- Use "Lean" or "Medium" preset
- Enable GPU acceleration
- Focus on MSB accuracy (visible in GUI)
- Export partial results for lattice methods

## 📈 GPU Acceleration

The solver automatically detects and uses available GPU:

```
[GPU] ✅ Intel IPEX with XPU (Intel iGPU) available
[GPU]    Device: Intel(R) UHD Graphics 770
[BitTransformer] 🚀 Preloading weights to intel_ipex...
[BitTransformer] ✅ Weights cached on GPU
```

### Supported Backends
| Backend | GPU Type | Package |
|---------|----------|---------|
| `intel_ipex` | Intel iGPU | `intel-extension-for-pytorch` |
| `torch_cuda` | NVIDIA | `torch` with CUDA |
| `torch_mps` | Apple Silicon | `torch` |
| `cupy` | NVIDIA | `cupy` |

## 📁 Project Structure

```
ML-Solver/
├── policy/
│   └── annealing/
│       └── hourglass/
│           ├── incremental_annealing_with_logging.py  # Core solver
│           ├── annealing_gui_v2.py                    # GUI interface
│           └── checkpoints/                           # Saved states
├── venv/                                              # Virtual environment
└── README.md
```

## 🔬 Technical Details

### BitTransformer Components
- **RoPE (Rotary Position Embedding)**: Encodes bit position information
- **GQA (Grouped Query Attention)**: Efficient multi-head attention
- **MoE (Mixture of Experts)**: Specialized expert networks for different bit patterns
- **SwiGLU**: Gated activation in feed-forward layers
- **RMSNorm**: Faster layer normalization

### Learning Features
- Online learning from factorization attempts
- Experience replay buffer
- Adaptive exploration rate
- Context memory for past configurations
- Correlation analysis between bits

## 📊 Output Metrics

- **Bit Accuracy**: Percentage of bits matching between P×Q and N
- **MSB Match**: Most significant bits correct (critical for Coppersmith)
- **Energy**: Combined constraint satisfaction score
- **Diff**: |P×Q - N| (goal is 0)

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional GPU backends
- Improved learning algorithms
- Better initialization strategies
- Coppersmith integration

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by quantum annealing approaches to factorization
- Transformer architecture based on modern LLM designs
- GPU acceleration via Intel Extension for PyTorch

---

**Note**: This is a research tool for exploring hybrid quantum-classical factorization approaches. It is not intended to break real-world cryptographic systems.

