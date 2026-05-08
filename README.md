# Torch Architecture Inspector
> **An Interactive HTML Model Profiler**
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)



A lightweight, zero-dependency interactive visualizer for PyTorch model architectures in HTML. 

Unlike heavy visualization tools, **Torch Architecture Inspector** generates a standalone, searchable HTML tree that helps you identify parameter-heavy layers, inspect dimensions, and copy layer paths directly for code access.

## Key Features
- **Zero External Dependencies:** Only requires `torch`. No Graphviz or ONNX needed.
- **Weight Profiling:** Calculation of parameter percentages per layer (perfect for SVD/Pruning analysis).
- **Interactive Search:** Filter by layer type (e.g., `Conv1d`), dimensions (e.g., `1536`), or specific paths.
- **Path Copying:** One-click to copy the internal PyTorch path (e.g., `tower.blocks.0.linear`) for direct manipulation.
- **Standalone HTML:** Generates a single file you can share or open in any browser.

## Installation
You can clone this repository and install it locally:
```bash
git clone https://github.com/SergioHdezG/torch-architecture-inspector.git
cd torch-architecture-inspector
pip install .
```

or just copy the `generate_model_inspector` function into your project or clone this repo.
```bash
pip install torch
```

## Quick Start
```python
import torch
from torch_architecture_inspector import generate_model_inspector

# Initialize your model
model = YourCustomModel()

# Generate the interactive report
generate_model_inspector(model, output_file="model_report.html")
```

## Why use this?

Deep learning models, especially **Transformers** or specialized genomic networks like **AlphaGenome**, often contain hundreds or thousands of nested layers. Managing and auditing these architectures presents several challenges:

*   **Standard Print:** Using `print(model)` in PyTorch produces a text output that is often too dense and vertically long to read or analyze effectively.
*   **Graph representation Tools:** While they are powerful tools for low-level operation debugging, they can be overwhelming and "too zoomed-in" when you only need to understand high-level class structures and parameter distribution.
*   **Torch Architecture Inspector:** Provides a clean, **hierarchical view** focused on:
    *   **Resource Allocation:** Instantly see which blocks consume the most parameters.
    *   **Architectural Metadata:** Quick access to kernels, strides, and layer dimensions without digging through code.
    *   **Developer Workflow:** Interactive search and path copying to bridge the gap between visualization and coding.


## Citation

```bibtext
@misc{torch_arch_inspector2026,
  author = {Hernández, S.},
  title = {Torch Architecture Inspector: An Interactive HTML Model Profiler},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SergioHdezG/torch-architecture-inspector}}
}
```

