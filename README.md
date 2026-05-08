# Torch Architecture Inspector

A lightweight, zero-dependency interactive visualizer for PyTorch model architectures. 

Unlike heavy visualization tools, **Torch Architecture Inspector** generates a standalone, searchable HTML tree that helps you identify parameter-heavy layers, inspect dimensions, and copy layer paths directly for code access.

## Key Features
- **Zero External Dependencies:** Only requires `torch`. No Graphviz or ONNX needed.
- **Weight Profiling:** Calculation of parameter percentages per layer (perfect for SVD/Pruning analysis).
- **Interactive Search:** Filter by layer type (e.g., `Conv1d`), dimensions (e.g., `1536`), or specific paths.
- **Path Copying:** One-click to copy the internal PyTorch path (e.g., `tower.blocks.0.linear`) for direct manipulation.
- **Standalone HTML:** Generates a single file you can share or open in any browser.

## Installation
Just copy the `generate_model_inspector` function into your project or clone this repo.

```bash
pip install torch
```

## Quick Start
```
import torch
from torch_arch_inspector import generate_model_inspector

model = YourCustomModel()
generate_model_inspector(model, output_file="model_report.html")
```

## Why use this?

Deep learning models like Transformers or genomic networks (like AlphaGenome) often have thousands of layers. Traditional print statements are unreadable, and graphical tools like Netron can be overwhelming. This tool provides a structured, hierarchical view focused on resource allocation and architectural metadata.
