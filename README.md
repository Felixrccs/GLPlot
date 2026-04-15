# GLPlot

High-performance, GPU-accelerated plotting library in Python, designed to handle **millions** of lines effortlessly. It provides an API similar to Matplotlib but runs natively over an OpenGL/GLFW backend, performing instanced rendering directly on the GPU.

## Features
- **Matplotlib API Compatibility**: Familiar `plot(x, y)`, `show()`, `savefig()`.
- **Phase Diagram Optimized (`plot_lines`)**: Explicitly supports passing millions of line parameters $(a, b)$ to calculate functions $y = ax + b$ securely bounded to bounds using shader math.
- **Logarithmic Density Heaps**: By displaying overlaps, `density=True` handles millions of parallel curves seamlessly for heatmaps.
- **Dynamic Camera**: Drag to pan, scroll to zoom with on-the-fly resolution subsampling.

## Installation

You can install this locally:

```bash
pip install .
```

Requirements: `numpy`, `glfw`, `PyOpenGL`, `scipy`, `matplotlib`.

## Usage

**Traditional Polylines**
```python
import numpy as np
import glplot.pyplot as gplt

x = np.linspace(0, 10, 100)
y = np.sin(x)

gplt.figure("Sine Wave")
gplt.plot(x, y, color=(1.0, 0.0, 0.0, 1.0))
gplt.show()
```

**Bulk Lines (Density Map)**
```python
import numpy as np
import glplot.pyplot as gplt

N = 1000000
a = np.random.randn(N)
b = np.random.randn(N)

gplt.figure("Density")
gplt.plot_lines(a, b, x_range=(-2, 2))
gplt.show(density=True)
```

## Testing

Uses `pytest` covering 100% of the internal application logic without popping visible windows:
```bash
pytest
```
