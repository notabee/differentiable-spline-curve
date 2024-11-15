# Differentiable Spline Renderer

A Python implementation for rendering smooth, differentiable splines using PyTorch, designed for use in machine learning and generative art tasks. This repository demonstrates the creation of quadratic Bézier curves and their visualization on a canvas with customizable colors and brush sizes.

## Features

- **Smooth spline generation**: Uses quadratic Bézier interpolation to produce smooth curves.
- **Customizable rendering**: Specify brush sizes and colors for spline rendering.
- **Differentiability**: Built with PyTorch, enabling gradient-based optimization.
- **Visualization**: Render and display the resulting canvas with `matplotlib`.

---

## Installation

```bash
git clone https://github.com/your-username/differentiable-spline-renderer.git
cd differentiable-spline-renderer
```

Ensure you have PyTorch and Matplotlib installed in your environment.

---

## Usage

### 1. Define Control Points
The spline requires **exactly three control points** for quadratic Bézier interpolation. Example:

```python
import torch
from differentiable_spline import DifferentiableSpline

control_points = torch.tensor([
    [0.2, 0.3],
    [0.5, 0.8],
    [0.8, 0.4]
], dtype=torch.float32)

spline = DifferentiableSpline(control_points, canvas_size=(128, 128), device='cpu')
```

### 2. Interpolate and Render
Specify colors and brush sizes, then generate the canvas:

```python
colors = torch.tensor([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0]   # Blue
])

brush_sizes = torch.tensor([10.0])  # Single brush size for all strokes

canvas = spline.generate_spline_canvas(colors, brush_sizes, num_points=300)
spline.show_canvas(canvas)
```

---

## API Reference

### `DifferentiableSpline`

#### Constructor
```python
DifferentiableSpline(control_points, canvas_size=(128, 128), device='cpu')
```
- **control_points**: `torch.Tensor` of shape `(3, 2)`. Control points for the quadratic Bézier curve.
- **canvas_size**: Tuple of integers specifying the canvas dimensions.
- **device**: PyTorch device (`'cpu'` or `'cuda'`).

#### Methods
- **`interpolate_points(num_points=100)`**: 
  Interpolates `num_points` along the Bézier curve.
- **`generate_spline_canvas(colors, brush_sizes, num_points=300, test_mode=False)`**: 
  Renders the spline on a canvas.
- **`show_canvas(canvas)`**: 
  Visualizes the generated canvas using `matplotlib`.

---

## Example Output

Below is a generated spline rendered with smooth blending:

![Spline Example]([example.png](https://github.com/notabee/differentiable-spline-curve/blob/main/splines.png))

---

## Contributions

Feel free to submit issues or pull requests to enhance functionality or fix bugs.

---

## License

This project is licensed under the [MIT License](LICENSE).
