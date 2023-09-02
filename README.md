# Differentiable Spline Curve Interpolation for Neural Networks

This repository contains a PyTorch implementation of a differentiable spline curve interpolation method that can be seamlessly integrated into neural networks for various applications. The code allows you to create smooth spline curves that can be rendered during the backpropagation step, making it suitable for tasks like image generation, data smoothing, and more.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Spline Interpolation](#spline-interpolation)
  - [Generating Spline Stroke Brushes](#generating-spline-stroke-brushes)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Spline interpolation is a widely used technique to create smooth curves that pass through a set of control points. This code provides a differentiable spline interpolation method that can be incorporated into neural networks, allowing you to optimize the control points while considering gradients for various applications.

Key features:
- Differentiable spline interpolation: The code is designed to work seamlessly with PyTorch, enabling you to use it within neural network architectures and optimize spline curves during training.
- Spline stroke brushes: You can generate smooth spline stroke brushes, which are commonly used in image generation tasks and artistic applications.
- Customizable: You can adjust the control points, brush radius, and canvas size to suit your specific requirements.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch numpy matplotlib
```

## Usage

### Spline Interpolation

The `spline_interpolation` function allows you to perform spline interpolation for a given set of control points. Here's how to use it:

```python
import torch
import numpy as np
from spline_interpolation import spline_interpolation

# Define control points (x and y coordinates)
x = [0, 2, 5, 7, 8]
y = [3, 2, 0, 2, 0]

# Point at which you want to interpolate the spline
t = 4.0

# Perform spline interpolation
result = spline_interpolation(x, y, t)
print(result)
```

### Generating Spline Stroke Brushes

The `get_spline_stroke_brush` function allows you to generate spline stroke brushes, which can be useful for image generation or artistic applications. Here's how to use it:

```python
import torch
from spline_interpolation import get_spline_stroke_brush

# Define control points (x and y coordinates) for multiple brushes
sx = torch.tensor([[0, 2, 5, 7, 8], [1, 3, 4, 6, 9]], dtype=torch.float32)
sy = torch.tensor([[3, 2, 0, 2, 0], [4, 2, 1, 3, 1]], dtype=torch.float32)

# Specify the brush radius for each brush
brush_radius = [5, 3]  # List of brush radii

# Generate spline stroke brushes
brushes = []
for x, y, r in zip(sx, sy, brush_radius):
    brush = get_spline_stroke_brush(x, y, r)
    brushes.append(brush)

# Display the generated brushes (for demonstration purposes)
import matplotlib.pyplot as plt
for brush in brushes:
    plt.imshow(brush)
    plt.show()
```

## Examples

For more detailed usage examples and application scenarios, please refer to the [example notebooks]([examples/](https://github.com/notabee/differentiable-spline-curve/blob/main/differentiable-rendered-splines-pytorch.ipynb)) included in this repository.

## Contributing

Contributions to this project are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request. 

## License

This project is licensed under the APACHE License.
