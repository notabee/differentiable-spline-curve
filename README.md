# Differentiable Spline Curve Renderer for Neural Networks

This repository contains a PyTorch implementation of a differentiable spline curve renderer that seamlessly integrates into neural networks for various applications. The code allows you to create smooth spline curves that can be rendered during the backpropagation step, making it suitable for tasks like image generation, data smoothing, and more.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Spline Interpolation](#spline-interpolation)
  - [Rendering Spline Strokes](#rendering-spline-strokes)
  - [Batch Rendering](#batch-rendering)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Spline interpolation is a widely used technique to create smooth curves that pass through a set of control points. This code provides a differentiable spline interpolation method that can be incorporated into neural networks, allowing you to optimize the control points while considering gradients for various applications.

Key features:
- Differentiable spline interpolation: The code is designed to work seamlessly with PyTorch, enabling you to use it within neural network architectures and optimize spline curves during training.
- Spline stroke rendering: You can generate smooth spline stroke brushes, which are commonly used in image generation tasks and artistic applications.
- Batch rendering: Render splines in batches, allowing for efficient processing in parallel.

## Requirements

Before you begin, ensure you have the following requirements installed:

- Python 3.x
- PyTorch 1.7.1
- NumPy
- Matplotlib
- Conda

tl;dr
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
the above requirements should be sufficient.

## Installation

1. Create a conda environment from the provided `spline.yml` file:

```bash
conda env create -f spline.yml
conda activate spline-env
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

### Rendering Spline Strokes

The `get_spline_stroke_brush` function allows you to generate spline stroke brushes, which can be useful for image generation or artistic applications. Here's how to use it:

```python
import torch
from spline_interpolation import get_spline_stroke_brush

# Define control points (x and y coordinates) for a single brush
sx = torch.tensor([0, 2, 5, 7, 8], dtype=torch.float32)
sy = torch.tensor([3, 2, 0, 2, 0], dtype=torch.float32)

# Specify the brush radius
brush_radius = 5

# Generate a spline stroke brush
brush = get_spline_stroke_brush(sx, sy, brush_radius)

# Display the generated brush (for demonstration purposes)
import matplotlib.pyplot as plt
plt.imshow(brush)
plt.show()
```

### Batch Rendering

You can render splines in batches to efficiently process multiple splines in parallel. For example, you can render splines for multiple data points simultaneously.

```python
import torch
from spline_interpolation import get_spline_stroke_brush

# Define control points (x and y coordinates) for multiple brushes in a batch
sx = torch.tensor([[0, 2, 5, 7, 8], [1, 3, 4, 6, 9]], dtype=torch.float32)
sy = torch.tensor([[3, 2, 0, 2, 0], [4, 2, 1, 3, 1]], dtype=torch.float32)

# Specify the brush radius for each brush in the batch
brush_radius = [5, 3]  # List of brush radii

# Generate spline stroke brushes for the batch
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

This project is licensed under the Apache License.
