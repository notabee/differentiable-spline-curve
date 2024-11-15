import torch
import matplotlib.pyplot as plt

class DifferentiableSpline:
    def __init__(self, control_points, canvas_size=(128, 128), device='cpu'):
        self.device = device
        self.canvas_size = canvas_size
        self.control_points = control_points.to(device).detach().requires_grad_(True)

    def interpolate_points(self, num_points=100):
        t_values = torch.linspace(0, 1, num_points, device=self.device, requires_grad=True)

        # Ensure exactly 3 control points for Quadratic Bezier curve
        if len(self.control_points) != 3:
            raise ValueError("Need exactly 3 control points for Quadratic Bezier spline interpolation")

        # Define control points for the Quadratic Bezier curve
        p0, p1, p2 = self.control_points

        # Use Quadratic Bezier formula
        points = []
        for t in t_values:
            # Quadratic Bezier formula
            interpolated_point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            points.append(interpolated_point)

        interpolated_points = torch.stack(points) * torch.tensor(self.canvas_size, device=self.device)
        return interpolated_points

    def generate_spline_canvas(self, colors, brush_sizes, num_points=300, test_mode=False):
        colors = colors.to(self.device)
        brush_sizes = brush_sizes.to(self.device)  # Adjust brush size multiplier for more presence

        canvas = torch.zeros((1, 3, *self.canvas_size), device=self.device, requires_grad=True)
        points = self.interpolate_points(num_points=num_points)

        for i, point in enumerate(points):
            X, Y = torch.meshgrid(torch.arange(self.canvas_size[0], device=self.device),
                                  torch.arange(self.canvas_size[1], device=self.device))
            dist_squared = (X - point[0]) ** 2 + (Y - point[1]) ** 2
            gaussian_mask = torch.exp(-dist_squared / (0.005 * brush_sizes[0] ** 2))  # Sharper strokes

            if test_mode:
                print(f"Dist Squared Min: {dist_squared.min().item()}, Max: {dist_squared.max().item()}")
                print(f"Gaussian Mask Min: {gaussian_mask.min().item()}, Max: {gaussian_mask.max().item()}")

            color = colors[i % len(colors)]
            colored_brush = gaussian_mask * color.view(3, 1, 1)

            # Instead of blending, take the max at each pixel to layer the strokes
            canvas = torch.max(canvas, colored_brush.unsqueeze(0))

        if test_mode:
            print(f"Canvas Min: {canvas.min().item()}, Max: {canvas.max().item()}")
        return torch.clamp(canvas, 0, 1)  # Clamp final result to [0, 1]

    def show_canvas(self, canvas):
        canvas_np = canvas.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(canvas_np)
        plt.axis('off')
        plt.show()
