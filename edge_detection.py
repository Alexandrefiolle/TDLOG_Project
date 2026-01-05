"""This file contains functions to perform edge detection on grayscale images.
It computes the gradient magnitude, smooths it with a Gaussian filter, 
and generates an edge weight map for use in pathfinding algorithms."""

import numpy as np
from scipy.ndimage import gaussian_filter
from math import sqrt
import point_class as pc
import manipulation as ui

# Default parameters
EDGE_EPSILON = 0.1      # ε to avoid division by zero in weight map
GAUSSIAN_SIGMA = 1.0    # σ for Gaussian smoothing


def compute_gradient_magnitude(grey_img: ui.GreyImage) -> np.ndarray:
    """
    Point 1 : Compute the gradient magnitude image |∇f| (image IG in the subject)
    Uses a simple 3x3 Sobel gradient. See https://fr.wikipedia.org/wiki/Filtre_de_Sobel
    Returns a 2D ndarray of the same size as the image, with floats.
    """
    arr = grey_img.image # shape (height, width), values 0..255
    arr = arr.astype(float)

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)

    grad_x = np.zeros_like(arr)
    grad_y = np.zeros_like(arr)

    h, w = arr.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            grad_x[y, x] = np.sum(arr[y-1:y+2, x-1:x+2] * sobel_x)
            grad_y[y, x] = np.sum(arr[y-1:y+2, x-1:x+2] * sobel_y)

    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return magnitude 


def smooth_gradient_magnitude(magnitude: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """
    Point 2: Smooth the gradient magnitude image with a Gaussian filter G_σ.
    Returns the smoothed image I* = G_σ * |∇f|.
    """
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    return smoothed


def compute_edge_weight_map(smoothed_magnitude: np.ndarray, epsilon: float = EDGE_EPSILON) -> np.ndarray:
    """
    Point 3 : Computes the edge weight map W(x,y) = 1 / (ε + G_σ * |∇f|(x,y))
    The higher the value, the more we want to pass through this pixel (strong edge).
    """
    weight_map = 1.0 / (epsilon + smoothed_magnitude)
    return weight_map

def cost_edges_edge_detection(epsilon: float, neighbor: pc.Point, dist: dict[pc.Point, float], 
                             grey_img: ui.GreyImage, weight_map: np.ndarray) -> float:
    """
    Cost function to be used in distances_costs when edge_detection=True.
    """
    w = weight_map[neighbor.y, neighbor.x]
    if w <= 0:
        return 1e10 
    return w


"""
def demo_edge_weight_map(grey_img: ui.GreyImage, sigma: float = GAUSSIAN_SIGMA, epsilon: float = EDGE_EPSILON) -> np.ndarray:
    
    import interface as vis

    magnitude = compute_gradient_magnitude(grey_img)
    smoothed = smooth_gradient_magnitude(magnitude, sigma)
    weight_map = compute_edge_weight_map(smoothed, epsilon)

    # Visualisation
    mag_vis = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)
    smooth_vis = np.clip(smoothed / smoothed.max() * 255, 0, 255).astype(np.uint8) if smoothed.max() > 0 else smoothed.astype(np.uint8)
    weight_vis = np.clip(weight_map / weight_map.max() * 255, 0, 255).astype(np.uint8) if weight_map.max() > 0 else weight_map.astype(np.uint8)

    img_mag = ui.Image.fromarray(mag_vis, mode='L')
    img_smooth = ui.Image.fromarray(smooth_vis, mode='L')
    img_weight = ui.Image.fromarray(weight_vis, mode='L')

    img_mag.show(title="Gradient Magnitude |∇f|")
    img_smooth.show(title="Smoothed Gradient Gσ * |∇f|")
    img_weight.show(title="Weight Map W(x,y)")

    return weight_map  # Return the weight map for further use in pathfinding
"""