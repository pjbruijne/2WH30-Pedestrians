import numpy as np
from PIL import Image
from ipywidgets import interact


def lattice_image(lattice: np.array) -> Image:
    """Turn 2d array into an image, the value of each pixel
    is determined via affine transformation of [-1, 1] -> [0, 255]."""

    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


def display_experiment(images: np.array):
    """Helper function to make the simulation widget."""

    def _show(frame=(0, len(images) - 1)):
        return images[frame]

    return interact(_show)
