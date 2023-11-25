import numpy as np
from PIL import Image
from ipywidgets import interact
import cv2


def lattice_to_image(lattice: np.array) -> Image:
    """Turn 2d array into an image, the value of each pixel
    is determined via affine transformation of [-1, 1] -> [0, 255]."""

    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


def lattices_to_images(lattices: [np.array]) -> [Image]:
    """Turn 2d array into an image, the value of each pixel
    is determined via affine transformation of [-1, 1] -> [0, 255]."""

    return [lattice_to_image(lattice) for lattice in lattices]


def display_experiment(lattices: np.array):
    """Helper function to make the simulation widget."""

    def _show(frame=(0, len(lattices) - 1)):
        return lattices[frame]

    return interact(_show)


def upscale_lattice(lattice: np.array, scale_factor: int) -> np.array:
    """Upscale array by integer multiple in all directions."""

    return lattice.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)


def upscale_lattices(lattices: [np.array], scale_factor: int) -> [np.array]:
    """Upscale list of arrays, each array gets upscaled by an integer
    multiple in all directions."""

    return [upscale_lattice(lattice, scale_factor) for lattice in lattices]


def save_gif(images: [Image], filename: str) -> None:
    """Save simulation as a GIF."""

    images[0].save(
        f"simulation-results/{filename}.gif",
        save_all=True,
        optimize=False,
        append_images=images[1:],
        loop=0,
    )
