import os
import shutil

import numpy as np
from PIL import Image
from ipywidgets import interact


def lattice_to_image(lattice: np.ndarray[int,int]) -> Image.Image:
    """Turn 2d array into an image, the value of each pixel
    is determined via affine transformation of [-1, 1] -> [0, 255]."""

    return Image.fromarray(np.uint8((lattice + 1) * 0.5 * 255))


def lattices_to_images(lattices: list[np.ndarray[int,int]]) -> list[Image.Image]:
    """Turn 2d array into an image, the value of each pixel
    is determined via affine transformation of [-1, 1] -> [0, 255]."""

    return [lattice_to_image(lattices[i]) for i in range(len(lattices))]


def display_experiment(lattices: list[Image.Image]):
    """Helper function to make the simulation widget."""

    def _show(frame=(0, len(lattices) - 1)):
        return lattices[frame]

    return interact(_show)


def upscale_lattice(lattice: np.ndarray[int,int], scale_factor: int) -> np.ndarray[int,int]:
    """Upscale array by integer multiple in all directions."""

    return lattice.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)


def upscale_lattices(lattices: list[np.array], scale_factor: int) -> list[np.ndarray[int,int]]:
    """Upscale list of arrays, each array gets upscaled by an integer
    multiple in all directions."""

    return [upscale_lattice(lattice, scale_factor) for lattice in lattices]


def save_gif(images: list[Image.Image], filename: str) -> None:
    """Save simulation as a GIF."""

    images[0].save(
        f"simulation-results/{filename}.gif",
        save_all=True,
        optimize=False,
        append_images=images[1:],
        loop=0,
    )


def split_gif(filename: str, num_key_frames: int = None) -> None:
    """Split gif loicated at {filename}.gif into separate images.
    A folder with the same name gets created in the same directory."""

    with Image.open(f"{filename}.gif") as im:
        if num_key_frames is None:
            num_key_frames = im.n_frames

        if os.path.exists(filename):
            shutil.rmtree(filename)
        os.makedirs(filename)

        for i in range(num_key_frames):
            im.seek(im.n_frames // num_key_frames * i)
            im.save(f"{filename}/im-{i}.png")
