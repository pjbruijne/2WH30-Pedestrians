import numpy as np
from numba import njit, prange

WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}


def checkerboard_weave(
    lattice: np.ndarray[int, int], square_color: SquareColor
) -> np.ndarray[int, int]:
    """Alternate an array with zeros, put the values of the array either
    on the WhiteSquares or the BlackSquares according to checkboard convention."""

    assert square_color in ("w", "white", "b", "black")

    n, m = lattice.shape

    if square_color in ("w", "white"):
        return lattice.repeat(2, axis=1) * (
            np.ones((n, 2 * m), dtype=np.int32) - np.indices((n, 2 * m)).sum(axis=0) % 2
        )

    if square_color in ("b", "black"):
        return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2)


def combine_checkerboard_lattices(
    black_lattice: np.ndarray[int, int], white_lattice: np.ndarray[int, int]
) -> np.ndarray[int, int]:
    """Combine the black_lattice and white_lattice into a single
    lattice using the checkerboard convention that top-left corner / bottom-right corner is white.
    Note that the dimensions of the black squares and the white squares need to be equal.
    """
    assert black_lattice.shape == white_lattice.shape

    return checkerboard_weave(black_lattice, "black") + checkerboard_weave(
        white_lattice, "white"
    )


def generate_random_array(n, m):
    """Create an array with samples from ~Unif [0, 1)."""
    return np.random.random_sample(size=(n, m))


@njit
def simulate(lattice: np.ndarray[int, int], J: float, B: float, T: float, steps: int):
    """Simulate evolution of a lattice using the checkerboard algorithm.
    This simulation algorithm is faster than the default metropolis algorithm."""

    pass
