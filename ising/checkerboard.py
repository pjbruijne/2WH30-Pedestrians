import numpy as np
from numba import njit, prange


SquareLattice = np.ndarray[int, int]

WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}


def unweave_checkerboard(lattice: SquareLattice) -> tuple[SquareLattice, SquareLattice]:
    """Split lattice into its black squares and white squares."""

    n, m = lattice.shape

    assert m % 2 == 0, "Cannot unweave uneven checkerboard; uneven number or columns."

    mask = np.indices((n, m)).sum(axis=0) % 2 == 0

    white_lattice = lattice[mask].reshape((n, m // 2))
    black_lattice = lattice[~mask].reshape((n, m // 2))

    return (white_lattice, black_lattice)


def weave_checkerboard(white_lattice, black_lattice) -> SquareLattice:
    """Weave together two lattices of the same size."""

    assert (
        white_lattice.shape == black_lattice.shape
    ), "Cannot weave together differnently shaped lattices; result not a rectangular lattice"

    return _weave_checkerboard(white_lattice, "w") + _weave_checkerboard(
        black_lattice, "b"
    )


def _weave_checkerboard(
    lattice: SquareLattice, square_color: SquareColor
) -> SquareLattice:
    """Create an array with double the number of columns, where the values are
     alternatingly values from the lattice and zeros, put the values of the array either
    on the WhiteSquares or the BlackSquares according to checkboard convention."""

    n, m = lattice.shape

    if square_color in ("w", "white"):
        return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 0)

    if square_color in ("b", "black"):
        return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 1)


def generate_random_array(n, m):
    """Create an array with samples from ~Unif [0, 1)."""
    return np.random.random_sample(size=(n, m))


@njit
def simulate(lattice: np.ndarray[int, int], J: float, B: float, T: float, steps: int):
    """Simulate evolution of a lattice using the checkerboard algorithm.
    This simulation algorithm is faster than the default metropolis algorithm."""

    pass
