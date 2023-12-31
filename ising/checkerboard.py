from typing import Generator
import numpy as np
from scipy.ndimage import convolve


AdjustedMagneticField = np.ndarray[int, int] | float
Lattice = np.ndarray
Lattices = np.ndarray
WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}


def unweave_checkerboard(lattice: Lattice) -> tuple[Lattice, Lattice]:
    """Split lattice into its white squares and black squares,
    according to the checkerboard convention."""

    n, m = lattice.shape

    assert m % 2 == 0, "Cannot unweave uneven checkerboard; uneven number of columns."

    mask = np.indices((n, m)).sum(axis=0) % 2 == 0

    white_lattice = lattice[mask].reshape((n, m // 2))
    black_lattice = lattice[~mask].reshape((n, m // 2))

    return (white_lattice, black_lattice)


def _weave_checkerboard(lattice: Lattice, square_color: SquareColor) -> Lattice:
    """Create an array with double the number of columns, where the values are
     alternatingly values from the lattice and zeros, put the values of the array either
    on the WhiteSquares or the BlackSquares according to checkboard convention."""

    n, m = lattice.shape

    if square_color in ("w", "white"):
        return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 0)

    if square_color in ("b", "black"):
        return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 1)


def generate_random_array(n: int, m: int) -> Lattice:
    """Create an n by m array with samples from ~Unif [0, 1)."""

    return np.random.random_sample((n, m))


def neighbour_sum_square(lattice: Lattice) -> Lattice:
    """Compute the neighboursum of each spin in a square lattice."""

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    return convolve(lattice, kernel, mode="wrap")


def hamiltonian(lattice: Lattice, h_J: float) -> Lattice:
    """Compute the energy contribution of each spin in a lattice."""

    return 2 * (neighbour_sum_square(lattice) + h_J) * lattice


def simulate(
    lattice: Lattice,
    h_J: AdjustedMagneticField,
    T: float,
    steps: int,
) -> Lattices:
    """Simulate evolution of a lattice using the checkerboard algorithm.
    This simulation algorithm converges faster than the default metropolis algorithm."""

    assert steps % 2 == 0, "Even number of steps required"

    n, m = lattice.shape
    lattices = np.empty((steps, n, m), dtype=np.int8)
    lattices[0] = lattice.copy()

    for i in range(1, steps - 1, 2):
        # update the white squares
        dE_white, _ = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_white / T)
        mask = _weave_checkerboard(mask, "white")

        lattice[mask] = -lattice[mask]
        lattices[i] = lattice.copy()

        # update the black squares
        _, dE_black = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_black / T)
        mask = _weave_checkerboard(mask, "black")

        lattice[mask] = -lattice[mask]
        lattices[i + 1] = lattice.copy()

    dE_white, _ = unweave_checkerboard(hamiltonian(lattice, h_J))

    mask = generate_random_array(n, m // 2) < np.exp(-dE_white / T)
    mask = _weave_checkerboard(mask, "white")

    lattice[mask] = -lattice[mask]
    lattices[steps-1] = lattice.copy()

    return lattices


def simulation_frame_generator(
    lattice: Lattice,
    h_J: AdjustedMagneticField,
    T: float,
) -> Generator[Lattice, None, None]:
    """Generator that evolves a lattice to equillibrium and yields each frame
    in the process."""

    n, m = lattice.shape

    while True:
        # update the white squares
        dE_white, _ = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_white / T)
        mask = _weave_checkerboard(mask, "white")

        lattice[mask] = -lattice[mask]
        yield lattice.copy()

        # update the black squares
        _, dE_black = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_black / T)
        mask = _weave_checkerboard(mask, "black")

        lattice[mask] = -lattice[mask]
        yield lattice.copy()


# The graveyard of unused functions

# def weave_checkerboard(white_lattice: Lattice, black_lattice: Lattice) -> Lattice:
#     """Weave together two lattices (of the same size) into their combined checkerboard lattice."""

#     assert (
#         white_lattice.shape == black_lattice.shape
#     ), "Cannot weave together differnently shaped lattices; result not a rectangular lattice."

#     return _weave_checkerboard(white_lattice, "w") + _weave_checkerboard(
#         black_lattice, "b"
#     )
