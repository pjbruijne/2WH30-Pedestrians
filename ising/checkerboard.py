import numpy as np
from scipy.ndimage import convolve


InteractionStrength = float
Temperature = float
MagneticField = np.ndarray[int, int] | float
SquareLattice = np.ndarray[int, int]
WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}


def unweave_checkerboard(lattice: SquareLattice) -> tuple[SquareLattice, SquareLattice]:
    """Split lattice into its white squares and black squares according to the checkerboard convention."""

    n, m = lattice.shape

    assert m % 2 == 0, "Cannot unweave uneven checkerboard; uneven number or columns."

    mask = np.indices((n, m)).sum(axis=0) % 2 == 0

    white_lattice = lattice[mask].reshape((n, m // 2))
    black_lattice = lattice[~mask].reshape((n, m // 2))

    return (white_lattice, black_lattice)


def weave_checkerboard(white_lattice, black_lattice) -> SquareLattice:
    """Weave together two lattices (of the same size) into their combined checkerboard lattice."""

    assert (
        white_lattice.shape == black_lattice.shape
    ), "Cannot weave together differnently shaped lattices; result not a rectangular lattice."

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
    """Create an n by m array with samples from ~Unif [0, 1)."""
    return np.random.random_sample((n, m))


def neighbour_sum_square(lattice: SquareLattice) -> SquareLattice:
    """Compute the square neighboursum for the entire square lattice."""

    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    return convolve(lattice, kernel, mode="wrap")


def hamiltonian_constant_J(lattice, J, h):
    """Compute the energy of a lattice."""

    return 2 * (J * neighbour_sum_square(lattice) + h) * lattice


def simulate(
    lattice: SquareLattice,
    J: InteractionStrength,
    h: MagneticField,
    T: Temperature,
    steps: int,
):
    """Simulate evolution of a lattice using the checkerboard algorithm.
    This simulation algorithm converges faster than the default metropolis algorithm."""

    if steps % 2 == 1:
        steps += 1

    n, m = lattice.shape
    lattices = np.empty((steps + 1, n, m))
    lattices[0] = lattice.copy()

    for i in range(1, steps + 1, 2):
        # update the white squares
        dE_white, _ = unweave_checkerboard(hamiltonian_constant_J(lattice, J, h))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_white / T)
        mask = _weave_checkerboard(mask, "white")

        lattice[mask] = -lattice[mask]
        lattices[i] = lattice.copy()

        # update the black squares
        _, dE_black = unweave_checkerboard(hamiltonian_constant_J(lattice, J, h))

        mask = generate_random_array(n, m // 2) < np.exp(-dE_black / T)
        mask = _weave_checkerboard(mask, "black")

        lattice[mask] = -lattice[mask]
        lattices[i + 1] = lattice.copy()

    return lattices
