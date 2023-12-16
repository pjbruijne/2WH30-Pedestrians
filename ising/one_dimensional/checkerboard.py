import numpy as np

AdjustedMagneticField = np.ndarray[int] | float
LineLattice = np.ndarray
Lattices = np.ndarray
WhiteSquares = str
BlackSquares = str
SquareColor = {WhiteSquares, BlackSquares}


def unweave_checkerboard(line_lattice: LineLattice) -> tuple[LineLattice, LineLattice]:
    """Partition line lattice into two line lattices, consisting of alternating
    elements from the original line lattice, analogous to the checkerboard algo in 2D.
    The 0th spin is part of the white spins and the 1st spin is part of the black spins.
    The first element from the returned tuple are the white spins and the second
    element are the black spins."""

    (l,) = line_lattice.shape

    assert l % 2 == 0, "Cannot unweave uneven checkerboard; uneven length"

    white_lattice = line_lattice[::2]
    black_lattice = line_lattice[1::2]

    return (white_lattice, black_lattice)


def _weave_checkerboard(line_lattice: LineLattice, square_color: SquareColor):
    """Transform a line lattice such that has alternating spins and zeroes.
    The spins are placed according to the 1D color analogy of a checkerboard."""

    (l,) = line_lattice.shape

    if square_color in ("w", "white"):
        return line_lattice.repeat(2) * (np.arange(2 * l) % 2 == 0)

    if square_color in ("b", "black"):
        return line_lattice.repeat(2) * (np.arange(2 * l) % 2 == 1)


def generate_random_array(l: int) -> LineLattice:
    """Create a line lattice of size l with samples from ~Unif [0, 1)."""

    return np.random.random_sample((l,))


def neighbour_sum(line_lattice: LineLattice) -> LineLattice:
    """Compute the neighbour sum for each spin in a line lattice."""

    return line_lattice + np.roll(line_lattice, 1)


def hamiltonian(line_lattice: LineLattice, h_J: float) -> LineLattice:
    """Compute the energy contribution for each spin in a line lattice."""

    return 2 * (neighbour_sum(line_lattice) + h_J) * line_lattice


def simulate(lattice: LineLattice, h_J: float, T: float, steps: int):
    """Evolve the line lattice a given number of steps
    using the 1D variant of the line lattice."""

    if steps % 2 == 1:
        steps += 1

    (l,) = lattice.shape
    lattices = np.empty((steps + 1, l))
    lattices[0] = lattice.copy()

    for i in range(1, steps + 1, 2):
        # update the white squares
        dE_white, _ = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(l // 2) < np.exp(-dE_white / T)
        mask = _weave_checkerboard(mask, "white")

        lattice[mask] = -lattice[mask]
        lattices[i] = lattice.copy()

        # update the black squares
        _, dE_black = unweave_checkerboard(hamiltonian(lattice, h_J))

        mask = generate_random_array(l // 2) < np.exp(-dE_black / T)
        mask = _weave_checkerboard(mask, "black")

        lattice[mask] = -lattice[mask]
        lattices[i + 1] = lattice.copy()

    return lattices
