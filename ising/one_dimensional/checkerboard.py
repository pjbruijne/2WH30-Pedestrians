import numpy as np


def unweave_checkerboard(lattice):
    (l,) = lattice.shape

    assert l % 2 == 0, "Cannot unweave uneven checkerboard; uneven length"

    white_lattice = lattice[::2]
    black_lattice = lattice[1::2]

    return (white_lattice, black_lattice)


def _weave_checkerboard(lattice, square_color):
    (l,) = lattice.shape

    if square_color in ("w", "white"):
        return lattice.repeat(2) * (np.arange(2 * l) % 2 == 0)

    if square_color in ("b", "black"):
        return lattice.repeat(2) * (np.arange(2 * l) % 2 == 1)


def generate_random_array(l):
    return np.random.random_sample((l,))


def neighbour_sum(lattice):
    return lattice + np.roll(lattice, 1)


def hamiltonian_constant_J(lattice, J, h):
    return 2 * (J * neighbour_sum(lattice) + h) * lattice


def simulate(lattice, J, h, T, steps):
    if steps % 2 == 1:
        steps += 1

    (l,) = lattice.shape
    lattices = np.empty((steps + 1, l))
    lattices[0] = lattice.copy()

    for i in range(1, steps + 1, 2):
        # update the white squares
        dE_white, _ = unweave_checkerboard(hamiltonian_constant_J(lattice, J, h))

        mask = generate_random_array(l // 2) < np.exp(-dE_white / T)
        mask = _weave_checkerboard(mask, "white")

        lattice[mask] = -lattice[mask]
        lattices[i] = lattice.copy()

        # update the black squares
        _, dE_black = unweave_checkerboard(hamiltonian_constant_J(lattice, J, h))

        mask = generate_random_array(l // 2) < np.exp(-dE_black / T)
        mask = _weave_checkerboard(mask, "black")

        lattice[mask] = -lattice[mask]
        lattices[i + 1] = lattice.copy()

    return lattices
