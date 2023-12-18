"""Some functions that help with working with a lot of data.

In particular focussed on generating, storing and retriving simulations.
Also contains some miscellaneous functions that help with data analysis.
"""
from ising import checkerboard

from itertools import count, filterfalse
from functools import partial
import os
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt

Lattice = np.ndarray
Lattices = np.ndarray
MagnetizationData = np.ndarray


def save_as_npz(
    kind: str, array: np.ndarray, h_J: float, T: float, file_id=None
) -> None:
    """Save the array in an .npz file located at simulation-data/{kind}/h_J={h_J}/T={T}.
    The file name will be as file_id if it is provided, otherwise
    the file will be saved as {next integer that does not occur in the directory}.npz.
    """

    if kind in ("mag", "magnetization"):
        kind = "magnetization"
    elif kind in ("lattice", "lattices"):
        kind = "lattices"
    else:
        print("Not a valid kind of data! Look at my docs.")
        return

    base_path = f"simulation-data/{kind}/h_J={h_J}/T={T}"

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if file_id is None:
        names = [
            int_if_possible(f.split(".")[0])
            for f in os.listdir(base_path)
            if isfile(join(base_path, f))
        ]
        file_id = str(next(filterfalse(set(names).__contains__, count(1))))

    data = {kind: array}

    np.savez(join(base_path, file_id), **data)
    print("Saved succesfully!")


def int_if_possible(x):
    """Save version of casting to int.
    Returns nothing if x is not readable as int."""
    try:
        return int(x)
    except ValueError:
        return


def load_from_npz(kind: str, h_J: float, T: float):
    """Load all data files stored in simulation-data/{kind}/h_J={h_J}/T={T} into a dictionary."""

    if kind in ("mag", "magnetization"):
        kind = "magnetization"
    elif kind in ("lattice", "lattices"):
        kind = "lattices"
    else:
        print("Not a valid kind of data! Look at my docs.")
        return

    base_path = f"simulation-data/{kind}/h_J={h_J}/T={T}"

    names = [f for f in os.listdir(base_path) if isfile(join(base_path, f))]

    return {f"a{name.split('.')[0]}": np.load(join(base_path, name)) for name in names}


load_lattices_data = partial(load_from_npz, "lattices")
load_magnetization_data = partial(load_from_npz, "magnetization")
save_lattices_data = partial(save_as_npz, "lattices")
save_magnetization_data = partial(save_as_npz, "magnetization")


def magnetization_from_lattices(lattices: Lattices) -> MagnetizationData:
    """Compute the average magnetization for each lattice in an array of lattices."""

    return np.average(lattices, axis=(1, 2))


def average_upto_index(array: np.array) -> np.array:
    """Returns an array where each element i is equal to the
    average value of the subarray indexed by [:i+1]."""

    return np.cumsum(array) / np.arange(1, len(array) + 1)


def moving_average(array: np.array, n: int):
    """Computes the moving average with window size n of an array,
    note that the first n elements of the returned array are not
    averages of last n values."""

    assert len(array) > n, "Averaging window too large"

    ret = np.cumsum(array, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    partial_window = ret[: n - 1] / np.arange(1, n)
    full_window = ret[n - 1 :] / n
    return np.concatenate((partial_window, full_window), axis=0)


def constant_harvester(
    lattice: Lattice, Ts: np.ndarray, h_J: float, iterations: int, file_id: str
):
    """For every value of T in Ts, simulate using checkerboard fixed
    number of iterations. Save both lattices and magnetization data in simulation-data
    using 100{file_id}."""

    file_id = "100" + file_id

    for T in Ts:
        lattices = checkerboard.simulate(
            lattice=lattice.copy(), h_J=h_J, T=T, steps=iterations
        )

        mag = magnetization_from_lattices(lattices=lattices)

        save_lattices_data(array=lattices, h_J=h_J, T=T, file_id=file_id)
        save_magnetization_data(array=mag, h_J=h_J, T=T, file_id=file_id)

    print("Done harvesting!")


def magnetization_convergence(magnetization: MagnetizationData, h_J: float, T: float):
    """Plot magnetization together with some smoothened versions of it.
    This function can help to see if and how the magnetization converges
    during the simulation process."""

    import ising.plot_styling

    (l,) = magnetization.shape

    plt.ylim(-1.1, 1.1)

    plt.plot(magnetization)
    plt.plot(moving_average(magnetization, l // 10))
    plt.plot(average_upto_index(magnetization))

    plt.legend(["Magnetization", "Moving avg magnetization", "Avg upto index"])

    plt.ylabel("Magnetization $(M/N)$")
    plt.xlabel("Iteration $(frame)$")
    plt.title(f"$h/J={h_J}, T={T}$")

    return plt.gca()
