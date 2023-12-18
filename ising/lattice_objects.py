import os
import shutil

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from numba import njit
from PIL import Image
from scipy.ndimage import convolve
#from ipywidgets import interact


class lattice(ABC):
    
    @njit()
    @abstractmethod
    def neighbor_sum(self, X:tuple[int,...]) -> int:
        """Calculates the sum of the neighbours of the spin at position (x1,x2,...). Implementation depends on subclass."""
        pass
    
    @njit()
    @abstractmethod
    def simulate(self, J: float, B: float, T: float, steps: int, method: Literal["metropolis","checkerboard"] | None = "metropolis"):
        """Simulate ising model using the Metropolis algorithm.
        J : global spin-spin interaction constant,
        B : external magnetic field,
        T : global temperature,
        steps : total amount of iterations before the method terminates"""
        pass
    
    @abstractmethod
    def lattice_to_image(self, matrix: np.ndarray[int,int], scale_factor: int = 1) -> Image.Image:
        """Turn 2d array into an image, the value of each pixel
        is determined via affine transformation of [-1, 1] -> [0, 255]. Is dependent on implementation of subclass."""
        pass

    @abstractmethod
    def lattices_to_images(self, scale_factor: int = 1) -> list[Image.Image]:
        """Turn list of 2d simulation steps into an image, the value of each pixel
        is determined via affine transformation of [-1, 1] -> [0, 255]."""
        pass
    
    @abstractmethod
    def upscale_lattice(self, matrix: np.ndarray[int,int], scale_factor: int) -> np.ndarray[int,int]:
        """Upscale array by integer multiple in all directions."""
        pass

    def upscale_lattices(self, matrices: list[np.ndarray[int,int]], scale_factor: int) -> list[np.ndarray[int,int]]:
        """Upscale list of arrays, each array gets upscaled by an integer
        multiple in all directions."""

        return [self.upscale_lattice(matrix, scale_factor) for matrix in matrices]
    
    @njit()
    @abstractmethod
    def repeated_simulations(self, J: float, B: float, T: float, steps: int, runs: int, method: Literal["metropolis","checkerboard"] | None = "metropolis") -> list[np.ndarray[int,int]]:
        pass
    
class lattice2d(lattice):
    
    def __init__(self, width: int, height: int, states: list[int] = [-1,1]) -> None:
        
        self.matrix:np.ndarray[int,int] = np.random.choice(states, size=(width, height))
        self.initialmatrix = self.matrix.copy()
        self.simulation:list[np.ndarray[int,int]] = []
        self.simulation_finals:list[np.ndarray[int,int]] = []
    
    def simulate(self, J: float, B: float, T: float, steps: int, method: Literal["metropolis","checkerboard"] | None = "metropolis") -> list[np.ndarray[int,int]]:
        if method == "metropolis":
            return self.__metropolis(J,B,T,steps)
        if method == "checkerboard":
            return self.__checkerboard(J,B,T,steps)

    def __metropolis(self, J, B, T, steps) -> list[np.ndarray[int,int]]:
        self.simulation.clear()
        self.simulation.append(self.initialmatrix)
        self.matrix = self.initialmatrix.copy()
        width, height = self.matrix.shape

        for _ in range(steps):
            x = np.random.randint(width)
            y = np.random.randint(height)

            spin:int = self.matrix[x, y]

            position = (x,y)
            neighbour_sum = self.neighbor_sum(position)

            dE = 2 * spin * (J * neighbour_sum - B)

            if (neighbour_sum * spin < 0) or (np.random.random(1) < np.exp(-dE / T)):
                self.matrix[x, y] = -self.matrix[x, y]

            self.simulation.append(self.matrix.copy())
        return self.simulation

    def __checkerboard(self, J, B, T, steps) -> list[np.ndarray[int,int]]:
        """Simulate evolution of a lattice using the checkerboard algorithm.
        This simulation algorithm converges faster than the default metropolis algorithm."""

        if steps % 2 == 1:
            steps += 1

        n, m = self.matrix.shape
        self.simulation.clear()
        self.matrix = self.initialmatrix.copy()
        self.simulation.append(self.matrix)

        for i in range(1, steps + 1, 2):
            # update the white squares
            dE_white, _ = self.__unweave_checkerboard(self.__hamiltonian_constant_J(self.matrix, J, B))

            mask = self.__generate_random_array(n, m // 2) < np.exp(-dE_white / T)
            mask = self.___weave_checkerboard(mask, "white")

            self.matrix[mask] = -self.matrix[mask]
            self.simulation.append(self.matrix)

            # update the black squares
            _, dE_black = self.__unweave_checkerboard(self.__hamiltonian_constant_J(self.matrix, J, B))

            mask = self.__generate_random_array(n, m // 2) < np.exp(-dE_black / T)
            mask = self.___weave_checkerboard(mask, "black")

            self.matrix[mask] = -self.matrix[mask]
            self.simulation.append(self.matrix)

        return self.simulation

    def __unweave_checkerboard(self, lattice: np.ndarray[int,int]) -> tuple[np.ndarray[int,int], np.ndarray[int,int]]:
        """Split lattice into its white squares and black squares according to the checkerboard convention."""
        n, m = lattice.shape

        assert m % 2 == 0, "Cannot unweave uneven checkerboard; uneven number or columns."

        mask = np.indices((n, m)).sum(axis=0) % 2 == 0

        white_lattice = lattice[mask].reshape((n, m // 2))
        black_lattice = lattice[~mask].reshape((n, m // 2))

        return (white_lattice, black_lattice)

    def __weave_checkerboard(self, white_lattice:np.ndarray[int,int], black_lattice:np.ndarray[int,int]) -> np.ndarray[int,int]:
        """Weave together two lattices (of the same size) into their combined checkerboard lattice."""

        assert (
            white_lattice.shape == black_lattice.shape
        ), "Cannot weave together differnently shaped lattices; result not a rectangular lattice."

        return self.___weave_checkerboard(white_lattice, "w") + self.___weave_checkerboard(
            black_lattice, "b"
        )

    def ___weave_checkerboard(self, lattice: np.ndarray[int,int], square_color: {"w","b"}) -> np.ndarray[int,int]:
        """Create an array with double the number of columns, where the values are
        alternatingly values from the lattice and zeros, put the values of the array either
        on the WhiteSquares or the BlackSquares according to checkboard convention."""

        n, m = lattice.shape

        if square_color in ("w", "white"):
            return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 0)

        if square_color in ("b", "black"):
            return lattice.repeat(2, axis=1) * (np.indices((n, 2 * m)).sum(axis=0) % 2 == 1)

    def __generate_random_array(self, n, m):
        """Create an n by m array with samples from ~Unif [0, 1)."""
        return np.random.random_sample((n, m))

    def __checkerboard_neighbor_sum(self,lattice:np.ndarray[int,int]) -> np.ndarray[int,int]:
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        return convolve(input=lattice, weights=kernel, mode="wrap")

    def __hamiltonian_constant_J(self, lattice, J, h):
        """Compute the energy of a lattice."""

        return 2 * (J * self.__checkerboard_neighbor_sum(lattice) + h) * lattice

    def lattices_to_images(self, scale_factor: int = 1) -> list[Image.Image]:
        return [self.lattice_to_image(matrix, scale_factor) for matrix in self.simulation]

    def upscale_lattice(self, matrix: np.ndarray[int, int], scale_factor: int) -> np.ndarray[int, int]:
        return matrix.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)
    
    def repeated_simulations(self, J: float, B: float, T: float, steps: int, runs: int, method: Literal["metropolis","checkerboard"] | None = "metropolis") -> list[np.ndarray[int,int]]:
        self.simulation_finals.clear()
        for i in range(runs):
            sim = self.simulate(J, B, T, steps, method)
            self.simulation_finals.append(sim[-1])
        return self.simulation_finals
    
class squarelattice2d(lattice2d):
    def neighbor_sum(self, X: tuple[int,...]):
        assert len(X) == 2
        a,b = X
        width, height = self.matrix.shape
        sum = 0

        if a > 0:  # a left beighbour exists
            sum += self.matrix[a - 1, b]
        if a < width - 1:  # a right beighbour exists
            sum += self.matrix[a + 1, b]
        if b > 0:  # an upper beighbour exists
            sum += self.matrix[a, b - 1]
        if b < height - 1:  # a lower beighbour exists
            sum += self.matrix[a, b + 1]
        
        return sum
    
    def lattice_to_image(self, matrix: np.ndarray[int, int], scale_factor: int = 1) -> Image:
        return Image.fromarray(np.uint8((self.upscale_lattice(matrix, scale_factor) + 1) * 0.5 * 255))
    
class triangularlattice2d(lattice2d):
    def neighbor_sum(self, X: tuple[int,...]):
        assert X.count() == 2
        a,b = X
        width, height = self.matrix.shape
        sum = 0
        
        # Pattern:
        #   .   .   .   .   .
        #     .   .   .   .   .
        #   .   .   .   .   .
        #     .   .   .   .   .
        
        if a > 0:  # a left beighbour exists
            sum += self.matrix[a - 1, b]
        if a < width - 1:  # a right beighbour exists
            sum += self.matrix[a + 1, b]
        if b > 0 and (a > 0 or b & 1):  # an upper-left beighbour exists
            sum += self.matrix[a - 1 + (b & 1), b - 1]
        if b < height - 1 and (a > 0 or b & 1):  # a lower-left beighbour exists
            sum += self.matrix[a - 1 + (b & 1), b + 1]
        if b > 0 and (a < width - 1 or not (b & 1)):  # an upper-right beighbour exists
            sum += self.matrix[a - (b & 1), b - 1]
        if b < height - 1 and (a < width - 1 or not (b & 1)):  # a lower-right beighbour exists
            sum += self.matrix[a - (b & 1), b + 1]

        return sum
    
    def lattice_to_image(self, matrix: np.ndarray[int, int], scale_factor: int = 1) -> Image:
        pass
    
class hexagonallattice2d(lattice2d):
    def neighbor_sum(self, X: tuple[int, ...]):
        assert X.count() == 2
        a,b = X
        width, height = self.matrix.shape
        sum = 0
        
        # Pattern:
        #     .  .      .  . 
        #   .      .  .      .
        #     .  .      .  . 
        #   .      .  .      .
        #     .  .      .  .
        
        if (a+b) & 1:     # points left
            if a > 0:  # a left beighbour exists
                sum += self.matrix[a - 1, b]
            if b > 0 and ((a+b) & 1):  # an upper-right beighbour exists
                sum += self.matrix[a, b - 1]
            if b < height - 1 and ((a+b) & 1):  # a lower-right beighbour exists
                sum += self.matrix[a, b + 1]

        if not ((a+b) & 1):   # points right
            if a < width - 1:  # a right beighbour exists
                sum += self.matrix[a + 1, b]
            if b < height - 1:  # a lower-left beighbour exists
                sum += self.matrix[a, b + 1]
            if b > 0:  # an upper-left beighbour exists
                sum += self.matrix[a, b - 1]
        
        return sum
    
    def lattice_to_image(self, matrix: np.ndarray[int, int], scale_factor: int = 1) -> Image:
        pass

A = squarelattice2d(10,10)
B = triangularlattice2d(10,10)
C = hexagonallattice2d(10,10)