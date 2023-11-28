import os
import shutil

from abc import ABC, abstractmethod
import numpy as np
from numba import njit
from PIL import Image
#from ipywidgets import interact


class lattice(ABC):
    
    @njit()
    @abstractmethod
    def neighbor_sum(self, X:tuple[int,...]):
        """Calculates the sum of the neighbours of the spin at position (x, y). Implementation depends on subclass."""
        pass
    
    @njit()
    @abstractmethod
    def simulate(self, J: float, B: float, T: float, steps: int):
        """Simulate ising model using the Metropolis algorithm.
        J : global spin-spin interaction constant,
        B : external magnetic field"""
        pass
    
    @abstractmethod
    def lattice_to_image(self, matrix: np.ndarray[int,int], scale_factor: int) -> Image.Image:
        """Turn 2d array into an image, the value of each pixel
        is determined via affine transformation of [-1, 1] -> [0, 255]. Is dependent on implementation of subclass."""
        pass

    @abstractmethod
    def lattices_to_images(self, scale_factor: int) -> list[Image.Image]:
        """Turn list of 2d simulation steps into an image, the value of each pixel
        is determined via affine transformation of [-1, 1] -> [0, 255]. Is dependent on implementation of subclass."""
        pass
    
    def lattices_to_images(self) -> list[Image.Image]:
        """Turn list of 2d simulation steps into an image, the value of each pixel
        is determined via affine transformation of [-1, 1] -> [0, 255]. Is dependent on implementation of subclass."""
        self.lattices_to_images(1)
    
    @abstractmethod
    def upscale_lattice(self, matrix: np.ndarray[int,int], scale_factor: int) -> np.ndarray[int,int]:
        """Upscale array by integer multiple in all directions."""
        pass

    def upscale_lattices(self, matrices: list[np.ndarray[int,int]], scale_factor: int) -> list[np.ndarray[int,int]]:
        """Upscale list of arrays, each array gets upscaled by an integer
        multiple in all directions."""

        return [self.upscale_lattice(matrix, scale_factor) for matrix in matrices]
    
class lattice2d(lattice):
    
    def __init__(self, width: int, height: int, states: list[int] = None) -> None:
        
        if states is None:
            states = [-1, 1]
        self.matrix:np.ndarray[int,int] = np.random.choice(states, size=(width, height))
        self.initialmatrix = self.matrix.copy()
        self.simulation:list[np.ndarray[int,int]] = []
    
    def simulate(self, J: float, B: float, T: float, steps: int):
        self.simulation.clear()
        self.simulation.append(self.initialmatrix)
        self.matrix = self.initialmatrix.copy()
        width, height = self.matrix.shape

        for _ in range(steps):
            x = np.random.randint(width)
            y = np.random.randint(height)

            spin = self.matrix[x, y]

            position = (x,y)
            neighbour_sum = self.neighbor_sum(position)

            dE = 2 * spin * (J * neighbour_sum - B)

            if (neighbour_sum * spin < 0) or (np.random.random(1) < np.exp(-dE / T)):
                self.matrix[x, y] = -self.matrix[x, y]

            self.simulation.append(self.matrix.copy())
        return self.simulation

    def lattices_to_images(self, scale_factor:int) -> list[Image.Image]:
        return [self.lattice_to_image(matrix, scale_factor) for matrix in self.simulation]

    def upscale_lattice(self, matrix: np.ndarray[int, int], scale_factor: int) -> np.ndarray[int, int]:
        return matrix.repeat(scale_factor, axis=0).repeat(scale_factor, axis=1)
    
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
    
    def lattice_to_image(self, matrix: np.ndarray[int, int], scale_factor:int) -> Image:
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
    
    def lattice_to_image(self, matrix: np.ndarray[int, int], scale_factor: int) -> Image:
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
    
    def lattice_to_image(matrix: np.ndarray[int, int]) -> Image:
        pass

A = squarelattice2d(10,10)
B = triangularlattice2d(10,10)
C = hexagonallattice2d(10,10)