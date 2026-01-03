from typing import List, Tuple, Dict, Any
import numpy as np

class GridEnvironment:
    """
    Represents the 2D grid environment for the maze.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the grid environment from configuration.

        Args:
            config: Configuration dictionary containing environment settings
        """
        self.grid_size = config['environment']['grid_size']
        self.start_pos = tuple(config['environment']['start_pos'])
        self.end_pos = tuple(config['environment']['end_pos'])

        # Validate positions
        self._validate_positions()

    def _validate_positions(self):
        """Validate that start and end positions are within grid bounds."""
        if not (0 <= self.start_pos[0] < self.grid_size and
                0 <= self.start_pos[1] < self.grid_size):
            raise ValueError(f"Start position {self.start_pos} out of bounds")

        if not (0 <= self.end_pos[0] < self.grid_size and
                0 <= self.end_pos[1] < self.grid_size):
            raise ValueError(f"End position {self.end_pos} out of bounds")

        if self.start_pos == self.end_pos:
            raise ValueError("Start and end positions cannot be the same")

    def create_empty_grid(self) -> List[List[int]]:
        """
        Create an empty grid (all zeros).

        Returns:
            2D list representing an empty grid
        """
        return [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def create_random_grid(self, wall_probability: float = 0.3) -> List[List[int]]:
        """
        Create a random grid with walls.

        Args:
            wall_probability: Probability of a cell being a wall

        Returns:
            2D list representing a random grid
        """
        grid = [[1 if np.random.random() < wall_probability else 0
                 for _ in range(self.grid_size)]
                for _ in range(self.grid_size)]

        # Ensure start and end are not walls
        grid[self.start_pos[0]][self.start_pos[1]] = 0
        grid[self.end_pos[0]][self.end_pos[1]] = 0

        return grid

    def grid_to_array(self, grid: List[List[int]]) -> np.ndarray:
        """Convert grid list to numpy array."""
        return np.array(grid)

    def array_to_grid(self, arr: np.ndarray) -> List[List[int]]:
        """Convert numpy array to grid list."""
        return arr.tolist()

    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the grid."""
        return (self.grid_size, self.grid_size)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within grid bounds)."""
        return (0 <= pos[0] < self.grid_size and
                0 <= pos[1] < self.grid_size)
