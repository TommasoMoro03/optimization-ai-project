from typing import List, Tuple, Dict, Any
import numpy as np
import random
from src.utils import is_solvable, get_path_length, simulate_solver_path
from src.environment import GridEnvironment
from src.solver_strategies import create_greedy_solver_genome, create_mixed_strategy_genome


class Architect:
    """
    Architect individual that evolves maze maps.
    Genome: Binary 2D grid (0=empty, 1=wall)
    """

    def __init__(self, grid_size: int, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        """
        Initialize an Architect with a random genome.

        Args:
            grid_size: Size of the grid
            start_pos: Start position (must be empty)
            end_pos: End position (must be empty)
        """
        self.grid_size = grid_size
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.genome = self._create_random_genome()
        self.fitness = 0.0

    def _create_random_genome(self, wall_prob: float = 0.05, max_wall_density: float = None) -> List[List[int]]:
        """Create a random valid genome (solvable maze).

        IMPORTANT: Start with very few walls (5%) so solvers have a chance!
        Evolution will increase complexity over time.
        
        Args:
            wall_prob: Probability of placing a wall in each cell
            max_wall_density: Maximum allowed wall density (0.0-1.0), enforced after creation
        """
        max_attempts = 100
        for _ in range(max_attempts):
            genome = [[1 if random.random() < wall_prob else 0
                      for _ in range(self.grid_size)]
                     for _ in range(self.grid_size)]

            # Ensure start and end are not walls
            genome[self.start_pos[0]][self.start_pos[1]] = 0
            genome[self.end_pos[0]][self.end_pos[1]] = 0

            # Enforce max wall density if specified (curriculum learning)
            if max_wall_density is not None:
                genome = self._enforce_max_wall_density(genome, max_wall_density)

            # Check if solvable
            if is_solvable(genome, self.start_pos, self.end_pos):
                return genome

        # If we can't find a solvable maze, create a simple path
        return self._create_simple_path()

    def _create_simple_path(self) -> List[List[int]]:
        """Create a simple solvable maze with a clear path."""
        genome = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        return genome
    
    def _count_walls(self, genome: List[List[int]]) -> int:
        """Count the number of walls in the genome."""
        return sum(sum(row) for row in genome)
    
    def _get_wall_density(self, genome: List[List[int]]) -> float:
        """Calculate wall density (0.0 to 1.0)."""
        total_cells = self.grid_size * self.grid_size
        wall_count = self._count_walls(genome)
        return wall_count / total_cells if total_cells > 0 else 0.0
    
    def _enforce_max_wall_density(self, genome: List[List[int]], max_density: float) -> List[List[int]]:
        """Enforce maximum wall density by randomly removing walls if needed.
        
        This implements curriculum learning - keeps mazes simpler early in evolution.
        """
        current_density = self._get_wall_density(genome)
        if current_density <= max_density:
            return genome
        
        # Need to remove walls to meet the limit
        total_cells = self.grid_size * self.grid_size
        max_walls = int(max_density * total_cells)
        current_walls = self._count_walls(genome)
        walls_to_remove = current_walls - max_walls
        
        # Create list of wall positions (excluding start/end)
        wall_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if genome[i][j] == 1 and (i, j) != self.start_pos and (i, j) != self.end_pos:
                    wall_positions.append((i, j))
        
        # Randomly remove walls
        if walls_to_remove > 0 and len(wall_positions) > 0:
            to_remove = random.sample(wall_positions, min(walls_to_remove, len(wall_positions)))
            new_genome = [row[:] for row in genome]
            for i, j in to_remove:
                new_genome[i][j] = 0
            return new_genome
        
        return genome
    
    def enforce_max_wall_density(self, max_density: float):
        """Enforce maximum wall density on this architect's genome (for curriculum learning)."""
        self.genome = self._enforce_max_wall_density(self.genome, max_density)
        # Ensure still solvable, if not, create simple path
        if not is_solvable(self.genome, self.start_pos, self.end_pos):
            self.genome = self._create_simple_path()

    def calculate_fitness(self, solvers: List['Solver'], config: Dict[str, Any]) -> float:
        """
        Calculate fitness based on how many solvers fail and path length.

        Args:
            solvers: List of Solver individuals to test against
            config: Configuration dictionary

        Returns:
            Fitness score (higher is better for the architect)
        """
        # Check if maze is solvable - if not, fitness is 0
        if not is_solvable(self.genome, self.start_pos, self.end_pos):
            self.fitness = 0.0
            return self.fitness

        # Get optimal path length
        optimal_length = get_path_length(self.genome, self.start_pos, self.end_pos)

        # Count how many solvers fail to reach the goal
        failures = 0
        for solver in solvers:
            path, final_pos = simulate_solver_path(self.genome, solver.genome, self.start_pos)
            if final_pos != self.end_pos:
                failures += 1

        # Fitness: weighted combination of failures and path length
        path_weight = config['fitness']['architect_path_weight']
        failure_weight = config['fitness']['architect_failure_weight']

        self.fitness = (failure_weight * failures) + (path_weight * optimal_length)
        return self.fitness

    def mutate(self, mutation_rate: float, max_wall_density: float = None):
        """
        Mutate the genome by flipping bits with given probability.
        Ensures maze remains solvable.

        Args:
            mutation_rate: Probability of flipping each cell
            max_wall_density: Maximum allowed wall density (for curriculum learning)
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            # Create a copy to test
            new_genome = [row[:] for row in self.genome]

            # Mutate cells
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Don't mutate start or end positions
                    if (i, j) == self.start_pos or (i, j) == self.end_pos:
                        continue

                    if random.random() < mutation_rate:
                        new_genome[i][j] = 1 - new_genome[i][j]  # Flip bit

            # Enforce max wall density if specified
            if max_wall_density is not None:
                new_genome = self._enforce_max_wall_density(new_genome, max_wall_density)

            # Check if still solvable
            if is_solvable(new_genome, self.start_pos, self.end_pos):
                self.genome = new_genome
                return

        # If mutation fails to produce solvable maze, keep original

    @staticmethod
    def crossover(parent1: 'Architect', parent2: 'Architect', max_wall_density: float = None) -> Tuple['Architect', 'Architect']:
        """
        Perform crossover between two parents to create two offspring.
        Uses single-point crossover.

        Args:
            parent1: First parent
            parent2: Second parent
            max_wall_density: Maximum allowed wall density (for curriculum learning)

        Returns:
            Tuple of two offspring Architects
        """
        child1 = Architect(parent1.grid_size, parent1.start_pos, parent1.end_pos)
        child2 = Architect(parent2.grid_size, parent2.start_pos, parent2.end_pos)

        # Single-point crossover at random row
        crossover_point = random.randint(1, parent1.grid_size - 1)

        # Create children genomes
        child1_genome = (parent1.genome[:crossover_point] +
                        parent2.genome[crossover_point:])
        child2_genome = (parent2.genome[:crossover_point] +
                        parent1.genome[crossover_point:])

        # Ensure start and end are not walls
        for genome in [child1_genome, child2_genome]:
            genome[parent1.start_pos[0]][parent1.start_pos[1]] = 0
            genome[parent1.end_pos[0]][parent1.end_pos[1]] = 0

        # Enforce max wall density if specified
        if max_wall_density is not None:
            child1_genome = child1._enforce_max_wall_density(child1_genome, max_wall_density)
            child2_genome = child2._enforce_max_wall_density(child2_genome, max_wall_density)

        # Only keep if solvable, otherwise use simple path
        if is_solvable(child1_genome, parent1.start_pos, parent1.end_pos):
            child1.genome = child1_genome
        else:
            child1.genome = child1._create_simple_path()

        if is_solvable(child2_genome, parent2.start_pos, parent2.end_pos):
            child2.genome = child2_genome
        else:
            child2.genome = child2._create_simple_path()

        return child1, child2


class Solver:
    """
    Solver individual that evolves move sequences.
    Genome: List of integers representing moves (0=N, 1=E, 2=S, 3=W)
    """

    def __init__(self, max_moves: int, start_pos: Tuple[int, int] = None,
                 end_pos: Tuple[int, int] = None, use_smart_init: bool = True):
        """
        Initialize a Solver with a genome.

        Args:
            max_moves: Maximum number of moves in the sequence
            start_pos: Starting position (for smart initialization)
            end_pos: Goal position (for smart initialization)
            use_smart_init: If True, use greedy heuristic initialization
        """
        self.max_moves = max_moves
        self.start_pos = start_pos
        self.end_pos = end_pos

        # Smart initialization: bias toward goal direction
        if use_smart_init and start_pos is not None and end_pos is not None:
            self.genome = create_mixed_strategy_genome(max_moves, start_pos, end_pos)
        else:
            # Fallback to random
            self.genome = [random.randint(0, 3) for _ in range(max_moves)]

        self.fitness = 0.0

    def calculate_fitness(self, architect: Architect, end_pos: Tuple[int, int],
                         config: Dict[str, Any]) -> float:
        """
        Calculate fitness based on reaching goal and path efficiency.
        Improved to give better rewards for partial progress.

        Args:
            architect: Architect whose maze to solve
            end_pos: Goal position
            config: Configuration dictionary

        Returns:
            Fitness score (higher is better for the solver)
        """
        start_pos = architect.start_pos
        path, final_pos = simulate_solver_path(architect.genome, self.genome, start_pos)

        goal_bonus = config['fitness']['solver_goal_bonus']
        distance_weight = config['fitness']['solver_distance_weight']

        # Calculate distances
        distance_to_goal = abs(final_pos[0] - end_pos[0]) + abs(final_pos[1] - end_pos[1])
        max_distance = abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])
        progress = max_distance - distance_to_goal  # How much closer to goal

        # Check if reached goal
        if final_pos == end_pos:
            # HUGE bonus for reaching goal, small penalty for path length
            path_length = len(path) - 1
            optimal_length = max_distance  # Manhattan distance
            efficiency = 1.0 - min((path_length - optimal_length) / max(optimal_length, 1), 1.0)

            self.fitness = goal_bonus + (efficiency * 50)  # 100-150 range for success
        else:
            # Improved fitness for partial progress
            # Use stronger reward to favor getting closer
            if max_distance > 0:
                progress_ratio = progress / max_distance
                # Stronger scaling: reward increases much faster as you get closer
                # Changed from 1.5 to 2.0 for stronger gradient
                self.fitness = distance_weight * (progress_ratio ** 2.0) * max_distance
            else:
                self.fitness = 0

            # Bonus for path diversity (exploration) - increased
            unique_positions = len(set(path))
            self.fitness += unique_positions * 1.0
            
            # Extra bonus for using fewer moves (efficiency)
            if len(path) > 0:
                efficiency_bonus = max(0, (self.max_moves - len(path)) / self.max_moves * 10)
                self.fitness += efficiency_bonus

        return self.fitness

    def mutate(self, mutation_rate: float):
        """
        Mutate the genome by randomly changing moves.
        Improved to add "retry" patterns that help when stuck.

        Args:
            mutation_rate: Probability of changing each move
        """
        # Standard mutation: randomly change moves
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                self.genome[i] = random.randint(0, 3)
        
        # Additional improvement: add "retry" patterns (10% chance)
        # This helps when solver hits a wall - tries same direction multiple times
        if random.random() < 0.1 and len(self.genome) >= 3:
            # Pick a random position and add a repeated pattern
            pos = random.randint(0, len(self.genome) - 3)
            direction = random.randint(0, 3)
            # Repeat same direction 2-3 times (helps push through obstacles)
            repeat_count = random.randint(2, 3)
            for j in range(repeat_count):
                if pos + j < len(self.genome):
                    self.genome[pos + j] = direction

    @staticmethod
    def crossover(parent1: 'Solver', parent2: 'Solver') -> Tuple['Solver', 'Solver']:
        """
        Perform crossover between two parents to create two offspring.
        Uses single-point crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Tuple of two offspring Solvers
        """
        child1 = Solver(parent1.max_moves, parent1.start_pos, parent1.end_pos,
                       use_smart_init=False)
        child2 = Solver(parent2.max_moves, parent2.start_pos, parent2.end_pos,
                       use_smart_init=False)

        # Single-point crossover
        crossover_point = random.randint(1, parent1.max_moves - 1)

        child1.genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        child2.genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]

        # Preserve parent info for future breeding
        child1.start_pos = parent1.start_pos
        child1.end_pos = parent1.end_pos
        child2.start_pos = parent2.start_pos
        child2.end_pos = parent2.end_pos

        return child1, child2


def tournament_selection(population: List, tournament_size: int):
    """
    Select an individual from population using tournament selection.

    Args:
        population: List of individuals (Architects or Solvers)
        tournament_size: Number of individuals in tournament

    Returns:
        Selected individual
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda ind: ind.fitness)
