from typing import List, Tuple, Dict, Any
import numpy as np
import random
from src.utils import is_solvable, get_path_length, simulate_solver_path
from src.environment import GridEnvironment


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

        We start with very few walls (5%) so solvers have chances
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

            # Enforce max wall density removing walls if specified in the config (curriculum learning)
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
        
        This implements curriculum learning - keeps mazes simpler in the earlier steps of the evolution.
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
        Calculate fitness based on maze difficulty, diversity, and solvability.

        A good architect maze should:
        1. Be solvable (mandatory)
        2. Challenge solvers (cause some to fail)
        3. Have a longer optimal path (more interesting)
        4. Have good wall diversity (not too simple)

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

        # Get optimal path length (longer path = more interesting maze)
        optimal_length = get_path_length(self.genome, self.start_pos, self.end_pos)
        manhattan_distance = (self.grid_size - 1) * 2  # Straight-line distance from start to end

        # Count how many solvers fail to reach the goal
        failures = 0
        total_solvers = len(solvers)
        solver_path_lengths = []

        for solver in solvers:
            path, final_pos = simulate_solver_path(self.genome, solver.genome, self.start_pos,
                                                   self.end_pos, solver.max_moves)
            if final_pos != self.end_pos:
                failures += 1
            else:
                solver_path_lengths.append(len(path) - 1)

        # Calculate failure rate (percentage of solvers that failed)
        failure_rate = failures / total_solvers if total_solvers > 0 else 0

        # Wall diversity: penalize if too few or too many walls
        wall_count = self._count_walls(self.genome)
        total_cells = self.grid_size * self.grid_size
        wall_density = wall_count / total_cells
        # Ideal density is around 0.25-0.35 (sweet spot for interesting mazes)
        density_penalty = abs(wall_density - 0.30) * 20

        # Extract weights from config
        difficulty_weight = config['fitness']['architect_maze_difficulty_weight']
        diversity_weight = config['fitness']['architect_diversity_weight']

        # Fitness components:
        # 1. Difficulty: reward for making solvers fail (but not all of them)
        # Ideal is 30-70% failure rate (too easy or impossible is bad)
        if failure_rate < 0.3:
            difficulty_score = failure_rate * 100  # Reward partial difficulty
        elif failure_rate <= 0.7:
            difficulty_score = 30 + (failure_rate - 0.3) * 100
        else:
            difficulty_score = 30 - (failure_rate - 0.7) * 50  # Penalize too hard

        # 2. Path complexity: reward mazes where optimal path is longer than Manhattan distance
        # Ratio > 1.0 means the maze forces detours (more interesting)
        path_complexity = (optimal_length / manhattan_distance) * 50

        # 3. Diversity: penalize extremes in wall density
        diversity_score = max(0, 20 - density_penalty)

        # Note: No solvability bonus needed since unsolvable mazes already get fitness=0
        self.fitness = (difficulty_weight * difficulty_score +
                       path_complexity +
                       diversity_weight * diversity_score)

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
    Solver individual that evolves reactive navigation policies.
    Genome: List of 4 float weights [w_goal_dist, w_wall_penalty, w_visited_penalty, w_random_exploration]

    At each step, the solver evaluates neighboring cells using these weights:
    Score = w_goal * (goal_distance_improvement) + w_wall * (is_wall) + w_visited * (visit_count) + w_random * (random_noise)
    """

    def __init__(self, max_moves: int, start_pos: Tuple[int, int] = None,
                 end_pos: Tuple[int, int] = None, use_smart_init: bool = True):
        """
        Initialize a Solver with a weight-based genome.

        Args:
            max_moves: Maximum number of moves allowed in simulation
            start_pos: Starting position (kept for compatibility)
            end_pos: Goal position (kept for compatibility)
            use_smart_init: If True, use heuristic initialization for weights
        """
        self.max_moves = max_moves
        self.start_pos = start_pos
        self.end_pos = end_pos

        # Weight-based genome: [w_goal_dist, w_wall_penalty, w_visited_penalty, w_random_exploration]
        if use_smart_init:
            # Smart initialization: reasonable starting weights
            self.genome = [
                random.uniform(0.5, 2.0),   # w_goal_dist: favor moving toward goal
                random.uniform(-2.0, -0.5), # w_wall_penalty: avoid walls (negative)
                random.uniform(-1.0, 0.0),  # w_visited_penalty: avoid revisiting (negative)
                random.uniform(0.0, 0.5)    # w_random_exploration: some randomness
            ]
        else:
            # Random initialization
            self.genome = [random.uniform(-2.0, 2.0) for _ in range(4)]

        self.fitness = 0.0

    def calculate_fitness(self, architect: Architect, end_pos: Tuple[int, int],
                         config: Dict[str, Any]) -> float:
        """
        Calculate fitness based on reaching goal and path efficiency.

        For weight-based reactive agents, we want to reward:
        1. Successfully reaching the goal (most important)
        2. Path efficiency (shorter paths are better)
        3. Progress toward goal (if goal not reached)

        Args:
            architect: Architect whose maze to solve
            end_pos: Goal position
            config: Configuration dictionary

        Returns:
            Fitness score (higher is better for the solver)
        """
        start_pos = architect.start_pos
        path, final_pos = simulate_solver_path(architect.genome, self.genome, start_pos,
                                               end_pos, self.max_moves)

        goal_bonus = config['fitness']['solver_goal_bonus']
        efficiency_weight = config['fitness']['solver_path_efficiency_weight']
        progress_weight = config['fitness']['solver_progress_weight']

        # Calculate distances
        distance_to_goal = abs(final_pos[0] - end_pos[0]) + abs(final_pos[1] - end_pos[1])
        max_distance = abs(start_pos[0] - end_pos[0]) + abs(start_pos[1] - end_pos[1])

        # Check if reached goal
        if final_pos == end_pos:
            # SUCCESS! Give bonus based on path efficiency
            path_length = len(path) - 1

            # Get optimal path length for this maze
            optimal_length = get_path_length(architect.genome, start_pos, end_pos)

            # Calculate efficiency: reward paths closer to optimal
            if optimal_length > 0 and path_length > 0:
                # Efficiency ratio: 1.0 if perfect, decreases as path gets longer
                efficiency_ratio = optimal_length / path_length
                efficiency_ratio = min(efficiency_ratio, 1.0)  # Cap at 1.0

                # Path penalty: penalize longer paths more strongly
                # Extra steps beyond optimal are heavily penalized
                extra_steps = max(0, path_length - optimal_length)
                path_penalty = extra_steps * 2.0  # Each extra step costs 2 fitness points
            else:
                efficiency_ratio = 1.0
                path_penalty = 0

            # Final fitness: goal bonus scaled by efficiency, minus path penalty
            # This makes shorter paths significantly better than longer ones
            self.fitness = (goal_bonus * efficiency_ratio) + efficiency_weight - path_penalty
        else:
            # FAILURE: didn't reach goal
            # Reward based on how close we got
            if max_distance > 0:
                progress = max_distance - distance_to_goal  # How much closer to goal
                progress_ratio = progress / max_distance

                # Quadratic scaling: getting very close is much better
                self.fitness = progress_weight * (progress_ratio ** 2.0)
            else:
                # Start and end are same position (edge case)
                self.fitness = 0

        return self.fitness

    def mutate(self, mutation_rate: float, mutation_stddev: float = 0.3):
        """
        Mutate the genome by perturbing weights with Gaussian noise.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_stddev: Standard deviation of Gaussian noise
        """
        for i in range(len(self.genome)):
            if random.random() < mutation_rate:
                # Add Gaussian noise to weight
                noise = np.random.normal(0, mutation_stddev)
                self.genome[i] += noise

    @staticmethod
    def crossover(parent1: 'Solver', parent2: 'Solver') -> Tuple['Solver', 'Solver']:
        """
        Perform crossover between two parents to create two offspring.
        Uses single-point crossover on weight vectors.

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

        # Single-point crossover on weight vector
        num_weights = len(parent1.genome)
        crossover_point = random.randint(1, num_weights - 1)

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
