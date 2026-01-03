from typing import List, Dict, Any
import numpy as np
from src.population import Architect, Solver
from src.utils import simulate_solver_path, count_walls


class MetricsTracker:
    """
    Tracks and stores metrics throughout the co-evolution process.
    """

    def __init__(self):
        """Initialize metrics tracker with empty history."""
        self.history = {
            'generation': [],
            'avg_architect_fitness': [],
            'max_architect_fitness': [],
            'avg_solver_fitness': [],
            'max_solver_fitness': [],
            'solver_success_rate': [],
            'avg_path_length': [],
            'avg_wall_count': []
        }

    def record_generation(self,
                         generation: int,
                         architects: List[Architect],
                         solvers: List[Solver],
                         end_pos: tuple):
        """
        Record metrics for a generation.

        Args:
            generation: Current generation number
            architects: List of Architect individuals
            solvers: List of Solver individuals
            end_pos: Goal position
        """
        # Architect metrics
        architect_fitnesses = [arch.fitness for arch in architects]
        avg_arch_fitness = np.mean(architect_fitnesses) if architect_fitnesses else 0
        max_arch_fitness = np.max(architect_fitnesses) if architect_fitnesses else 0

        # Solver metrics
        solver_fitnesses = [sol.fitness for sol in solvers]
        avg_sol_fitness = np.mean(solver_fitnesses) if solver_fitnesses else 0
        max_sol_fitness = np.max(solver_fitnesses) if solver_fitnesses else 0

        # Calculate solver success rate (across all architect-solver pairs)
        success_count = 0
        total_tests = 0
        path_lengths = []

        for architect in architects:
            for solver in solvers:
                path, final_pos = simulate_solver_path(
                    architect.genome,
                    solver.genome,
                    architect.start_pos
                )
                if final_pos == end_pos:
                    success_count += 1
                    path_lengths.append(len(path) - 1)
                total_tests += 1

        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        avg_path_length = np.mean(path_lengths) if path_lengths else 0

        # Wall count (average across all architects)
        wall_counts = [count_walls(arch.genome) for arch in architects]
        avg_wall_count = np.mean(wall_counts) if wall_counts else 0

        # Store metrics
        self.history['generation'].append(generation)
        self.history['avg_architect_fitness'].append(avg_arch_fitness)
        self.history['max_architect_fitness'].append(max_arch_fitness)
        self.history['avg_solver_fitness'].append(avg_sol_fitness)
        self.history['max_solver_fitness'].append(max_sol_fitness)
        self.history['solver_success_rate'].append(success_rate)
        self.history['avg_path_length'].append(avg_path_length)
        self.history['avg_wall_count'].append(avg_wall_count)

    def get_best_architect(self, architects: List[Architect]) -> Architect:
        """Get the architect with highest fitness."""
        return max(architects, key=lambda x: x.fitness)

    def get_best_solver(self, solvers: List[Solver], architects: List[Architect] = None,
                       end_pos: tuple = None) -> Solver:
        """Get the best solver.

        Prefers a solver that actually reaches the goal if possible.
        Falls back to highest fitness if no successful solvers.
        """
        if architects and end_pos:
            # Try to find a solver that succeeds on at least one maze
            for solver in sorted(solvers, key=lambda x: x.fitness, reverse=True):
                for architect in architects:
                    path, final_pos = simulate_solver_path(
                        architect.genome, solver.genome, architect.start_pos
                    )
                    if final_pos == end_pos:
                        return solver  # Found a successful solver!

        # Fallback: highest fitness
        return max(solvers, key=lambda x: x.fitness)

    def print_generation_summary(self, generation: int):
        """
        Print a summary of the current generation's metrics.

        Args:
            generation: Current generation number
        """
        idx = generation
        print(f"\n{'='*60}")
        print(f"Generation {generation} Summary")
        print(f"{'='*60}")
        print(f"Architects:")
        print(f"  Average Fitness: {self.history['avg_architect_fitness'][idx]:.2f}")
        print(f"  Max Fitness:     {self.history['max_architect_fitness'][idx]:.2f}")
        print(f"\nSolvers:")
        print(f"  Average Fitness: {self.history['avg_solver_fitness'][idx]:.2f}")
        print(f"  Max Fitness:     {self.history['max_solver_fitness'][idx]:.2f}")
        print(f"\nPerformance:")
        print(f"  Success Rate:    {self.history['solver_success_rate'][idx]:.2f}%")
        print(f"  Avg Path Length: {self.history['avg_path_length'][idx]:.2f}")
        print(f"  Avg Wall Count:  {self.history['avg_wall_count'][idx]:.1f}")
        print(f"{'='*60}\n")

    def get_history(self) -> Dict[str, List]:
        """Get the complete history dictionary."""
        return self.history

    def save_metrics(self, filepath: str):
        """
        Save metrics to a JSON file.

        Args:
            filepath: Path to save the metrics
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_final_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the final generation.

        Returns:
            Dictionary containing final metrics
        """
        if not self.history['generation']:
            return {}

        last_idx = -1
        return {
            'final_generation': self.history['generation'][last_idx],
            'final_avg_architect_fitness': self.history['avg_architect_fitness'][last_idx],
            'final_max_architect_fitness': self.history['max_architect_fitness'][last_idx],
            'final_avg_solver_fitness': self.history['avg_solver_fitness'][last_idx],
            'final_max_solver_fitness': self.history['max_solver_fitness'][last_idx],
            'final_success_rate': self.history['solver_success_rate'][last_idx],
            'final_avg_path_length': self.history['avg_path_length'][last_idx],
            'final_avg_wall_count': self.history['avg_wall_count'][last_idx]
        }
