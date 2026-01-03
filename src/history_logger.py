"""
History logger for genome tracking throughout evolution.
Saves genome data for analysis and visualization.
"""

import json
from typing import List, Dict, Any
from src.population import Architect, Solver


class HistoryLogger:
    """
    Logs detailed genome and fitness information for each generation.
    """

    def __init__(self):
        """Initialize the history logger."""
        self.history = []
        self.genetic_operations_log = []

    def log_generation(self,
                      generation: int,
                      architects: List[Architect],
                      solvers: List[Solver]):
        """
        Log complete genome data for a generation.

        Args:
            generation: Current generation number
            architects: List of Architect individuals
            solvers: List of Solver individuals
        """
        # Find best individuals
        best_architect = max(architects, key=lambda x: x.fitness)
        best_solver = max(solvers, key=lambda x: x.fitness)

        # Create generation record
        generation_record = {
            'generation_id': generation,
            'best_architect_genome': best_architect.genome,
            'best_architect_fitness': best_architect.fitness,
            'best_solver_genome': best_solver.genome,
            'best_solver_fitness': best_solver.fitness,
            # Additional statistics
            'avg_architect_fitness': sum(a.fitness for a in architects) / len(architects),
            'avg_solver_fitness': sum(s.fitness for s in solvers) / len(solvers),
            'num_architects': len(architects),
            'num_solvers': len(solvers)
        }

        self.history.append(generation_record)

    def log_crossover_event(self,
                           generation: int,
                           population_type: str,
                           parent1_genome: Any,
                           parent2_genome: Any,
                           child1_genome: Any,
                           child2_genome: Any):
        """
        Log a crossover event for verification.

        Args:
            generation: Current generation
            population_type: 'architect' or 'solver'
            parent1_genome: First parent's genome
            parent2_genome: Second parent's genome
            child1_genome: First child's genome
            child2_genome: Second child's genome
        """
        event = {
            'generation': generation,
            'operation': 'crossover',
            'population_type': population_type,
            'parent1_genome': parent1_genome,
            'parent2_genome': parent2_genome,
            'child1_genome': child1_genome,
            'child2_genome': child2_genome
        }
        self.genetic_operations_log.append(event)

    def log_mutation_event(self,
                          generation: int,
                          population_type: str,
                          genome_before: Any,
                          genome_after: Any):
        """
        Log a mutation event for verification.

        Args:
            generation: Current generation
            population_type: 'architect' or 'solver'
            genome_before: Genome before mutation
            genome_after: Genome after mutation
        """
        event = {
            'generation': generation,
            'operation': 'mutation',
            'population_type': population_type,
            'genome_before': genome_before,
            'genome_after': genome_after
        }
        self.genetic_operations_log.append(event)

    def save_history(self, filepath: str = 'output/evolution_history.json'):
        """
        Save the complete evolution history to a JSON file.

        Args:
            filepath: Path to save the history file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nEvolution history saved to: {filepath}")
        print(f"Total generations logged: {len(self.history)}")

    def save_genetic_operations(self, filepath: str = 'output/genetic_operations.json'):
        """
        Save genetic operations log for verification.

        Args:
            filepath: Path to save the operations file
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.genetic_operations_log, f, indent=2)

        print(f"Genetic operations log saved to: {filepath}")
        print(f"Total operations logged: {len(self.genetic_operations_log)}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the complete history."""
        return self.history

    def get_generation_data(self, generation: int) -> Dict[str, Any]:
        """
        Get data for a specific generation.

        Args:
            generation: Generation number

        Returns:
            Generation data dictionary or None if not found
        """
        for record in self.history:
            if record['generation_id'] == generation:
                return record
        return None

    def print_genome_summary(self, generation: int):
        """
        Print a human-readable summary of genomes for a generation.

        Args:
            generation: Generation number
        """
        data = self.get_generation_data(generation)
        if not data:
            print(f"No data found for generation {generation}")
            return

        print(f"\n{'='*70}")
        print(f"GENOME SUMMARY - Generation {generation}")
        print(f"{'='*70}")

        print(f"\nBest Architect:")
        print(f"  Fitness: {data['best_architect_fitness']:.2f}")
        print(f"  Genome (first 3 rows of grid):")
        for i in range(min(3, len(data['best_architect_genome']))):
            row = data['best_architect_genome'][i]
            row_str = ''.join(['█' if cell == 1 else '·' for cell in row])
            print(f"    {row_str}")

        print(f"\nBest Solver:")
        print(f"  Fitness: {data['best_solver_fitness']:.2f}")
        print(f"  Genome (first 20 moves):")
        moves = data['best_solver_genome'][:20]
        move_names = ['N', 'E', 'S', 'W']
        move_str = ''.join([move_names[m] for m in moves])
        print(f"    {move_str}...")
        print(f"    (Total moves: {len(data['best_solver_genome'])})")

        print(f"\nPopulation Stats:")
        print(f"  Avg Architect Fitness: {data['avg_architect_fitness']:.2f}")
        print(f"  Avg Solver Fitness: {data['avg_solver_fitness']:.2f}")
        print(f"{'='*70}\n")

    def print_genetic_operations_summary(self, num_operations: int = 5):
        """
        Print a summary of recent genetic operations.

        Args:
            num_operations: Number of recent operations to show
        """
        if not self.genetic_operations_log:
            print("No genetic operations logged yet.")
            return

        print(f"\n{'='*70}")
        print(f"GENETIC OPERATIONS SUMMARY (Last {num_operations} operations)")
        print(f"{'='*70}")

        operations_to_show = self.genetic_operations_log[-num_operations:]

        for i, op in enumerate(operations_to_show, 1):
            print(f"\n[{i}] Generation {op['generation']} - {op['operation'].upper()} ({op['population_type']})")

            if op['operation'] == 'crossover':
                print(f"  Parents → Children transformation:")
                if op['population_type'] == 'solver':
                    # Show first 15 moves for solvers
                    p1 = op['parent1_genome'][:15]
                    p2 = op['parent2_genome'][:15]
                    c1 = op['child1_genome'][:15]
                    c2 = op['child2_genome'][:15]
                    move_names = ['N', 'E', 'S', 'W']
                    print(f"    P1: {''.join([move_names[m] for m in p1])}...")
                    print(f"    P2: {''.join([move_names[m] for m in p2])}...")
                    print(f"    C1: {''.join([move_names[m] for m in c1])}...")
                    print(f"    C2: {''.join([move_names[m] for m in c2])}...")
                else:
                    # Show first 2 rows for architects
                    print(f"    P1 (first 2 rows):")
                    for row in op['parent1_genome'][:2]:
                        print(f"      {''.join(['█' if cell == 1 else '·' for cell in row])}")
                    print(f"    → Children created (see genetic_operations.json for full data)")

            elif op['operation'] == 'mutation':
                print(f"  Genome changes:")
                if op['population_type'] == 'solver':
                    before = op['genome_before'][:15]
                    after = op['genome_after'][:15]
                    move_names = ['N', 'E', 'S', 'W']
                    before_str = ''.join([move_names[m] for m in before])
                    after_str = ''.join([move_names[m] for m in after])
                    # Highlight differences
                    diff_str = ''.join(['*' if b != a else ' ' for b, a in zip(before, after)])
                    print(f"    Before: {before_str}...")
                    print(f"    After:  {after_str}...")
                    print(f"    Diff:   {diff_str}")
                else:
                    # Count changed cells for architects
                    changes = 0
                    before = op['genome_before']
                    after = op['genome_after']
                    for i in range(len(before)):
                        for j in range(len(before[i])):
                            if before[i][j] != after[i][j]:
                                changes += 1
                    print(f"    Changed {changes} cells in the grid")

        print(f"{'='*70}\n")
