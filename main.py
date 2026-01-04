"""
Main script for running the Co-Evolutionary Race.
"""

import random
import numpy as np
from typing import List
from src.utils import load_config
from src.environment import GridEnvironment
from src.population import Architect, Solver, tournament_selection
from src.metrics import MetricsTracker
from src.visualization import Dashboard
from src.history_logger import HistoryLogger


def evolve_architects(architects: List[Architect],
                     solvers: List[Solver],
                     config: dict,
                     history_logger: HistoryLogger = None,
                     generation: int = 0,
                     max_wall_density: float = None) -> List[Architect]:
    """
    Evolve the architect population for one generation.

    Args:
        architects: Current architect population
        solvers: Current solver population (for fitness evaluation)
        config: Configuration dictionary

    Returns:
        New architect population
    """
    # Evaluate fitness
    for architect in architects:
        architect.calculate_fitness(solvers, config)

    # Sort by fitness
    architects.sort(key=lambda x: x.fitness, reverse=True)

    # Elitism: keep top performers
    elite_size = config['architects']['elite_size']
    new_population = architects[:elite_size]

    # Generate rest of population through selection, crossover, and mutation
    population_size = config['architects']['population_size']
    tournament_size = config['architects']['tournament_size']
    crossover_rate = config['architects']['crossover_rate']
    mutation_rate = config['architects']['mutation_rate']

    while len(new_population) < population_size:
        # Selection
        parent1 = tournament_selection(architects, tournament_size)
        parent2 = tournament_selection(architects, tournament_size)

        # Crossover
        if random.random() < crossover_rate:
            # Log first few crossovers for verification
            if history_logger and len(new_population) < elite_size + 3:
                p1_genome = [row[:] for row in parent1.genome]
                p2_genome = [row[:] for row in parent2.genome]

            child1, child2 = Architect.crossover(parent1, parent2, max_wall_density)

            if history_logger and len(new_population) < elite_size + 3:
                history_logger.log_crossover_event(
                    generation, 'architect',
                    p1_genome, p2_genome,
                    child1.genome, child2.genome
                )
        else:
            # Clone parents if no crossover
            child1 = Architect(parent1.grid_size, parent1.start_pos, parent1.end_pos)
            child1.genome = [row[:] for row in parent1.genome]
            child2 = Architect(parent2.grid_size, parent2.start_pos, parent2.end_pos)
            child2.genome = [row[:] for row in parent2.genome]

        # Mutation
        # Log first few mutations for verification
        if history_logger and len(new_population) < elite_size + 3:
            genome_before = [row[:] for row in child1.genome]
            child1.mutate(mutation_rate, max_wall_density)
            history_logger.log_mutation_event(
                generation, 'architect',
                genome_before, child1.genome
            )
        else:
            child1.mutate(mutation_rate, max_wall_density)

        child2.mutate(mutation_rate, max_wall_density)

        # Add to new population
        new_population.append(child1)
        if len(new_population) < population_size:
            new_population.append(child2)

    return new_population[:population_size]


def evolve_solvers(solvers: List[Solver],
                  architects: List[Architect],
                  end_pos: tuple,
                  config: dict,
                  history_logger: HistoryLogger = None,
                  generation: int = 0) -> List[Solver]:
    """
    Evolve the solver population for one generation.

    Args:
        solvers: Current solver population
        architects: Current architect population (for fitness evaluation)
        end_pos: Goal position
        config: Configuration dictionary

    Returns:
        New solver population
    """
    # Evaluate fitness (test against a sample of architects for better variance)
    # Using a sample instead of all architects creates more fitness diversity
    num_test_architects = min(10, len(architects))  # Test against 10 random architects

    for solver in solvers:
        # Test against a random sample of architects and SUM the fitness
        # SUM (not mean) creates more variance between good and bad solvers
        test_architects = random.sample(architects, num_test_architects)
        fitness_sum = 0
        for architect in test_architects:
            fitness = solver.calculate_fitness(architect, end_pos, config)
            fitness_sum += fitness
        solver.fitness = fitness_sum / num_test_architects  # Normalize by sample size

    # Sort by fitness
    solvers.sort(key=lambda x: x.fitness, reverse=True)

    # Elitism: keep top performers
    elite_size = config['solvers']['elite_size']
    new_population = solvers[:elite_size]

    # Generate rest of population
    population_size = config['solvers']['population_size']
    tournament_size = config['solvers']['tournament_size']
    crossover_rate = config['solvers']['crossover_rate']
    mutation_rate = config['solvers']['mutation_rate']
    mutation_stddev = config['solvers'].get('weight_mutation_stddev', 0.3)

    while len(new_population) < population_size:
        # Selection
        parent1 = tournament_selection(solvers, tournament_size)
        parent2 = tournament_selection(solvers, tournament_size)

        # Crossover
        if random.random() < crossover_rate:
            # Log first few crossovers for verification
            if history_logger and len(new_population) < elite_size + 3:
                p1_genome = parent1.genome[:]
                p2_genome = parent2.genome[:]

            child1, child2 = Solver.crossover(parent1, parent2)

            if history_logger and len(new_population) < elite_size + 3:
                history_logger.log_crossover_event(
                    generation, 'solver',
                    p1_genome, p2_genome,
                    child1.genome, child2.genome
                )
        else:
            # Clone parents if no crossover
            child1 = Solver(parent1.max_moves, parent1.start_pos, parent1.end_pos,
                           use_smart_init=False)
            child1.genome = parent1.genome[:]
            child2 = Solver(parent2.max_moves, parent2.start_pos, parent2.end_pos,
                           use_smart_init=False)
            child2.genome = parent2.genome[:]

        # Mutation
        # Log first few mutations for verification
        if history_logger and len(new_population) < elite_size + 3:
            genome_before = child1.genome[:]
            child1.mutate(mutation_rate, mutation_stddev)
            history_logger.log_mutation_event(
                generation, 'solver',
                genome_before, child1.genome
            )
        else:
            child1.mutate(mutation_rate, mutation_stddev)

        child2.mutate(mutation_rate, mutation_stddev)

        # Add to new population
        new_population.append(child1)
        if len(new_population) < population_size:
            new_population.append(child2)

    return new_population[:population_size]


def run_coevolution(config: dict):
    """
    Main co-evolution loop.

    Args:
        config: Configuration dictionary
    """
    print("="*60)
    print("CO-EVOLUTIONARY RACE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Grid Size: {config['environment']['grid_size']}x{config['environment']['grid_size']}")
    print(f"  Architect Population: {config['architects']['population_size']}")
    print(f"  Solver Population: {config['solvers']['population_size']}")
    print(f"  Max Generations: {config['evolution']['max_generations']}")
    print("="*60)

    # Initialization of the environment
    env = GridEnvironment(config)

    # Initialize populations
    print("\nInitializing populations...")
    # Curriculum learning: gradually increase max wall density
    curriculum_config = config['evolution'].get('curriculum_learning', {})
    use_curriculum = curriculum_config.get('enabled', True)
    initial_wall_density = curriculum_config.get('initial_max_wall_density', 0.15) if use_curriculum else None
    
    architects = [
        Architect(env.grid_size, env.start_pos, env.end_pos)
        for _ in range(config['architects']['population_size'])
    ]
    
    # Enforce initial wall density for curriculum learning
    if use_curriculum and initial_wall_density is not None:
        for arch in architects:
            arch.enforce_max_wall_density(initial_wall_density)
        print(f"  Applied curriculum learning: initial max wall density = {initial_wall_density:.2f}")

    # Initialize solvers with smart greedy heuristic
    print("  Using smart initialization for solvers (greedy toward goal)...")
    solvers = [
        Solver(config['solvers']['max_moves'], env.start_pos, env.end_pos,
               use_smart_init=True)
        for _ in range(config['solvers']['population_size'])
    ]

    # Initialize metrics tracker and history logger
    metrics = MetricsTracker()
    history = HistoryLogger()

    # Co-evolution loop
    max_generations = config['evolution']['max_generations']

    # Get curriculum learning parameters (already loaded above)
    final_wall_density = curriculum_config.get('final_max_wall_density', 0.45) if use_curriculum else None
    transition_gens = curriculum_config.get('transition_generations', 50) if use_curriculum else 50

    print("\nStarting co-evolution...\n")
    print("CO-EVOLUTION STRATEGY:")
    print("  - Architects(t) evaluated against Solvers(t-1)")
    print("  - Solvers(t) evaluated against Architects(t-1)")
    print("  - This ensures proper competitive co-evolution")
    if use_curriculum:
        print(f"\nCURRICULUM LEARNING: Enabled")
        print(f"  - Initial max wall density: {initial_wall_density:.2f}")
        print(f"  - Final max wall density: {final_wall_density:.2f}")
        print(f"  - Transition over {transition_gens} generations\n")
    else:
        print()

    for generation in range(max_generations):
        # IMPORTANT: Store previous generation populations for proper co-evolution
        # Each population should be evaluated against the OTHER population from t-1
        prev_architects = architects[:]  # Shallow copy of list (individuals are references)
        prev_solvers = solvers[:]

        # Calculate current max wall density (curriculum learning)
        if use_curriculum:
            if generation < transition_gens:
                # Linear interpolation from initial to final
                progress = generation / transition_gens
                max_wall_density = initial_wall_density + (final_wall_density - initial_wall_density) * progress
            else:
                max_wall_density = final_wall_density
        else:
            max_wall_density = None
        
        # Enforce max wall density on existing architects (for curriculum learning)
        if max_wall_density is not None:
            for arch in architects:
                arch.enforce_max_wall_density(max_wall_density)

        # Evolve architects using solvers from PREVIOUS generation (t-1)
        architects = evolve_architects(architects, prev_solvers, config, history, generation, max_wall_density)

        # Evolve solvers using architects from PREVIOUS generation (t-1)
        solvers = evolve_solvers(solvers, prev_architects, env.end_pos, config, history, generation)

        # Record metrics and genome history (with UPDATED populations)
        metrics.record_generation(generation, architects, solvers, env.end_pos)
        history.log_generation(generation, architects, solvers)

        # Print progress every 10 generations
        if generation % 10 == 0 or generation == max_generations - 1:
            metrics.print_generation_summary(generation)

        # Print genome summary for first and last generations
        if generation == 0 or generation == max_generations - 1:
            history.print_genome_summary(generation)

    # Final summary
    print("\n" + "="*60)
    print("CO-EVOLUTION COMPLETE")
    print("="*60)

    final_summary = metrics.get_final_summary()
    print(f"\nFinal Results:")
    print(f"  Best Architect Fitness: {final_summary['final_max_architect_fitness']:.2f}")
    print(f"  Best Solver Fitness: {final_summary['final_max_solver_fitness']:.2f}")
    print(f"  Final Success Rate: {final_summary['final_success_rate']:.2f}%")
    print(f"  Final Avg Path Length: {final_summary['final_avg_path_length']:.2f}")
    print("="*60)

    # Get best individuals
    best_architect = metrics.get_best_architect(architects)
    best_solver = metrics.get_best_solver(solvers, architects, env.end_pos)

    # Create visualizations
    print("\nGenerating visualizations...")
    dashboard = Dashboard(config)

    # Save metrics and history
    metrics_path = f"{config['output']['plots_dir']}/metrics.json"
    metrics.save_metrics(metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    # Save evolution history
    history.save_history(f"{config['output']['plots_dir']}/evolution_history.json")
    history.save_genetic_operations(f"{config['output']['plots_dir']}/genetic_operations.json")

    # Print genetic operations summary
    print("\nGenetic Operations Sample:")
    history.print_genetic_operations_summary(num_operations=5)

    # Create dashboard
    dashboard.create_full_dashboard(
        metrics.get_history(),
        best_architect,
        best_solver
    )

    # Save individual plots
    dashboard.save_individual_plots(
        metrics.get_history(),
        best_architect,
        best_solver
    )

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load configuration (parameter values)
    config = load_config("config.json")

    # Run co-evolution
    run_coevolution(config)


if __name__ == "__main__":
    main()
