import random
import numpy as np
from src.utils import load_config
from src.environment import GridEnvironment
from src.population import Architect, Solver
from src.evolution import evolve_architects, evolve_solvers
from src.metrics import MetricsTracker
from src.visualization import Dashboard
from src.history_logger import HistoryLogger


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
    initial_wall_density = curriculum_config.get('initial_max_wall_density', 0.10) if use_curriculum else None

    # Create architects with initial wall density from curriculum learning
    architects = [
        Architect(env.grid_size, env.start_pos, env.end_pos, max_wall_density=initial_wall_density)
        for _ in range(config['architects']['population_size'])
    ]

    if use_curriculum and initial_wall_density is not None:
        print(f"  Applied curriculum learning: initial max wall density = {initial_wall_density:.2f}")

    # Initialize solvers using use_smart_init (reasonable starting values for the weights)
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
        # Store previous generation populations for proper co-evolution
        # Each population is evaluated against the other population from t-1
        prev_architects = architects[:]
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

        # Record metrics and genome history (with updated populations)
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
