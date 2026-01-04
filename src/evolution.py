import random
from typing import List
from src.population import Architect, Solver
from src.history_logger import HistoryLogger


def evolve_architects(architects: List[Architect],
                     solvers: List[Solver],
                     config: dict,
                     history_logger: HistoryLogger = None,
                     generation: int = 0,
                     max_wall_density: float = None) -> List[Architect]:
    """
    Evolve the architect population for one generation using Genetic Algorithm (GA).

    Uses tournament selection, crossover, and mutation.

    Args:
        architects: Current architect population
        solvers: Current solver population (for fitness evaluation)
        config: Configuration dictionary
        history_logger: Optional history logger for tracking genetic operations
        generation: Current generation number
        max_wall_density: Maximum wall density for curriculum learning

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
        # Selection using tournament selection
        parent1 = Architect.tournament_selection(architects, tournament_size)
        parent2 = Architect.tournament_selection(architects, tournament_size)

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
            child1 = Architect(parent1.grid_size, parent1.start_pos, parent1.end_pos, max_wall_density)
            child1.genome = [row[:] for row in parent1.genome]
            child2 = Architect(parent2.grid_size, parent2.start_pos, parent2.end_pos, max_wall_density)
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
    Evolve the solver population using Evolutionary Strategies (ES).

    ES is more appropriate for continuous-valued optimization (weight vectors)
    than traditional Genetic Algorithms. Uses:
    - Truncated selection (deterministic, keep top performers)
    - No crossover (not effective for continuous optimization)
    - Gaussian mutation only
    - (μ + λ) strategy

    Args:
        solvers: Current solver population
        architects: Current architect population (for fitness evaluation)
        end_pos: Goal position
        config: Configuration dictionary
        history_logger: Optional history logger for tracking genetic operations
        generation: Current generation number

    Returns:
        New solver population
    """
    # Evaluate fitness (test against a sample of architects for better variance)
    # Using a sample instead of all architects creates more fitness diversity
    num_test_architects = min(10, len(architects))  # Test against 10 random architects

    for solver in solvers:
        # Test against a random sample of architects
        test_architects = random.sample(architects, num_test_architects)
        fitness_sum = 0
        for architect in test_architects:
            fitness = solver.calculate_fitness(architect, end_pos, config)
            fitness_sum += fitness
        solver.fitness = fitness_sum / num_test_architects  # Normalize by sample size

    # Sort by fitness (descending)
    solvers.sort(key=lambda x: x.fitness, reverse=True)

    # EVOLUTIONARY STRATEGY: (μ + λ) strategy
    # Keep top μ parents + generate λ offspring, then selection
    population_size = config['solvers']['population_size']
    selection_size = config['solvers']['selection_size']  # Number of parents (μ)
    mutation_stddev = config['solvers'].get('weight_mutation_stddev', 0.3)

    # Keep top performers as parents (truncated selection)
    parents = solvers[:selection_size]

    # Generate offspring by mutating parents
    offspring = []
    offspring_count = 0
    offspring_needed = population_size - selection_size  # λ offspring

    while len(offspring) < offspring_needed:
        # Select parent deterministically from top performers (round-robin)
        parent = parents[offspring_count % len(parents)]

        # Create offspring by cloning and mutating parent
        child = Solver(parent.max_moves, parent.start_pos, parent.end_pos,
                      use_smart_init=False)
        child.genome = parent.genome[:]

        # Log first few mutations for verification
        if history_logger and len(offspring) < 5:
            genome_before = child.genome[:]
            child.mutate(1.0, mutation_stddev)  # Always mutate in ES
            history_logger.log_mutation_event(
                generation, 'solver',
                genome_before, child.genome
            )
        else:
            child.mutate(1.0, mutation_stddev)  # mutation_rate = 1.0 (always mutate)

        # Evaluate offspring fitness immediately
        test_architects = random.sample(architects, num_test_architects)
        fitness_sum = 0
        for architect in test_architects:
            fitness = child.calculate_fitness(architect, end_pos, config)
            fitness_sum += fitness
        child.fitness = fitness_sum / num_test_architects

        offspring.append(child)
        offspring_count += 1

    # (μ + λ): Combine parents and offspring, then select best μ+λ for next generation
    # This ensures we keep the evaluated fitnesses
    combined = parents + offspring
    combined.sort(key=lambda x: x.fitness, reverse=True)

    return combined[:population_size]
