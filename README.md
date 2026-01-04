# Co-Evolutionary Maze Race

A competitive co-evolutionary algorithm where **Architects** evolve increasingly complex mazes while **Solvers** evolve smarter navigation strategies. This creates an "arms race" dynamic where both populations continuously adapt to each other.

## Project Overview

This project implements a **competitive co-evolution system** inspired by biological arms races (e.g., predator-prey evolution). Two populations compete:

- **Architects**: Evolve maze layouts to challenge solvers
- **Solvers**: Evolve navigation policies to solve mazes efficiently

Unlike traditional optimization where there's a fixed target, here the target continuously evolves. As solvers get better at navigating mazes, architects evolve more complex layouts. As mazes become harder, solvers develop better strategies.

## Key Features

### 1. Weight-Based Reactive Navigation
Solvers use a **reactive policy** with 4 learned weights instead of pre-planned paths:

```
Genome: [w_goal_dist, w_wall_penalty, w_visited_penalty, w_random_exploration]
```

At each step, the solver evaluates neighboring cells using:
```
Score(cell) = w_goal × (distance_improvement_to_goal)
            + w_wall × (is_wall)
            + w_visited × (visit_count)
            + w_random × (random_noise)
```

The cell with the highest score is chosen.

**Why this approach?**
- **Adaptive**: Weights define a policy that works across different mazes
- **Generalizable**: Same weights navigate various maze configurations
- **Evolvable**: Continuous weight space allows smooth evolution
- **Reactive**: No planning required, decisions made on-the-fly

**Weight Interpretation:**
- `w_goal_dist` (typically positive ~0.8-2.0): Attracts agent toward goal
- `w_wall_penalty` (typically negative ~-7.0): Strongly repels from walls
- `w_visited_penalty` (typically negative ~-1.3): Discourages revisiting cells
- `w_random_exploration` (typically small ~-0.5 to 0.5): Adds stochasticity

Evolution discovers that **wall avoidance** is critical (w_wall ≈ -7.0), more important than just moving toward the goal!

### 2. Multi-Objective Fitness Functions

**Architect Fitness** (create challenging but solvable mazes):
```python
fitness = difficulty_score × 2.0 + path_complexity + diversity_score × 0.5
```

Components:
- **Difficulty**: Rewards 30-70% solver failure rate (too easy or impossible is bad)
- **Path Complexity**: Rewards mazes where optimal path > Manhattan distance (forces detours)
- **Diversity**: Rewards 25-35% wall density (sweet spot for interesting mazes)
- **Solvability**: Hard constraint - unsolvable mazes get fitness = 0

**Solver Fitness** (reach goal efficiently):
```python
# If goal reached:
fitness = (goal_bonus × efficiency_ratio) + 50 - (extra_steps × 2.0)

# If goal not reached:
fitness = 10 × (progress_ratio²)
```

Components:
- **Goal Bonus**: Large reward (100) for success, scaled by efficiency
- **Efficiency Ratio**: `optimal_length / actual_length` (1.0 = perfect)
- **Path Penalty**: Each extra step beyond optimal costs 2 fitness points
- **Progress**: Quadratic reward for getting close if goal not reached

### 3. Curriculum Learning

Gradually increases maze complexity over generations:

```python
# Generations 0-100: Wall density increases from 15% → 45%
current_max_density = 0.15 + (generation / 100) × (0.45 - 0.15)
```

**Why curriculum learning?**
- Prevents architects from creating impossible mazes early on
- Gives solvers time to develop basic navigation skills
- Creates smoother co-evolution dynamics
- Mirrors natural evolution (simple → complex environments)

### 4. Competitive Co-Evolution Strategy

```python
# At generation t:
Architects(t) evaluated against Solvers(t-1)
Solvers(t) evaluated against Architects(t-1)
```

This ensures proper competitive dynamics where each population responds to the other's previous state.

### 5. Smart Initialization

**Solvers** start with heuristic-guided weights:
```python
w_goal_dist: 0.5 to 2.0      # Move toward goal
w_wall_penalty: -2.0 to -0.5  # Avoid walls
w_visited_penalty: -1.0 to 0.0 # Avoid revisiting
w_random: 0.0 to 0.5          # Some exploration
```

This gives evolution a better starting point than random weights.

## Project Structure

```
optimization-ai-project/
│
├── main.py                 # Main entry point and evolution loop
├── config.json            # All hyperparameters
│
├── src/
│   ├── environment.py     # Grid environment setup
│   ├── population.py      # Architect & Solver classes with fitness functions
│   ├── metrics.py         # Performance tracking and best individual selection
│   ├── visualization.py   # Dashboard and plot generation
│   ├── history_logger.py  # Evolution history tracking
│   └── utils.py          # A* pathfinding, simulation, solvability checks
│
└── output/               # Generated visualizations and metrics
    ├── dashboard.png
    ├── metrics.json
    ├── evolution_history.json
    └── genetic_operations.json
```

## Installation

```bash
# Install dependencies
pip install numpy matplotlib

# Run the co-evolution
python main.py
```

## Configuration

All parameters are in `config.json`:

```json
{
  "environment": {
    "grid_size": 10,
    "start_pos": [0, 0],
    "end_pos": [9, 9]
  },
  "architects": {
    "population_size": 20,
    "mutation_rate": 0.15,
    "crossover_rate": 0.8,
    "elite_size": 2,
    "tournament_size": 5
  },
  "solvers": {
    "population_size": 30,
    "max_moves": 200,
    "mutation_rate": 0.3,
    "crossover_rate": 0.8,
    "elite_size": 3,
    "tournament_size": 5,
    "weight_mutation_stddev": 0.3
  },
  "evolution": {
    "max_generations": 150,
    "curriculum_learning": {
      "enabled": true,
      "initial_max_wall_density": 0.15,
      "final_max_wall_density": 0.45,
      "transition_generations": 100
    }
  },
  "fitness": {
    "architect_maze_difficulty_weight": 2.0,
    "architect_diversity_weight": 0.5,
    "solver_goal_bonus": 100.0,
    "solver_path_efficiency_weight": 50.0,
    "solver_progress_weight": 10.0
  }
}
```

## Genetic Operators

### Architects (Binary Grid Genome)
- **Crossover**: Single-point crossover at random row
- **Mutation**: Bit-flip with 15% probability per cell
- **Constraints**: Ensures start/end are empty, maze is solvable, respects wall density limits

### Solvers (Weight Vector Genome)
- **Crossover**: Single-point crossover on weight vector
- **Mutation**: Gaussian noise (σ=0.3) added to each weight with 30% probability
- **Selection**: Tournament selection (size=5) for both populations

## Results

Typical evolution dynamics after 150 generations:

- **Wall Count**: 14 → 38 (mazes become more complex)
- **Success Rate**: 63% → 73% (fluctuates as populations adapt)
- **Path Complexity**: Optimal paths increase from ~18 to ~35 steps
- **Solver Weights**: Converge to strong wall avoidance (w_wall ≈ -7.0)
- **Fitness Evolution**: Clear co-evolutionary dynamics (not flat lines)

### Visualization

The dashboard (`output/dashboard.png`) shows:
1. **Fitness Over Time**: Arms race dynamics between populations
2. **Success Rate**: Percentage of solver-architect pairs where solver succeeds
3. **Path Length**: Average successful path length over generations
4. **Wall Count**: Maze complexity evolution
5. **Best Architect's Maze**: Most challenging evolved maze
6. **Best Solver's Path**: Solution on the hardest maze
7. **Optimal Solution**: A* reference path

## Design Decisions

### Why Fitness Variance Matters
Initially, solver fitness was constant (all ~1000) because:
- **Scale too large**: Variations were invisible
- **Averaging**: Testing all 20 architects smoothed out differences

**Solution**:
- Reduced fitness scale by 10× (100 instead of 1000)
- Sample 10 random architects instead of all 20
- Creates meaningful fitness variance and better evolution

### Why Path Penalty is Critical
Original fitness gave almost same reward to 100-step and 20-step paths.

**New approach**: Each extra step costs 2× fitness points, making efficiency truly matter:
```
20-step path: (100 × 1.0) + 50 - 0 = 150
100-step path: (100 × 0.2) + 50 - 160 = -90
Difference: 240 points! (was only 80)
```

### Why Best Solver Selection Was Fixed
Original `get_best_solver()` found a solver that succeeded on *any* maze, but visualized it on the *hardest* maze. Mismatch caused path to appear unsuccessful.

**Solution**: Find best solver specifically for the best architect's maze, ensuring visualization shows a successful path.

### Why No Solvability Bonus?
You might notice `architect_solvability_bonus` was removed from the config. This is because unsolvable mazes already get `fitness = 0` as a hard constraint. Adding a bonus for solvability would be redundant - all evaluated mazes are already solvable by definition!

### Path Complexity Metric
The variable is called `manhattan_distance` (not `max_possible_length`) because it represents the straight-line distance from start to goal. When we compute `optimal_length / manhattan_distance`, we're measuring how much the maze forces detours. A ratio > 1.0 means the optimal path is longer than the Manhattan distance, indicating a more interesting maze.

## Key Insights

1. **Co-evolution creates complexity**: Mazes evolve from simple (14 walls) to complex (38 walls)
2. **Wall avoidance dominates**: Evolved solvers have w_wall ≈ -7.0 (strongest weight)
3. **Curriculum learning essential**: Gradual difficulty ramp prevents premature convergence
4. **Fitness scale matters**: Small differences need to be visible for selection pressure
5. **Sampling creates variance**: Random subsets prevent over-smoothing of fitness signals

## Future Enhancements

Potential improvements:
- **Larger grids**: Test scalability to 20×20 or 50×50 mazes
- **Multi-objective optimization**: Pareto front for architect objectives
- **Neural network solvers**: Replace weights with small neural networks
- **Dynamic obstacles**: Moving walls or time-varying mazes
- **Multiple goals**: Mazes with waypoints or multiple exits
- **Transfer learning**: Test if evolved weights generalize to new maze sizes

## References

This project demonstrates concepts from:
- Competitive co-evolution and arms races
- Genetic algorithms with tournament selection and elitism
- Reactive agent architectures
- Curriculum learning in evolutionary systems
- Multi-objective fitness optimization
