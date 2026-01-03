# Co-Evolutionary Arms Race Optimization

A Python implementation of a co-evolutionary optimization system where two populations compete:
- **Architects**: Evolve increasingly complex solvable mazes
- **Solvers**: Evolve strategies to navigate through the mazes

## Project Structure

```
optimization-ai-project/
├── config.json           # Configuration parameters
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
├── COEVOLUTION_LOGIC.md # Detailed explanation of co-evolution logic and code flow
├── src/
│   ├── __init__.py
│   ├── environment.py    # 2D Grid environment
│   ├── population.py     # Architect and Solver classes
│   ├── utils.py          # A* pathfinding and utilities
│   ├── metrics.py        # Metrics tracking
│   ├── visualization.py  # Dashboard generation
│   └── history_logger.py # Detailed genome tracking
├── verify_genetics.py    # Verification script for genetic operations
└── output/               # Generated plots and metrics (created at runtime)
```

## Features

- **Configuration-Driven**: All parameters defined in `config.json`
- **Modular Design**: Clean separation of concerns across modules
- **A* Validation**: Ensures all architect mazes are solvable
- **Comprehensive Metrics**: Tracks fitness, success rates, path lengths, and complexity
- **Rich Visualizations**: Matplotlib dashboard showing the "arms race" evolution
- **Detailed Logging**: Complete genome history with crossover/mutation tracking

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Co-Evolution

```bash
python3 main.py
```

The system will:
1. Initialize architect and solver populations
2. Run co-evolution for the configured number of generations
3. Track metrics and detailed genome data throughout the process
4. Generate visualizations in the `output/` directory

This will display visual examples of:
- Solver crossover (showing parent and child move sequences)
- Solver mutation (showing before/after genomes)
- Architect crossover (showing parent and child maze patterns)
- Architect mutation (showing cell changes in mazes)

## Configuration

Edit `config.json` to adjust:
- **Environment**: Grid size, start/end positions
- **Architects**: Population size, mutation rate, crossover rate
- **Solvers**: Population size, max moves, mutation rate
- **Evolution**: Number of generations, evolution cycles per generation
- **Fitness**: Weights for different fitness components

## How It Works


### Quick Summary

**Co-Evolution Strategy (t-1 evaluation):**
- Architects(t) are evaluated against Solvers(t-1)
- Solvers(t) are evaluated against Architects(t-1)
- This ensures symmetric competition

**Architects:**
- **Genome**: Binary 2D grid (0=empty, 1=wall)
- **Constraint**: Must be solvable (validated via A* algorithm)
- **Fitness**: Reward for complex paths and solver failures

**Solvers:**
- **Genome**: Sequence of moves (0=North, 1=East, 2=South, 3=West)
- **Fitness**: Reward for reaching the goal and path efficiency

**Evolution Process:**
1. Architects evolve to create harder mazes (evaluated against previous solvers)
2. Solvers evolve to solve mazes (evaluated against previous architects)
3. Repeat for 100 generations, creating an "arms race" dynamic

## Output

The system generates:

### Visualizations
- `output/dashboard.png`: Comprehensive visualization dashboard
- `output/arms_race.png`: Fitness evolution over time
- `output/best_maze.png`: Final best maze
- `output/solver_path.png`: Best solver's path

### Data Files
- `output/metrics.json`: Statistical metrics for each generation
- `output/evolution_history.json`: **Complete genome data for all generations**
  - Contains exact genomes (move sequences and maze grids) for best individuals
  - Fitness values for architects and solvers
  - Population statistics
- `output/genetic_operations.json`: **Detailed crossover/mutation logs**
  - Shows exact parent and child genomes
  - Demonstrates genetic operations are working correctly

### Evolution History Structure

The `evolution_history.json` file contains an array of generation objects:

```json
[
  {
    "generation_id": 0,
    "best_architect_genome": [[0, 1, 0, ...], [1, 0, 1, ...], ...],
    "best_architect_fitness": 45.2,
    "best_solver_genome": [0, 1, 2, 3, 0, 1, ...],
    "best_solver_fitness": 98.5,
    "avg_architect_fitness": 32.1,
    "avg_solver_fitness": 67.3,
    "num_architects": 10,
    "num_solvers": 20
  },
  ...
]
```
