"""
Improved solver strategies that can actually navigate mazes.
Provides smarter initialization and genome representations.
"""

import random
from typing import Tuple, List


def create_greedy_solver_genome(max_moves: int, start_pos: Tuple[int, int],
                                end_pos: Tuple[int, int]) -> List[int]:
    """
    Create a solver genome using greedy heuristic toward goal.

    This gives solvers a better starting point than pure random.
    The genome will mostly try to move toward the goal.

    Args:
        max_moves: Maximum number of moves
        start_pos: Starting position
        end_pos: Goal position

    Returns:
        List of moves with bias toward goal direction
    """
    genome = []

    for _ in range(max_moves):
        delta_row = end_pos[0] - start_pos[0]
        delta_col = end_pos[1] - start_pos[1]

        # Determine preferred directions
        preferred_moves = []

        if delta_row > 0:  # Goal is South
            preferred_moves.append(2)  # South
        elif delta_row < 0:  # Goal is North
            preferred_moves.append(0)  # North

        if delta_col > 0:  # Goal is East
            preferred_moves.append(1)  # East
        elif delta_col < 0:  # Goal is West
            preferred_moves.append(3)  # West

        # 70% chance to move toward goal, 30% random exploration
        if preferred_moves and random.random() < 0.7:
            move = random.choice(preferred_moves)
        else:
            move = random.randint(0, 3)

        genome.append(move)

    return genome


def create_random_walk_genome(max_moves: int) -> List[int]:
    """
    Create a random walk genome.
    Pure random - baseline strategy.
    """
    return [random.randint(0, 3) for _ in range(max_moves)]


def create_mixed_strategy_genome(max_moves: int, start_pos: Tuple[int, int],
                                 end_pos: Tuple[int, int]) -> List[int]:
    """
    Create a genome mixing greedy and random strategies.
    50% greedy toward goal, 50% random.
    """
    genome = []
    use_greedy = random.choice([True, False])

    if use_greedy:
        return create_greedy_solver_genome(max_moves, start_pos, end_pos)
    else:
        return create_random_walk_genome(max_moves)
