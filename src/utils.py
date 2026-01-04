import json
import heapq
from typing import List, Tuple, Optional, Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """Manhattan distance heuristic for A* algorithm."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def astar_pathfind(grid: List[List[int]],
                   start: Tuple[int, int],
                   end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm to find shortest path in a grid.

    Args:
        grid: 2D grid where 0=empty, 1=wall
        start: Starting position (row, col)
        end: Goal position (row, col)

    Returns:
        List of positions forming the path, or None if no path exists
    """
    rows, cols = len(grid), len(grid[0])

    # Validate start and end positions
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return None
    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        return None

    # Priority queue: (f_score, counter, position, path)
    counter = 0
    heap = [(heuristic(start, end), counter, start, [start])]
    visited = set()

    while heap:
        f_score, _, current, path = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        # Check if we reached the goal
        if current == end:
            return path

        # Explore neighbors (N, E, S, W)
        row, col = current
        neighbors = [
            (row - 1, col),  # North
            (row, col + 1),  # East
            (row + 1, col),  # South
            (row, col - 1)   # West
        ]

        for next_pos in neighbors:
            next_row, next_col = next_pos

            # Check bounds
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue

            # Check if it's a wall or already visited
            if grid[next_row][next_col] == 1 or next_pos in visited:
                continue

            # Calculate new scores
            new_g_score = len(path)
            new_f_score = new_g_score + heuristic(next_pos, end)

            counter += 1
            new_path = path + [next_pos]
            heapq.heappush(heap, (new_f_score, counter, next_pos, new_path))

    # No path found
    return None


def is_solvable(grid: List[List[int]],
                start: Tuple[int, int],
                end: Tuple[int, int]) -> bool:
    """Check if a grid maze is solvable using A* pathfinding."""
    path = astar_pathfind(grid, start, end)
    return path is not None


def get_path_length(grid: List[List[int]],
                    start: Tuple[int, int],
                    end: Tuple[int, int]) -> int:
    """
    Get the length of the shortest path, or 0 if no path exists.

    Returns:
        Path length (number of steps) or 0 if unsolvable
    """
    path = astar_pathfind(grid, start, end)
    if path is None:
        return 0
    return len(path) - 1  # Number of steps (-1 because i'm counting "edges", not nodes)


def simulate_solver_path(grid: List[List[int]],
                        weights: List[float],
                        start: Tuple[int, int],
                        end: Tuple[int, int] = None,
                        max_moves: int = 150) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Simulate a solver's path through the grid using weight-based reactive navigation.

    The solver evaluates neighboring cells at each step using weights:
    Score = w_goal * (goal_distance_improvement) + w_wall * (is_wall) + w_visited * (visit_count) + w_random * (random_noise)

    Args:
        grid: 2D grid where 0=empty, 1=wall
        weights: List of 4 floats [w_goal_dist, w_wall_penalty, w_visited_penalty, w_random_exploration]
        start: Starting position (row, col)
        end: Goal position (row, col) - required for goal-based scoring
        max_moves: Maximum number of moves to simulate

    Returns:
        Tuple of (path taken as list of positions, final position)
    """
    import random

    if end is None:
        # Fallback: if no goal specified, can't use reactive navigation properly
        # Just return start position
        return [start], start

    rows, cols = len(grid), len(grid[0])
    current = start
    path = [current]
    visited_count = {current: 1}

    # Extract weights
    w_goal, w_wall, w_visited, w_random = weights

    # Direction mappings: 0=N, 1=E, 2=S, 3=W
    directions = [
        (-1, 0),  # North
        (0, 1),   # East
        (1, 0),   # South
        (0, -1)   # West
    ]

    for _ in range(max_moves):
        # Check if we reached the goal
        if current == end:
            break

        # Calculate current distance to goal
        current_dist = abs(current[0] - end[0]) + abs(current[1] - end[1])

        # Evaluate all 4 neighbors
        scores = []
        for direction_idx, (delta_row, delta_col) in enumerate(directions):
            next_row = current[0] + delta_row
            next_col = current[1] + delta_col
            next_pos = (next_row, next_col)

            # Check bounds
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                scores.append((float('-inf'), direction_idx, next_pos))  # Invalid
                continue

            # Calculate score components
            is_wall = grid[next_row][next_col]

            # Goal distance improvement (negative = closer to goal)
            next_dist = abs(next_row - end[0]) + abs(next_col - end[1])
            goal_delta = current_dist - next_dist  # Positive if getting closer

            # Visit count
            visit_count = visited_count.get(next_pos, 0)

            # Random exploration
            random_noise = random.random()

            # Calculate total score
            score = (w_goal * goal_delta +
                    w_wall * is_wall +
                    w_visited * visit_count +
                    w_random * random_noise)

            scores.append((score, direction_idx, next_pos))

        # Pick the highest scoring move
        best_score, best_direction, best_pos = max(scores, key=lambda x: x[0])

        # If best move is invalid (out of bounds), stop
        if best_score == float('-inf'):
            break

        # If best move is a wall, stay in place (or could stop)
        if grid[best_pos[0]][best_pos[1]] == 1:
            # Don't move, but count this as a step
            continue

        # Make the move
        current = best_pos
        path.append(current)
        visited_count[current] = visited_count.get(current, 0) + 1

    return path, current


def count_walls(grid: List[List[int]]) -> int:
    """
    Count the number of walls in a maze grid.

    Args:
        grid: 2D grid where 1=wall, 0=empty

    Returns:
        Number of walls in the grid
    """
    return sum(sum(row) for row in grid)
