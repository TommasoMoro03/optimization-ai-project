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
                        moves: List[int],
                        start: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], bool]:
    """
    Simulate a solver's path through the grid based on its move sequence.

    Args:
        grid: 2D grid where 0=empty, 1=wall
        moves: List of move directions (0=N, 1=E, 2=S, 3=W)
        start: Starting position (row, col)

    Returns:
        Tuple of (path taken as list of positions, whether goal was reached)
    """
    rows, cols = len(grid), len(grid[0])
    current = start
    path = [current]

    # Direction mappings: 0=N, 1=E, 2=S, 3=W
    directions = [
        (-1, 0),  # North
        (0, 1),   # East
        (1, 0),   # South
        (0, -1)   # West
    ]

    for move in moves:
        if move < 0 or move >= 4:
            continue  # Invalid move

        delta_row, delta_col = directions[move]
        next_row = current[0] + delta_row
        next_col = current[1] + delta_col

        # Check bounds
        if not (0 <= next_row < rows and 0 <= next_col < cols):
            continue  # Out of bounds, stay in place

        # Check if it's a wall
        if grid[next_row][next_col] == 1:
            continue  # Hit a wall, stay in place

        current = (next_row, next_col)
        path.append(current)

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
