import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
from src.population import Architect, Solver
from src.utils import simulate_solver_path, astar_pathfind
import os


class Dashboard:
    """
    Creates visualization dashboards for the co-evolution process.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dashboard with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = config['output']['plots_dir']
        self.format = config['output']['plot_format']

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_full_dashboard(self,
                             metrics_history: Dict[str, List],
                             best_architect: Architect,
                             best_solver: Solver):
        """
        Create a comprehensive dashboard with all visualizations.

        Args:
            metrics_history: Dictionary containing metrics history
            best_architect: Best architect from final generation
            best_solver: Best solver from final generation
        """
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Fitness over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_race(ax1, metrics_history)

        # 2. Success Rate over time
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_success_rate(ax2, metrics_history)

        # 3. Path Length over time
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_path_length(ax3, metrics_history)

        # 4. Wall Count over time
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_wall_count(ax4, metrics_history)

        # 5. Best Map Visualization
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_maze(ax5, best_architect, "Best Architect's Maze")

        # 6. Best Solver Path
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_solver_path(ax6, best_architect, best_solver, "Best Solver's Path")

        # 7. Optimal Path
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_optimal_path(ax7, best_architect, "Optimal Solution (A*)")

        # Main title
        fig.suptitle('Co-Evolutionary Race Dashboard', fontsize=16, fontweight='bold')

        # Save figure
        if self.config['output']['save_plots']:
            filepath = os.path.join(self.output_dir, f'dashboard.{self.format}')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nDashboard saved to: {filepath}")

        plt.show()

    def _plot_race(self, ax, metrics_history: Dict[str, List]):
        """Plot the race between architects and solvers."""
        generations = metrics_history['generation']

        # Plot both average and max fitness
        ax.plot(generations, metrics_history['avg_architect_fitness'],
               label='Avg Architect Fitness', color='red', linewidth=2, alpha=0.7)
        ax.plot(generations, metrics_history['max_architect_fitness'],
               label='Max Architect Fitness', color='darkred', linewidth=2, linestyle='--')

        ax.plot(generations, metrics_history['avg_solver_fitness'],
               label='Avg Solver Fitness', color='blue', linewidth=2, alpha=0.7)
        ax.plot(generations, metrics_history['max_solver_fitness'],
               label='Max Solver Fitness', color='darkblue', linewidth=2, linestyle='--')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Arms Race: Architect vs Solver Fitness Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_success_rate(self, ax, metrics_history: Dict[str, List]):
        """Plot solver success rate over time."""
        generations = metrics_history['generation']
        success_rate = metrics_history['solver_success_rate']

        ax.plot(generations, success_rate, color='green', linewidth=2)
        ax.fill_between(generations, success_rate, alpha=0.3, color='green')

        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.set_title('Solver Success Rate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    def _plot_path_length(self, ax, metrics_history: Dict[str, List]):
        """Plot average path length over time."""
        generations = metrics_history['generation']
        path_length = metrics_history['avg_path_length']

        ax.plot(generations, path_length, color='purple', linewidth=2)
        ax.fill_between(generations, path_length, alpha=0.3, color='purple')

        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Path Length', fontsize=10)
        ax.set_title('Average Path Length', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_wall_count(self, ax, metrics_history: Dict[str, List]):
        """Plot average wall count over time."""
        generations = metrics_history['generation']
        wall_count = metrics_history['avg_wall_count']

        ax.plot(generations, wall_count, color='orange', linewidth=2)
        ax.fill_between(generations, wall_count, alpha=0.3, color='orange')

        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Number of Walls', fontsize=10)
        ax.set_title('Average Wall Count', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_maze(self, ax, architect: Architect, title: str):
        """Visualize a maze."""
        grid = np.array(architect.genome)

        # Create color map: white=empty, black=wall
        cmap = plt.cm.colors.ListedColormap(['white', 'black'])

        ax.imshow(grid, cmap=cmap, interpolation='nearest')

        # Mark start and end
        start = architect.start_pos
        end = architect.end_pos

        ax.plot(start[1], start[0], 'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(end[1], end[0], 'r*', markersize=20, label='End', markeredgecolor='darkred', markeredgewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_solver_path(self, ax, architect: Architect, solver: Solver, title: str):
        """Visualize the path taken by a solver."""
        grid = np.array(architect.genome)
        cmap = plt.cm.colors.ListedColormap(['white', 'black'])

        ax.imshow(grid, cmap=cmap, interpolation='nearest')

        # Get solver's path
        path, final_pos = simulate_solver_path(architect.genome, solver.genome, architect.start_pos)

        # Plot path
        if len(path) > 1:
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, alpha=0.6, label='Solver Path')

        # Mark start, end, and final position
        start = architect.start_pos
        end = architect.end_pos

        ax.plot(start[1], start[0], 'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(end[1], end[0], 'r*', markersize=20, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        ax.plot(final_pos[1], final_pos[0], 'bs', markersize=12, label='Solver End', markeredgecolor='darkblue', markeredgewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_optimal_path(self, ax, architect: Architect, title: str):
        """Visualize the optimal path using A*."""
        grid = np.array(architect.genome)
        cmap = plt.cm.colors.ListedColormap(['white', 'black'])

        ax.imshow(grid, cmap=cmap, interpolation='nearest')

        # Get optimal path using A*
        optimal_path = astar_pathfind(architect.genome, architect.start_pos, architect.end_pos)

        # Plot path
        if optimal_path:
            path_array = np.array(optimal_path)
            ax.plot(path_array[:, 1], path_array[:, 0], 'g-', linewidth=2, alpha=0.6, label='A* Path')

        # Mark start and end
        start = architect.start_pos
        end = architect.end_pos

        ax.plot(start[1], start[0], 'go', markersize=15, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(end[1], end[0], 'r*', markersize=20, label='End', markeredgecolor='darkred', markeredgewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    def save_individual_plots(self, metrics_history: Dict[str, List],
                             best_architect: Architect, best_solver: Solver):
        """
        Save individual plots separately.

        Args:
            metrics_history: Dictionary containing metrics history
            best_architect: Best architect from final generation
            best_solver: Best solver from final generation
        """
        # Arms race plot
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_race(ax, metrics_history)
        plt.savefig(os.path.join(self.output_dir, f'race.{self.format}'), dpi=300, bbox_inches='tight')
        plt.close()

        # Best maze
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_maze(ax, best_architect, "Best Architect's Maze")
        plt.savefig(os.path.join(self.output_dir, f'best_maze.{self.format}'), dpi=300, bbox_inches='tight')
        plt.close()

        # Best solver path
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_solver_path(ax, best_architect, best_solver, "Best Solver's Path")
        plt.savefig(os.path.join(self.output_dir, f'solver_path.{self.format}'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Individual plots saved to: {self.output_dir}")
