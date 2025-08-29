"""
ARC (Abstraction and Reasoning Corpus) Neural Program Generation Package

This package provides data structures and utilities for working with the
ARC dataset for neural program generation research.

Main Classes:
    ARCGrid: Represents a single grid with analysis methods
    ARCExample: Represents an input-output pair 
    ARCTask: Represents a complete task with training examples and test case
    ARCDataset: Main class for loading and managing the entire dataset

Analysis Functions:
    analyze_grid_sizes: Analyze distribution of grid sizes
    detailed_grid_size_analysis: Comprehensive grid size analysis
    find_tasks_by_color_count: Find tasks using specific number of colors
    find_shape_preserving_tasks: Find tasks where shapes don't change
    analyze_size_categories: Categorize grids by size ranges
    find_size_outliers: Find unusually large or small grids
    find_30x30_grids: Find all maximum-size grids
"""

from .grid import ARCGrid, ARCExample
from .task import ARCTask
from .dataset import ARCDataset
from .analysis import (
    analyze_grid_sizes,
    detailed_grid_size_analysis,
    find_tasks_by_color_count,
    find_shape_preserving_tasks,
    analyze_size_categories,
    find_size_outliers,
    find_30x30_grids
)

__all__ = [
    'ARCGrid',
    'ARCExample', 
    'ARCTask',
    'ARCDataset',
    'analyze_grid_sizes',
    'detailed_grid_size_analysis',
    'find_tasks_by_color_count',
    'find_shape_preserving_tasks',
    'analyze_size_categories',
    'find_size_outliers',
    'find_30x30_grids'
]

__version__ = "1.0.0"