"""
Grid data structures for ARC (Abstraction and Reasoning Corpus) dataset.

This module contains the core data classes for representing individual grids
and input-output examples in ARC tasks.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ARCGrid:
    """Represents a single grid in an ARC task"""
    data: List[List[int]]
    
    def __post_init__(self):
        """Validate grid data and convert to numpy array"""
        if not self.data or not self.data[0]:
            raise ValueError("Grid cannot be empty")
        
        # Ensure all rows have the same length
        row_length = len(self.data[0])
        if not all(len(row) == row_length for row in self.data):
            raise ValueError("All rows must have the same length")
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get grid dimensions (height, width)"""
        return len(self.data), len(self.data[0])
    
    @property
    def height(self) -> int:
        """Get grid height"""
        return len(self.data)
    
    @property
    def width(self) -> int:
        """Get grid width"""
        return len(self.data[0])
    
    @property
    def size(self) -> int:
        """Get total number of cells"""
        return self.height * self.width
    
    @property
    def unique_colors(self) -> set:
        """Get set of unique colors/values in the grid"""
        return set(cell for row in self.data for cell in row)
    
    @property
    def color_counts(self) -> Dict[int, int]:
        """Get count of each color/value"""
        counts = {}
        for row in self.data:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
        return counts
    
    def get_cell(self, row: int, col: int) -> int:
        """Get value at specific position"""
        return self.data[row][col]
    
    def to_numpy(self):
        """Convert to numpy array (requires numpy)"""
        return np.array(self.data)
    
    def visualize(self, title: str = "", figsize: Tuple[int, int] = (6, 6)):
        """Visualize the grid using matplotlib"""
        # Define color map for ARC colors (0-9)
        colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
                 '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create color array
        color_array = np.array([[colors[cell] for cell in row] for row in self.data])
        
        # Display grid
        ax.imshow([[cell for cell in row] for row in self.data], 
                 cmap='tab10', vmin=0, vmax=9)
        
        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='white', linewidth=1)
        
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig, ax


@dataclass 
class ARCExample:
    """Represents a single input-output example in an ARC task"""
    input_grid: ARCGrid
    output_grid: ARCGrid
    
    def __post_init__(self):
        """Convert list data to ARCGrid objects if needed"""
        if isinstance(self.input_grid, list):
            self.input_grid = ARCGrid(self.input_grid)
        if isinstance(self.output_grid, list):
            self.output_grid = ARCGrid(self.output_grid)
    
    @property
    def input_shape(self) -> Tuple[int, int]:
        """Get input grid shape"""
        return self.input_grid.shape
    
    @property  
    def output_shape(self) -> Tuple[int, int]:
        """Get output grid shape"""
        return self.output_grid.shape
    
    @property
    def shape_change(self) -> Tuple[int, int]:
        """Get change in dimensions from input to output"""
        return (self.output_shape[0] - self.input_shape[0], 
                self.output_shape[1] - self.input_shape[1])
    
    def visualize(self, title: str = ""):
        """Visualize input and output side by side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Use the grid visualization method
        self.input_grid.visualize(f"{title} - Input")
        self.output_grid.visualize(f"{title} - Output")
        
        return fig, (ax1, ax2)