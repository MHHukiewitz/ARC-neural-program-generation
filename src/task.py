"""
Task data structures for ARC (Abstraction and Reasoning Corpus) dataset.

This module contains the ARCTask class that represents a complete ARC task
with training examples and test input/output.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .grid import ARCGrid, ARCExample


@dataclass
class ARCTask:
    """Represents a complete ARC task with training examples and test input/output"""
    task_id: str
    train_examples: List[ARCExample]
    test_input: ARCGrid
    test_output: Optional[ARCGrid] = None  # Only available for training/evaluation sets
    
    def __post_init__(self):
        """Convert raw data to proper objects"""
        # Convert train examples
        if self.train_examples and len(self.train_examples) > 0 and isinstance(self.train_examples[0], dict):
            examples = []
            for example in self.train_examples:
                if isinstance(example, dict):
                    examples.append(ARCExample(
                        input_grid=ARCGrid(example['input']),
                        output_grid=ARCGrid(example['output'])
                    ))
                else:
                    examples.append(example)
            self.train_examples = examples
        
        # Convert test input
        if isinstance(self.test_input, list):
            self.test_input = ARCGrid(self.test_input)
            
        # Convert test output if available
        if self.test_output is not None and isinstance(self.test_output, list):
            self.test_output = ARCGrid(self.test_output)
    
    @property
    def num_train_examples(self) -> int:
        """Number of training examples"""
        return len(self.train_examples)
    
    @property
    def train_input_shapes(self) -> List[Tuple[int, int]]:
        """Shapes of all training input grids"""
        return [ex.input_shape for ex in self.train_examples]
    
    @property
    def train_output_shapes(self) -> List[Tuple[int, int]]:
        """Shapes of all training output grids"""
        return [ex.output_shape for ex in self.train_examples]
    
    @property
    def test_input_shape(self) -> Tuple[int, int]:
        """Shape of test input grid"""
        return self.test_input.shape
    
    @property
    def test_output_shape(self) -> Optional[Tuple[int, int]]:
        """Shape of test output grid (if available)"""
        return self.test_output.shape if self.test_output else None
    
    @property
    def all_colors_used(self) -> set:
        """All colors used across all grids in this task"""
        colors = set()
        for example in self.train_examples:
            colors.update(example.input_grid.unique_colors)
            colors.update(example.output_grid.unique_colors)
        colors.update(self.test_input.unique_colors)
        if self.test_output:
            colors.update(self.test_output.unique_colors)
        return colors
    
    @property
    def shape_changes(self) -> List[Tuple[int, int]]:
        """Shape changes for each training example"""
        return [ex.shape_change for ex in self.train_examples]
    
    @property
    def consistent_shape_change(self) -> bool:
        """Whether all training examples have the same shape change"""
        changes = self.shape_changes
        return len(set(changes)) <= 1
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics about this task"""
        return {
            'task_id': self.task_id,
            'num_train_examples': self.num_train_examples,
            'train_input_shapes': self.train_input_shapes,
            'train_output_shapes': self.train_output_shapes,
            'test_input_shape': self.test_input_shape,
            'test_output_shape': self.test_output_shape,
            'all_colors_used': sorted(list(self.all_colors_used)),
            'shape_changes': self.shape_changes,
            'consistent_shape_change': self.consistent_shape_change,
            'max_grid_size': max([
                max(shape) for shape in 
                self.train_input_shapes + self.train_output_shapes + [self.test_input_shape]
            ])
        }
    
    def visualize(self, include_test: bool = True):
        """Visualize all examples in the task"""
        print(f"Task {self.task_id}")
        print(f"Training examples: {self.num_train_examples}")
        
        for i, example in enumerate(self.train_examples):
            print(f"\nTraining Example {i+1}:")
            example.visualize(f"Example {i+1}")
        
        if include_test:
            print(f"\nTest Input:")
            self.test_input.visualize("Test Input")
            
            if self.test_output:
                print(f"\nTest Output:")
                self.test_output.visualize("Test Output")