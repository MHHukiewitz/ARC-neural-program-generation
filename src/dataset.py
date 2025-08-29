"""
Dataset management for ARC (Abstraction and Reasoning Corpus).

This module contains the ARCDataset class for loading and managing
the complete ARC dataset including training, evaluation, and test sets.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from .task import ARCTask


class ARCDataset:
    """Main class for loading and managing ARC dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.training_tasks: Dict[str, ARCTask] = {}
        self.evaluation_tasks: Dict[str, ARCTask] = {}
        self.test_tasks: Dict[str, ARCTask] = {}
        
    def load_training_data(self) -> None:
        """Load training challenges and solutions"""
        challenges_file = self.data_dir / "arc-agi_training_challenges.json"
        solutions_file = self.data_dir / "arc-agi_training_solutions.json"
        
        print(f"Loading training data from {challenges_file}")
        
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)
            
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
        
        self.training_tasks = {}
        for task_id, task_data in challenges.items():
            # Get test output from solutions
            test_output = solutions.get(task_id, [None])[0] if task_id in solutions else None
            
            task = ARCTask(
                task_id=task_id,
                train_examples=task_data['train'],
                test_input=task_data['test'][0]['input'],
                test_output=test_output
            )
            self.training_tasks[task_id] = task
        
        print(f"Loaded {len(self.training_tasks)} training tasks")
    
    def load_evaluation_data(self) -> None:
        """Load evaluation challenges and solutions"""
        challenges_file = self.data_dir / "arc-agi_evaluation_challenges.json"
        solutions_file = self.data_dir / "arc-agi_evaluation_solutions.json"
        
        print(f"Loading evaluation data from {challenges_file}")
        
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)
            
        with open(solutions_file, 'r') as f:
            solutions = json.load(f)
        
        self.evaluation_tasks = {}
        for task_id, task_data in challenges.items():
            # Get test output from solutions
            test_output = solutions.get(task_id, [None])[0] if task_id in solutions else None
            
            task = ARCTask(
                task_id=task_id,
                train_examples=task_data['train'],
                test_input=task_data['test'][0]['input'],
                test_output=test_output
            )
            self.evaluation_tasks[task_id] = task
        
        print(f"Loaded {len(self.evaluation_tasks)} evaluation tasks")
    
    def load_test_data(self) -> None:
        """Load test challenges (no solutions available)"""
        challenges_file = self.data_dir / "arc-agi_test_challenges.json"
        
        print(f"Loading test data from {challenges_file}")
        
        with open(challenges_file, 'r') as f:
            challenges = json.load(f)
        
        self.test_tasks = {}
        for task_id, task_data in challenges.items():
            task = ARCTask(
                task_id=task_id,
                train_examples=task_data['train'],
                test_input=task_data['test'][0]['input'],
                test_output=None  # No solutions for test set
            )
            self.test_tasks[task_id] = task
        
        print(f"Loaded {len(self.test_tasks)} test tasks")
    
    def load_all_data(self) -> None:
        """Load all available data"""
        self.load_training_data()
        self.load_evaluation_data()
        self.load_test_data()
    
    def get_task(self, task_id: str, dataset: str = "auto") -> ARCTask:
        """Get a specific task by ID. Raises KeyError if task not found."""
        if dataset == "auto":
            # Search in all datasets
            if task_id in self.training_tasks:
                return self.training_tasks[task_id]
            elif task_id in self.evaluation_tasks:
                return self.evaluation_tasks[task_id]
            elif task_id in self.test_tasks:
                return self.test_tasks[task_id]
            else:
                raise KeyError(f"Task ID '{task_id}' not found in any dataset")
        elif dataset == "train":
            if task_id not in self.training_tasks:
                raise KeyError(f"Task ID '{task_id}' not found in training dataset")
            return self.training_tasks[task_id]
        elif dataset == "eval":
            if task_id not in self.evaluation_tasks:
                raise KeyError(f"Task ID '{task_id}' not found in evaluation dataset") 
            return self.evaluation_tasks[task_id]
        elif dataset == "test":
            if task_id not in self.test_tasks:
                raise KeyError(f"Task ID '{task_id}' not found in test dataset")
            return self.test_tasks[task_id]
        else:
            raise ValueError("dataset must be 'auto', 'train', 'eval', or 'test'")
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the entire dataset"""
        def analyze_tasks(tasks: Dict[str, ARCTask]) -> Dict:
            if not tasks:
                return {}
            
            all_shapes = []
            all_colors = set()
            shape_changes = []
            
            for task in tasks.values():
                task_stats = task.get_stats()
                all_shapes.extend(task_stats['train_input_shapes'])
                all_shapes.extend(task_stats['train_output_shapes'])
                all_shapes.append(task_stats['test_input_shape'])
                if task_stats['test_output_shape']:
                    all_shapes.append(task_stats['test_output_shape'])
                all_colors.update(task_stats['all_colors_used'])
                shape_changes.extend(task_stats['shape_changes'])
            
            return {
                'num_tasks': len(tasks),
                'total_examples': sum(task.num_train_examples for task in tasks.values()),
                'unique_shapes': len(set(all_shapes)),
                'max_grid_size': max(max(shape) for shape in all_shapes) if all_shapes else 0,
                'colors_used': sorted(list(all_colors)),
                'unique_shape_changes': len(set(shape_changes))
            }
        
        return {
            'training': analyze_tasks(self.training_tasks),
            'evaluation': analyze_tasks(self.evaluation_tasks),
            'test': analyze_tasks(self.test_tasks)
        }