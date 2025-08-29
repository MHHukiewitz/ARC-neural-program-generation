# ARC Neural Program Generation

This repository contains tools and data structures for working with the ARC (Abstraction and Reasoning Corpus) dataset for neural program generation research.

## Project Structure

```
ARC-neural-program-generation/
├── src/                          # Modular Python package
│   ├── __init__.py              # Package initialization and exports
│   ├── grid.py                  # ARCGrid and ARCExample classes
│   ├── task.py                  # ARCTask class
│   ├── dataset.py               # ARCDataset class for data loading
│   └── analysis.py              # Analysis utilities and functions
├── data/                        # ARC dataset files
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   └── arc-agi_test_challenges.json
├── sample.ipynb                 # Original comprehensive analysis notebook
├── example_usage.ipynb          # Clean example using modular structure
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Quick Start

### Installation

   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```python
# Import the main classes
from src import ARCDataset, ARCGrid, ARCTask

# Load the dataset
dataset = ARCDataset()
dataset.load_training_data()
dataset.load_evaluation_data()

# Get a specific task
task = dataset.get_task('00576224')

# Access grid properties
print(f"Test input shape: {task.test_input.shape}")
print(f"Colors used: {task.test_input.unique_colors}")
print(f"Total cells: {task.test_input.size}")

# Visualize a grid
task.test_input.visualize("Test Input")
```

### Analysis Functions

```python
from src import (
    analyze_grid_sizes,
    find_tasks_by_color_count,
    find_shape_preserving_tasks,
    detailed_grid_size_analysis,
    find_30x30_grids
)

# Analyze grid size distribution
sizes = analyze_grid_sizes(dataset, "training")

# Find specific types of tasks
binary_tasks = find_tasks_by_color_count(dataset, 2)  # Tasks with only 2 colors
shape_preserving = find_shape_preserving_tasks(dataset)  # Input/output same size
large_grids = find_30x30_grids(dataset)  # Maximum size grids

# Comprehensive analysis
grid_data, shape_data = detailed_grid_size_analysis(dataset, "training")
```

## Core Classes

### ARCGrid
Represents a single grid with analysis methods:
- **Properties**: `shape`, `height`, `width`, `size`, `unique_colors`, `color_counts`
- **Methods**: `visualize()`, `to_numpy()`, `get_cell()`

### ARCExample  
Represents an input-output pair:
- **Properties**: `input_shape`, `output_shape`, `shape_change`
- **Methods**: `visualize()` (side-by-side)

### ARCTask
Represents a complete task with training examples and test case:
- **Properties**: `num_train_examples`, `all_colors_used`, `consistent_shape_change`
- **Methods**: `get_stats()`, `visualize()`

### ARCDataset
Main class for loading and managing the entire dataset:
- **Methods**: `load_training_data()`, `load_evaluation_data()`, `get_task()`, `get_dataset_stats()`

## Analysis Functions

### Grid Size Analysis
- `analyze_grid_sizes()`: Basic size distribution statistics
- `detailed_grid_size_analysis()`: Comprehensive analysis with shape frequencies
- `analyze_size_categories()`: Categorize grids by size ranges
- `find_size_outliers()`: Find unusually large/small grids

### Task Classification
- `find_tasks_by_color_count()`: Tasks using specific number of colors
- `find_shape_preserving_tasks()`: Tasks where input/output shapes match
- `find_30x30_grids()`: Find maximum-size grids

## Dataset Statistics

Based on the training and evaluation sets:

### Grid Sizes
- **Range**: 1 to 900 cells (1×1 to 30×30)
- **Average**: ~186 cells
- **Distribution**:
  - Tiny (1-9 cells): 10.1%
  - Small (10-49 cells): 19.5% 
  - Medium (50-225 cells): 47.0%
  - Large (226-400 cells): 14.2%
  - Extra Large (401-900 cells): 9.2%

### Key Insights for Neural Program Generation
1. **Memory Planning**: Support up to 900 cells per grid
2. **Input Representation**: Maximum dimensions 30×30, most grids much smaller
3. **Architecture**: 61.7% square grids, need to handle rectangular effectively
4. **Processing**: Most computation on small-medium grids (66.6% ≤225 cells)
5. **Model Capacity**: Handle 514 unique grid shapes across datasets

## Example Notebooks

- **`sample.ipynb`**: Original comprehensive analysis with all functionality inline
- **`example_usage.ipynb`**: Clean example using the modular structure

## Neural Program Generation Applications

This codebase provides the foundation for:
- **Data Loading**: Efficient loading and preprocessing of ARC tasks
- **Grid Analysis**: Understanding size distributions and complexity patterns  
- **Task Classification**: Identifying different types of reasoning patterns
- **Visualization**: Interactive exploration of tasks and transformations
- **Benchmarking**: Systematic evaluation of neural program synthesis models

The modular structure makes it easy to extend with new analysis functions, integrate into training pipelines, and adapt for specific research needs.

## Contributing

The codebase is organized for easy extension:
1. Add new grid analysis methods to `src/grid.py`
2. Add new task-level analysis to `src/task.py` 
3. Add new dataset-wide analysis functions to `src/analysis.py`
4. Update `src/__init__.py` to export new functions

## License

This project is part of ARC neural program generation research.