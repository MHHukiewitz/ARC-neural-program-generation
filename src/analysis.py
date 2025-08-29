"""
Analysis utilities for ARC (Abstraction and Reasoning Corpus) dataset.

This module contains helper functions for analyzing grid sizes, colors,
transformations, and other properties of ARC tasks.
"""

from typing import List, Dict, Tuple, Any
from .dataset import ARCDataset


def analyze_grid_sizes(arc_dataset: ARCDataset, dataset_name: str = "training") -> List[int]:
    """Analyze distribution of grid sizes in the dataset"""
    tasks = getattr(arc_dataset, f"{dataset_name}_tasks")
    
    sizes = []
    for task in tasks.values():
        for example in task.train_examples:
            sizes.append(example.input_grid.size)
            sizes.append(example.output_grid.size)
        sizes.append(task.test_input.size)
        if task.test_output:
            sizes.append(task.test_output.size)
    
    print(f"\nGrid size analysis for {dataset_name} set:")
    print(f"- Total grids: {len(sizes)}")
    print(f"- Min size: {min(sizes)}")
    print(f"- Max size: {max(sizes)}")
    print(f"- Average size: {sum(sizes) / len(sizes):.1f}")
    
    return sizes


def find_tasks_by_color_count(arc_dataset: ARCDataset, num_colors: int, dataset_name: str = "training") -> List[str]:
    """Find tasks that use exactly the specified number of colors"""
    tasks = getattr(arc_dataset, f"{dataset_name}_tasks")
    
    matching_tasks = []
    for task_id, task in tasks.items():
        if len(task.all_colors_used) == num_colors:
            matching_tasks.append(task_id)
    
    print(f"\nTasks using exactly {num_colors} colors: {len(matching_tasks)}")
    return matching_tasks


def find_shape_preserving_tasks(arc_dataset: ARCDataset, dataset_name: str = "training") -> List[str]:
    """Find tasks where input and output have the same shape"""
    tasks = getattr(arc_dataset, f"{dataset_name}_tasks")
    
    shape_preserving = []
    for task_id, task in tasks.items():
        if task.consistent_shape_change and task.shape_changes[0] == (0, 0):
            shape_preserving.append(task_id)
    
    print(f"\nShape-preserving tasks: {len(shape_preserving)}")
    return shape_preserving


def detailed_grid_size_analysis(arc_dataset: ARCDataset, dataset_name: str = "training") -> Tuple[List[Dict], List[Tuple]]:
    """Comprehensive analysis of grid sizes and dimensions"""
    tasks = getattr(arc_dataset, f"{dataset_name}_tasks")
    
    # Collect all grid data
    grid_data = []
    shape_data = []
    
    for task_id, task in tasks.items():
        # Training examples
        for i, example in enumerate(task.train_examples):
            # Input grids
            h, w = example.input_grid.shape
            grid_data.append({
                'task_id': task_id,
                'type': 'train_input',
                'example_idx': i,
                'height': h,
                'width': w,
                'size': h * w,
                'aspect_ratio': w / h,
                'is_square': h == w
            })
            shape_data.append((h, w))
            
            # Output grids
            h, w = example.output_grid.shape
            grid_data.append({
                'task_id': task_id,
                'type': 'train_output',
                'example_idx': i,
                'height': h,
                'width': w,
                'size': h * w,
                'aspect_ratio': w / h,
                'is_square': h == w
            })
            shape_data.append((h, w))
        
        # Test input
        h, w = task.test_input.shape
        grid_data.append({
            'task_id': task_id,
            'type': 'test_input',
            'example_idx': 0,
            'height': h,
            'width': w,
            'size': h * w,
            'aspect_ratio': w / h,
            'is_square': h == w
        })
        shape_data.append((h, w))
        
        # Test output (if available)
        if task.test_output:
            h, w = task.test_output.shape
            grid_data.append({
                'task_id': task_id,
                'type': 'test_output',
                'example_idx': 0,
                'height': h,
                'width': w,
                'size': h * w,
                'aspect_ratio': w / h,
                'is_square': h == w
            })
            shape_data.append((h, w))
    
    # Calculate statistics
    heights = [g['height'] for g in grid_data]
    widths = [g['width'] for g in grid_data]
    sizes = [g['size'] for g in grid_data]
    aspect_ratios = [g['aspect_ratio'] for g in grid_data]
    
    print(f"\n{'='*60}")
    print(f"DETAILED GRID SIZE ANALYSIS - {dataset_name.upper()} SET")
    print(f"{'='*60}")
    
    print(f"\nBasic Statistics:")
    print(f"- Total grids analyzed: {len(grid_data)}")
    print(f"- Number of tasks: {len(tasks)}")
    
    print(f"\nHeight Statistics:")
    print(f"- Min height: {min(heights)}")
    print(f"- Max height: {max(heights)}")
    print(f"- Average height: {sum(heights) / len(heights):.2f}")
    print(f"- Unique heights: {sorted(set(heights))}")
    
    print(f"\nWidth Statistics:")
    print(f"- Min width: {min(widths)}")
    print(f"- Max width: {max(widths)}")
    print(f"- Average width: {sum(widths) / len(widths):.2f}")
    print(f"- Unique widths: {sorted(set(widths))}")
    
    print(f"\nGrid Size (total cells) Statistics:")
    print(f"- Min size: {min(sizes)}")
    print(f"- Max size: {max(sizes)}")
    print(f"- Average size: {sum(sizes) / len(sizes):.2f}")
    
    print(f"\nShape Distribution:")
    unique_shapes = list(set(shape_data))
    unique_shapes.sort(key=lambda x: (x[0], x[1]))
    print(f"- Total unique shapes: {len(unique_shapes)}")
    print(f"- Most common shapes:")
    
    # Count shape frequencies
    shape_counts = {}
    for shape in shape_data:
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    # Sort by frequency and show top 10
    sorted_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (shape, count) in enumerate(sorted_shapes[:10]):
        print(f"  {i+1}. {shape[0]}x{shape[1]}: {count} grids ({count/len(grid_data)*100:.1f}%)")
    
    print(f"\nAspect Ratio Analysis:")
    square_grids = sum(1 for g in grid_data if g['is_square'])
    print(f"- Square grids: {square_grids} ({square_grids/len(grid_data)*100:.1f}%)")
    print(f"- Rectangular grids: {len(grid_data) - square_grids} ({(len(grid_data) - square_grids)/len(grid_data)*100:.1f}%)")
    print(f"- Min aspect ratio: {min(aspect_ratios):.2f}")
    print(f"- Max aspect ratio: {max(aspect_ratios):.2f}")
    print(f"- Average aspect ratio: {sum(aspect_ratios) / len(aspect_ratios):.2f}")
    
    return grid_data, shape_data


def analyze_size_categories(grid_data: List[Dict]) -> None:
    """Categorize grids by size ranges"""
    print(f"\n{'='*40}")
    print(f"GRID SIZE CATEGORIES")
    print(f"{'='*40}")
    
    # Define size categories
    categories = [
        ("Tiny", 1, 9),          # 1x1 to 3x3
        ("Small", 10, 49),       # Up to 7x7
        ("Medium", 50, 225),     # Up to 15x15
        ("Large", 226, 400),     # Up to 20x20
        ("Extra Large", 401, 900) # Up to 30x30
    ]
    
    for cat_name, min_size, max_size in categories:
        count = sum(1 for g in grid_data if min_size <= g['size'] <= max_size)
        percentage = count / len(grid_data) * 100
        print(f"{cat_name:12} ({min_size:3}-{max_size:3} cells): {count:4} grids ({percentage:5.1f}%)")
    
    # Show extreme cases
    print(f"\nExtreme Cases:")
    sizes = [g['size'] for g in grid_data]
    min_size = min(sizes)
    max_size = max(sizes)
    
    smallest_grids = [g for g in grid_data if g['size'] == min_size]
    largest_grids = [g for g in grid_data if g['size'] == max_size]
    
    print(f"- Smallest grids ({min_size} cells):")
    for g in smallest_grids[:3]:  # Show first 3
        print(f"  {g['height']}x{g['width']} in task {g['task_id']} ({g['type']})")
    
    print(f"- Largest grids ({max_size} cells):")
    for g in largest_grids[:3]:  # Show first 3
        print(f"  {g['height']}x{g['width']} in task {g['task_id']} ({g['type']})")


def find_size_outliers(grid_data: List[Dict], threshold_percentile: int = 95) -> None:
    """Find unusually large or small grids"""
    sizes = [g['size'] for g in grid_data]
    sizes_sorted = sorted(sizes)
    
    # Calculate percentiles
    p5_idx = int(len(sizes_sorted) * 0.05)
    p95_idx = int(len(sizes_sorted) * 0.95)
    
    p5_size = sizes_sorted[p5_idx] if p5_idx < len(sizes_sorted) else sizes_sorted[0]
    p95_size = sizes_sorted[p95_idx] if p95_idx < len(sizes_sorted) else sizes_sorted[-1]
    
    print(f"\n{'='*40}")
    print(f"SIZE OUTLIERS")
    print(f"{'='*40}")
    
    print(f"5th percentile size: {p5_size} cells")
    print(f"95th percentile size: {p95_size} cells")
    
    # Find outliers
    small_outliers = [g for g in grid_data if g['size'] <= p5_size]
    large_outliers = [g for g in grid_data if g['size'] >= p95_size]
    
    print(f"\nSmall outliers (≤{p5_size} cells): {len(small_outliers)} grids")
    for g in small_outliers[:5]:
        print(f"  {g['height']}x{g['width']} ({g['size']} cells) - {g['task_id']} ({g['type']})")
    
    print(f"\nLarge outliers (≥{p95_size} cells): {len(large_outliers)} grids")
    for g in large_outliers[:5]:
        print(f"  {g['height']}x{g['width']} ({g['size']} cells) - {g['task_id']} ({g['type']})")


def find_grids_by_shape(arc_dataset: ARCDataset, target_shape: Tuple[int, int]) -> List[Dict]:
    """Find all grids with a specific shape (height, width) in the dataset"""
    matching_grids = []
    
    # Check training tasks
    for task_id, task in arc_dataset.training_tasks.items():
        for i, example in enumerate(task.train_examples):
            if example.input_grid.shape == target_shape:
                matching_grids.append({
                    'task_id': task_id,
                    'type': 'train_input',
                    'example_idx': i,
                    'grid': example.input_grid,
                    'dataset': 'training'
                })
            if example.output_grid.shape == target_shape:
                matching_grids.append({
                    'task_id': task_id,
                    'type': 'train_output', 
                    'example_idx': i,
                    'grid': example.output_grid,
                    'dataset': 'training'
                })
        
        if task.test_input.shape == target_shape:
            matching_grids.append({
                'task_id': task_id,
                'type': 'test_input',
                'example_idx': 0,
                'grid': task.test_input,
                'dataset': 'training'
            })
        
        if task.test_output and task.test_output.shape == target_shape:
            matching_grids.append({
                'task_id': task_id,
                'type': 'test_output',
                'example_idx': 0,
                'grid': task.test_output,
                'dataset': 'training'
            })

    # Check evaluation tasks
    for task_id, task in arc_dataset.evaluation_tasks.items():
        for i, example in enumerate(task.train_examples):
            if example.input_grid.shape == target_shape:
                matching_grids.append({
                    'task_id': task_id,
                    'type': 'train_input',
                    'example_idx': i,
                    'grid': example.input_grid,
                    'dataset': 'evaluation'
                })
            if example.output_grid.shape == target_shape:
                matching_grids.append({
                    'task_id': task_id,
                    'type': 'train_output',
                    'example_idx': i,
                    'grid': example.output_grid,
                    'dataset': 'evaluation'
                })
        
        if task.test_input.shape == target_shape:
            matching_grids.append({
                'task_id': task_id,
                'type': 'test_input',
                'example_idx': 0,
                'grid': task.test_input,
                'dataset': 'evaluation'
            })
        
        if task.test_output and task.test_output.shape == target_shape:
            matching_grids.append({
                'task_id': task_id,
                'type': 'test_output',
                'example_idx': 0,
                'grid': task.test_output,
                'dataset': 'evaluation'
            })
    
    return matching_grids