# ARC Neural Program Generation

A neural program generation project for solving the Abstraction and Reasoning Corpus (ARC) challenge. This project explores machine learning approaches to solve abstract reasoning tasks that require understanding patterns and generating program-like solutions.

## 🎯 Project Overview

The ARC (Abstraction and Reasoning Corpus) challenge tests AI systems' ability to acquire new skills and solve novel problems through pattern recognition and abstract reasoning. This project implements neural approaches to generate programs or rules that can solve ARC tasks.

### Key Features
- 🧠 Neural program synthesis for abstract reasoning
- 📊 Complete ARC dataset loading and analysis
- 🎨 Grid visualization and pattern analysis tools
- 🔧 Modular architecture for different neural approaches

## 📁 Project Structure

```
ARC-neural-program-generation/
├── data/                                    # ARC dataset files
│   ├── arc-agi_training_challenges.json    # Training tasks (3.8MB)
│   ├── arc-agi_training_solutions.json     # Training solutions (0.6MB)
│   ├── arc-agi_evaluation_challenges.json  # Evaluation tasks (0.9MB)
│   ├── arc-agi_evaluation_solutions.json   # Evaluation solutions (0.2MB)
│   ├── arc-agi_test_challenges.json        # Test tasks (1.0MB)
│   ├── sample_submission.json              # Sample submission format
│   └── arc-prize-2025.zip                  # Complete dataset archive
├── models/                                  # Neural model implementations (empty - ready for your models!)
├── sample.ipynb                            # Main analysis and experimentation notebook
├── requirements.txt                        # Python dependencies
└── README.md                              # This file
```

## 🚀 Quick Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.13+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MHHukiewitz/ARC-neural-program-generation.git
   cd ARC-neural-program-generation
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n arc-neural-program-generation python=3.13 -y
   conda activate arc-neural-program-generation
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter notebook**
   ```bash
   jupyter notebook sample.ipynb
   ```

## 📦 Dependencies

- **torch** (2.8.0) - Deep learning framework
- **torchvision** (0.23.0) - Computer vision utilities
- **numpy** (2.3.2) - Numerical computing
- **matplotlib** (3.10.5) - Plotting and visualization

## 🔍 Dataset Overview

The ARC dataset contains abstract reasoning tasks where each task includes:

- **Input grids**: Small colored grids (typically 3x3 to 30x30)
- **Output grids**: Expected transformations of the input
- **Training examples**: 1-4 input/output pairs per task
- **Test input**: A single grid to transform (solution not provided)

### Dataset Stats
- **Training set**: ~400 tasks with solutions
- **Evaluation set**: ~400 tasks with solutions  
- **Test set**: ~100 tasks (solutions withheld for competition)
- **Total**: ~900 unique reasoning challenges

## 🛠 Key Components

### ARCGrid Class
The core data structure for handling ARC grids with features:
- Grid validation and shape checking
- Color/pattern analysis utilities
- Visualization capabilities
- NumPy integration for numerical operations

### Data Loading Pipeline
- JSON dataset parsing
- Task structure validation
- Efficient grid representation
- Batch processing capabilities

## 🎨 Visualization Features

The project includes rich visualization tools:
- **Grid rendering**: Color-coded cell visualization
- **Pattern analysis**: Highlighting transformations
- **Statistical plots**: Color distributions, grid sizes
- **Interactive exploration**: Jupyter-based analysis

## 🧪 Getting Started

1. **Explore the dataset**:
   Open `sample.ipynb` and run the initial cells to load and examine ARC tasks

2. **Visualize tasks**:
   Use the built-in visualization tools to understand task patterns

3. **Implement your approach**:
   Add your neural models in the `models/` directory

4. **Experiment**:
   Use the notebook environment to test and iterate on your solutions

## 🏆 ARC Challenge Context

The ARC challenge tests three key capabilities:
1. **Core Knowledge**: Basic understanding of objects, counting, spatial relationships
2. **Broad Generalization**: Ability to apply learning to novel situations  
3. **Few-shot Learning**: Learning from very few examples (1-4 demonstrations)

This makes it an ideal testbed for neural program synthesis, meta-learning, and abstract reasoning approaches.

## 🔬 Research Directions

Potential approaches to explore:
- **Neural Module Networks**: Compositional reasoning
- **Program Synthesis**: Learning to generate transformation rules
- **Meta-Learning**: Few-shot adaptation to new tasks
- **Graph Neural Networks**: Spatial relationship modeling
- **Transformer Architectures**: Sequence-to-sequence transformations

## 📈 Hardware Requirements

- **CPU**: Any modern processor (training may be slow)
- **GPU**: CUDA or Apple Silicon (MPS) recommended for faster training
- **Memory**: 8GB+ RAM recommended for larger models
- **Storage**: ~10GB for datasets and model checkpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add your neural model implementations to `models/`
4. Document your approach in the notebook
5. Submit a pull request

## 📚 References

- [ARC Challenge](https://arcprize.org/) - Official challenge page
- [Original ARC Paper](https://arxiv.org/abs/1911.01547) - Chollet, François. "On the Measure of Intelligence"
- [ARC Dataset](https://github.com/fchollet/ARC) - Official dataset repository

## 📄 License

[MIT License](https://opensource.org/licenses/MIT)

---

Ready to tackle abstract reasoning? Start exploring the notebook and build your neural program generator! 🚀
