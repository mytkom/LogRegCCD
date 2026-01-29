# LogRegCCD

Implementation of logistic regression with L1 penalty using Cyclical Coordinate Descent, based on [Friedman et al. (2010)](https://www.jstatsoft.org/article/view/v033i01).

**Authors:** Marek Mytkowski, Julia Dudzińska, Weronika Gozdera  
**Course:** Advanced Machine Learning, supervised by Dawid Płudowski

## Overview

This project implements from scratch a coordinate descent algorithm for L1-regularized logistic regression, providing efficient feature selection and regularization capabilities. The implementation was validated against sklearn's `LogisticRegression` on multiple real-world and synthetic datasets.

## Key Results

The algorithm was evaluated on four high-dimensional datasets from OpenML and compared against sklearn's `LogisticRegression` (no penalty, max_iter=1000). Experiments used 70-10-20 train-validation-test splits, with regularization parameter selected via ROC AUC validation.

### Performance Comparison

| Dataset | Test Accuracy | ROC AUC | PR AUC | F1 Score | Balanced Acc. | Features Reduced |
|---------|---------------|---------|--------|----------|---------------|-----------------|
| **Arcene** (n=200, d=10K) | 77.5% vs 67.5% | 0.812 vs 0.841 | 0.875 vs 0.881 | **0.824 vs 0.649** | **0.750 vs 0.719** | 3,259/3,414 (95%) |
| **Arrhythmia** (n=452, d=279) | **67.0% vs 62.6%** | **0.681 vs 0.666** | **0.594 vs 0.570** | **0.693 vs 0.630** | **0.673 vs 0.627** | 34/244 (14%) |
| **Bioresponse** (n=3.7K, d=1.8K) | **75.8% vs 69.8%** | **0.793 vs 0.714** | **0.810 vs 0.736** | **0.793 vs 0.735** | **0.749 vs 0.693** | 818/1,875 (44%) |
| **Madelon** (n=2.6K, d=500) | **62.9% vs 52.0%** | **0.657 vs 0.540** | **0.645 vs 0.531** | **0.632 vs 0.505** | **0.629 vs 0.522** | ~495/500 (99%) |

**Key findings:**
- **Feature selection**: Achieved 95% feature reduction on Arcene (3,259/3,414) while maintaining competitive performance
- **Overfitting mitigation**: On Madelon, sklearn achieved 100% training accuracy but only 52% test accuracy; LogRegCCD achieved 62.9% test accuracy with effective regularization
- **Class imbalance**: Superior performance on imbalanced datasets (Arcene, Arrhythmia) with better F1 Score and Balanced Accuracy
- **Feature identification**: On Madelon, correctly identified ~5 informative features (close to the expected 5 true features)
- **Algorithm validation**: Verified correctness against sklearn's implementation on regularization paths

## Requirements

- Python 3.11
- Install project dependencies from `requirements.txt`:
  ```bash
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  ```
  VSCode can execute this automatically using "Python: Create Environment..." action.  
  To use Jupyter notebooks (recommended), ensure `jupyter` package is installed.

## Usage

### Running comparison for new data

In `notebooks/experiments/real_data/new_data.ipynb`, change the `DATASET_NAME` variable to the desired dataset name from OpenML. Run all cells in the notebook.

## Project Structure

```plaintext
├── notebooks/                        # Jupyter notebooks
│   ├── initial_eda/                  # Initial EDA of prechosen 6 datasets
│   ├── eda/                          # EDA of finally chosen 4 datasets
│   ├── experiments/                  # Experiments on LogRegCCD implementation
│   │   ├── real_data/                # Experiments on real datasets from OpenML
│   │   │   ├── new_data.ipynb        # Template for experiments on new real dataset
│   │   │   ├── arcene.ipynb          # Arcene dataset experiments
│   │   │   ├── arrhythmia.ipynb      # Arrhythmia dataset experiments
│   │   │   ├── bioresponse.ipynb     # Bioresponse dataset experiments
│   │   │   ├── madelon.ipynb         # Madelon dataset experiments
│   │   │   └── results/               # Experiment results and visualizations
│   │   ├── synthetic_data/           # Experiments on generated synthetic dataset
│   │   └── algorithm_correctness/    # Test of algorithm correctness with sklearn's implementation as truth
├── src/                              # Main source files
│   ├── data/                         # Data loader and data interface related code
│   │   ├── data_loader.py            # Dataset loading utilities
│   │   └── dataset_interface.py      # Dataset interface implementation
│   ├── eda/                          # EDA related functions
│   │   └── eda.py                    # Exploratory data analysis utilities
│   ├── log_reg_ccd.py                # LogRegCCD implementation
│   ├── measures.py                   # Measure classes (ROC AUC, PR AUC, F1, Balanced Accuracy)
│   └── utils.py                      # Utility functions for evaluation and visualization
├── tests/                            # Test files
│   └── data/                         # Data-related tests
│       ├── test_data_loader.py       # Tests for data loader
│       └── test_dataset_interface.py # Tests for dataset interface
├── requirements.txt                  # Python dependencies
└── README.md                         # Project description and setup guide
```
