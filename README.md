# LogRegCCD
Implementation of logistic regression with cyclic coordinate descent based on [this article](https://www.jstatsoft.org/article/view/v033i01).

## Requirements

- Python 3.11
- Install project dependencies from `requirements.txt`:
  ```bash
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  ```
  or VSCode can execute this command automagically by using "Python: Create Enviroment..." action.
  To use jupyter notebooks (recommended way), `jupyter` package must be installed.

## Running comparison for new data

In the `notebooks/experiments/real_data/new_data.ipynb`, change the `DATASET_NAME` variable name to the desired dataset name from OpenML. Run all cells in the notebook.


## Directory structure

```plaintext
├── notebooks/                        # Jupyter notebooks
│   ├── initial_eda/                  # Initial EDA of prechosen 6 datasets
│   ├── eda/                          # EDA of finally chosen 4 datasets
│   ├── experiments/                  # Experiments on LogRegCCD implementation
│   │   ├── real_data/                # Experiments on real datasets from OpenML
│   │   │   └── new_data.ipynb        # Template for experiments on new real dataset
│   │   ├── synthetic_data/           # Experiments on generated synthetic dataset
│   │   └── algorithm_correctness/    # Test of algorithm correctness with sklearn's implementation as truth
├── src/                              # Main source files
│   ├── data/                         # Data loader and data interface related code
│   ├── eda/                          # EDA related functions
│   ├── log_reg_ccd.py                # LogRegCCD implementation
│   └── measures.py                   # Measure classes
├── tests/                            # Test files
├── requirements.txt                  # Python dependencies
└── README.md                         # Project description and setup guide
```
