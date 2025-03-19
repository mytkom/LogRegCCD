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


## Directory structure

```plaintext
├── notebooks/                       # Jupyter notebooks for data preparation and training
├── src/                             # Main source files
├── tests/                           # Tests files
├── requirements.txt                 # Python dependencies
└── README.md                        # Project description and setup guide
```
