# ML_project

## Overview

This project implements a modular neural network framework from scratch in Python, designed for both regression and classification tasks. The framework is applied to two main problems:

- **CUP Dataset**: A regression task involving the prediction of 3D coordinates from 12 input features.
- **MONK Datasets**: Three classic binary classification tasks (MONK-1, MONK-2, MONK-3).

The project supports advanced features such as:
- Customizable neural network architectures
- Batch normalization, dropout, and various activation functions
- Early stopping and learning rate scheduling
- Random search for hyperparameter optimization
- Ensemble learning (majority voting, cascade correlation)
- K-fold cross-validation
- Modular code for easy experimentation

## Team Members

- Dieudonne
- Mostafa
- Matteo

## Project Structure

```
ML_project/
│
├── CUP.ipynb                # Main notebook for CUP regression task
├── MONK.ipynb               # Main notebook for MONK classification tasks
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── data/
│   ├── cup/
│   │   ├── ML-CUP24-TR.csv  # CUP training data
│   │   └── ML-CUP24-TS.csv  # CUP test data
│   ├── Monk_1/
│   │   ├── monks-1.train    # MONK-1 training data
│   │   └── monks-1.test     # MONK-1 test data
│   ├── Monk_2/
│   │   ├── monks-2.train
│   │   └── monks-2.test
│   └── Monk_3/
│       ├── monks-3.train
│       └── monks-3.test
│
├── src/                     # Source code (modular neural network implementation)
│   ├── activation_functions.py
│   ├── batch_normalization.py
│   ├── data_preprocessing.py
│   ├── dropout.py
│   ├── early_stopping.py
│   ├── ensemble/
│   │   └── cascade_correlation.py
│   ├── ensemble_learning.py
│   ├── k_fold_cross_validation.py
│   ├── layer.py
│   ├── loss_functions.py
│   ├── neural_network.py
│   ├── optimizers.py
│   ├── random_search.py
│   ├── scheduler.py
│   ├── train_and_evaluate.py
│   └── utils.py
│
├── top_5_results.csv        # Best hyperparameters/results for CUP
├── monk1_top5res.csv        # Best results for MONK-1
├── monk2_top5res.csv        # Best results for MONK-2
├── monk3_top5res.csv        # Best results for MONK-3
└── cup_top5res.csv          # Best results for CUP
```

## Installation

1. **Clone the repository** and navigate to the project folder.
2. **Install dependencies** (preferably in a virtual environment):

   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### 1. CUP Task

- Open `CUP.ipynb` in Jupyter Notebook or VSCode.
- The notebook will:
  - Load and preprocess the CUP data (`data/cup/ML-CUP24-TR.csv`)
  - Perform random search for hyperparameter optimization
  - Train the best model and plot results
  - Save the best results to `cup_top5res.csv`
- You can adjust hyperparameters and model architecture in the notebook.

### 2. MONK Tasks

- Open `MONK.ipynb`.
- The notebook will:
  - Load and preprocess the selected MONK dataset (`data/Monk_X/`)
  - Perform random search for hyperparameter optimization
  - Train the best model and plot results
  - Save the best results to `monkX_top5res.csv`
- You can select which MONK dataset to use by changing the `MONK_NUM` variable.

### 3. Source Code

- All neural network logic is in the `src/` directory.
- Key modules:
  - `neural_network.py`: Main NN class, supports custom architectures
  - `train_and_evaluate.py`: Training loop, early stopping, evaluation
  - `ensemble_learning.py`: Ensemble methods (majority voting)
  - `random_search.py`: Hyperparameter optimization
  - `k_fold_cross_validation.py`: K-fold support
  - `ensemble/cascade_correlation.py`: Cascade correlation implementation

## Features

- **Custom Neural Networks**: Build deep networks with any number of layers, activations, batch norm, dropout, etc.
- **Ensemble Learning**: Train multiple models and combine predictions via majority voting.
- **Cascade Correlation**: Dynamically add neurons during training for improved performance.
- **Early Stopping & Scheduling**: Prevent overfitting and tune learning rates automatically.
- **Random Search**: Efficient hyperparameter search.
- **K-Fold Cross-Validation**: Reliable model evaluation.
- **Visualization**: Training/validation accuracy and loss plots are saved in `results/`.

## Results

- Best hyperparameters and validation scores are saved in the `*_top5res.csv` files.
- Training and validation curves are saved as PNGs in the `results/` folder.

## How to Extend

- Add new activation functions in `src/activation_functions.py`.
- Implement new optimizers in `src/optimizers.py`.
- Add new datasets in the `data/` folder and update the notebooks accordingly.

## Requirements

See `requirements.txt` for all dependencies. Main libraries:
- numpy
- pandas
- matplotlib
- jupyter

## Acknowledgements


---

For any questions, please contact the project members.
