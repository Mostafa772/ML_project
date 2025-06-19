# ML_project

## Team Members

- Mostafa Eid eid.m2@studenti.unipi.it
- Dieudonne Iyabivuze d.iyabivuze@studenti.unipi.it
- Matteo Piredda m.piredda2@gmail.com

## Overview

This project implements a modular neural network framework from scratch in Python, designed for both regression and classification tasks. The framework is applied to two main problems:

- **CUP Dataset**: A regression task involving the prediction of 3D coordinates from 12 input features.
- **MONK Datasets**: Three classic binary classification tasks (MONK-1, MONK-2, MONK-3).

## Project Structure

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
│   ├── cascade_correlation.py
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
└── final_results.csv          # CUP results for submission

## Installation

1. **Clone the repository** and navigate to the project folder.

   ```bash
   https://github.com/Mostafa772/ML_project.git
   ```
2. **Install dependencies** (preferably in a virtual environment):

   ```bash
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
  - `random_search.py`: Hyperparameter optimization
  - `k_fold_cross_validation.py`: K-fold support
  - `optimizers.py`: Various optimizers implementations (Adam, Adagrad, RMS_Prop, SGD)
  - `loss_functions.py`: Various loss functions implementations
  - `activation_functions`: Various activation functions implementations (Linear, Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SoftMax)

## Features

- **Custom neural network architecture**

  - Fully connected networks with any number of hidden layers
  - Configurable layer sizes, weight initializations, and activation functions
- **Activation functions**

  - ReLU, Leaky ReLU, Tanh, Sigmoid, Softmax, and easily extendable to custom activations
- **Regularization techniques**

  - Dropout (per layer configurable rates)
  - L2 weight decay
  - Batch normalization (layer-wise normalization for stable training)
- **Optimizers**

  - Stochastic Gradient Descent (SGD) with momentum, Adam, Adagrad, RMS_Prop
- **Training strategies**

  - Early stopping based on validation performance
  - Learning rate scheduling (decay, adaptive)
  - Mini-batch gradient descent
  - K-fold cross-validation support
- **Hyperparameter optimization**

  - Random search over architecture and training hyperparameters
  - Top-N configuration logging
- **Ensemble learning**

  - Majority voting ensembles
  - Cascade correlation
- **Evaluation and visualization**

  - Automatic training/validation loss and accuracy plotting
  - Export of results to CSV and PNG
  - Modular codebase for easy experimentation and extension

## Requirements

See `requirements.txt` for all dependencies. Main libraries:

- numpy
- pandas
- matplotlib
- jupyter

## References

[Machine Learning A. Micheli lessons and slides](https://elearning.di.unipi.it/course/view.php?id=994)

[Neural Networks from scratch](https://github.com/GeorgeQLe/Textbooks-and-Papers/blob/master/%5BML%5D%20Harrison%20Kinsley%2C%20Daniel%20Kukie%C5%82a%20-%20Neural%20Networks%20from%20Scratch%20in%20Python%20(2020).pdf)
