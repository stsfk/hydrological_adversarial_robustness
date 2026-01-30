# Hydrological Adversarial Robustness

This project investigates the adversarial robustness of hydrological models, on both conceptual models (HBV) and data-driven deep learning models (LSTM). It employs the Fast Gradient Sign Method (FGSM) to generate adversarial examples by perturbing meteorological forcings (Precipitation, Temperature, Potential Evapotranspiration) and analyzing the impact on discharge predictions using the CAMELS-DE dataset.

## Project Overview

The core objective is to understand how small, imperceptible changes in input data can affect the output of hydrological models. The project includes:

- **Adversarial Perturbations**: Implementation of FGSM perturbations to test model robustness.
- **Analysis Tools**: Scripts to analyze model linearity, parameter sensitivity, and perturbation effectiveness.

## Key Components

### Models

- **`hbv.py`**: A differentiable PyTorch implementation of the HBV conceptual model (based on Hy2DL). It supports static and dynamic parameterization and includes a Unit Hydrograph routing module (adopted from [Hy2DL](https://github.com/KIT-HYD/Hy2DL)).
- **`lstm_train_final_model.py`**: Trains and evaluates the Catchment-Embedding LSTM model. It loads hyperparameters, trains on the training set, measures performance on the test set, and saves the trained model artifacts.

### Adversarial Perturbations (FGSM)

- **`hbv_fgsm.py`**: Performs FGSM perturbations on the HBV model. It calculates gradients of Earth observation inputs with respect to discharge error and perturbs inputs to maximize error (epsilon=0.2 by default), while ensuring physical constraints (non-negative precipitation and PET).
- **`lstm_fgsm.py`**: Performs FGSM perturbations on the pre-trained LSTM model. It loads the saved model states and computes the drop in performance (MSE, NSE, KGE) under adversarial perturbations.

### Analysis & Visualization

- **`analyze_*.py`**: Various scripts for understanding model behavior:
  - `analyze_Q_linearity_each_time_step.py`: Analyzes linearity of discharge response.
  - `analyze_hbv_internal_linearity.py` / `analyze_lstm_internal_linearity.py`: Internal state linearity checks.
  - `analyze_sweep_effect_all_catchments.py`: Analyzes the effect of perturbation magnitudes.
- **`visualize_*.py`**: Scripts for plotting results:
  - `visualize_effect_on_timeseries.py`: Shows original vs. perturbed hydrographs.
  - `visualize_lstm_epsilon_sweep.py`: Visualizes how model performance degrades with increasing perturbation magnitudes.
  - `visualize_overall_effectiveness.py`: Summarizes perturbation effectiveness across catchments.

### Data Processing

- **`CAMELS_DE_data_processing.R`**: R script for preprocessing the CAMELS-DE dataset for use in this project.

## Setup and Dependencies

This project relies on **Python 3** and **PyTorch**.
Key Python dependencies include:

- `torch`
- `numpy`
- `pandas`
- `HydroErr` (for hydrological metrics)
- `tqdm`
- `joblib`
- `matplotlib` / `seaborn` (for visualization)

Data is expected to be in the `data/` directory. The [CAMELS-DE V1.0](https://zenodo.org/records/13837553) dataset is used in this project. The R script `CAMELS_DE_data_processing.R` is used to preprocess the dataset for use in this project. The processed data is saved to `data/data_train_CAMELS_DE1.00.csv`, `data/data_test_CAMELS_DE1.00.csv`, and `data/data_validation_CAMELS_DE1.00.csv`.

Then, lstm_hyperparameter_search.py is used to find the best hyperparameters for the LSTM model. lstm_train_final_model.py is used to train the LSTM model with the best hyperparameters.
