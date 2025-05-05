# PINN for Poisson Equation

This project implements a Physics-Informed Neural Network (PINN) to solve the Poisson (or Laplace) equation.

## Project Structure

```
pinn_project/
│
├─ configs/             # Hydra configuration files (.yaml)
│   └─ poisson.yaml
│
├─ main/                # Main script for running experiments
│   └─ run.py
│
├─ model/               # Core PINN implementation (network, operators, loss)
│   ├─ __init__.py
│   ├─ nets.py
│   ├─ operators.py
│   └─ loss.py
│
├─ visualizations/      # Plotting and visualization code
│   ├─ plots.py
│   └─ wandb_callbacks.py # Optional custom wandb logging
│
├─ tests/               # Unit tests
│   └─ test_loss.py
│
├─ .gitignore
├─ requirements.txt     # Python dependencies
└─ README.md            # This file
```

## Setup

1.  **Create environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script using Hydra:

```bash
python main/run.py [optionally override config parameters]
```

Examples:

```bash
# Run with default config (configs/poisson.yaml)
python main/run.py

# Override learning rate and epochs
python main/run.py model.lr=1e-4 training.epochs=20000

# Run multiple experiments (if multirun is configured)
# python main/run.py --multirun model.width=32,64,128
```

Configuration is managed by [Hydra](https://hydra.cc/). Edit `configs/poisson.yaml` to change default parameters.

## Development

-   The `model/` directory contains the core, side-effect-free PINN logic.
-   The `main/run.py` script orchestrates the process (config loading, data sampling, training calls, visualization calls).
-   The `visualizations/` directory handles all plotting.
-   Tests are located in `tests/`. 