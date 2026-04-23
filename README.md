# 📈 Mathematical Modelling for Financial Risk

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange.svg)](https://jupyter.org/)

> **Predicting Value at Risk (VaR) using GARCH(1,1) volatility modelling**

## 📖 Overview

This repository contains a complete mathematical modelling project focused on financial risk management. The core objective is to implement and validate a **GARCH(1,1) model** for forecasting **Value at Risk (VaR)** — a statistical measure that quantifies the potential loss in value of a portfolio over a defined period for a given confidence interval.

The project bridges theoretical financial mathematics with practical implementation, featuring a production‑ready Python module for VaR prediction, comprehensive backtesting, and an academic paper documenting the methodology.

## ✨ Features

- **GARCH(1,1) Engine** — A dedicated module for estimating conditional volatility using the Generalized Autoregressive Conditional Heteroskedasticity model.
- **Value at Risk Prediction** — Rolling VaR forecasts at standard confidence levels (e.g., 95%, 99%).
- **Backtesting Framework** — Statistical tests (e.g., Kupiec’s Proportion of Failures test) to validate model accuracy.
- **Jupyter Notebook Workflow** — Step‑by‑step exploration and visualisation of the modelling process.
- **Academic Documentation** — A complete exposition paper (PDF + LaTeX source) with figures and references for reproducible research.
- **Modular Codebase** — Clean separation of data loading, model logic, and backtesting for easy extension.

## 📂 Project Structure
```text
MathModelling/
├── Study/ # Academic paper and supporting materials
│ ├── Draft of Exposition Paper.pdf # Final write‑up
│ ├── Draft of Exposition Paper.tex # LaTeX source
│ ├── reference.bib # Bibliography
│ └── figure*.png # Visualisations (6 figures)
├── Values_at_Risk_predictor/ # VaR prediction Python module
│ ├── GARCH_Engine.py # GARCH(1,1) model implementation
│ ├── data_loader.py # Data ingestion utilities
│ ├── backtest.py # VaR backtesting routines
│ └── init.py # Package initialiser
├── notebook/ # Jupyter notebooks
│ └── main.ipynb # Main analysis workflow
├── pyproject.toml # Project build & dependency configuration
├── .gitignore # Git ignore rules
└── LICENSE # MIT License


## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- (Optional) virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/williamjames10231/MathModelling.git
   cd MathModelling

2. **Seting up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

3. **Installing the Dependencies**
   ```bash
   pip install pip-tools
   pip-compile pyproject.toml
   pip install -r requirements.txt

4. **Run the analysis***
   ```bash
   jupyter notebook notebook/main.ipynb

## 📊 Key results  
The GARCH(1,1) model successfully captures volatility clustering and produces accurate VaR forecasts. Backtesting confirms that the predicted VaR violations align with expected frequencies, validating the model’s reliability for risk management. Further details
on Study/ section