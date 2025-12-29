# Appliances Energy Prediction

A machine learning system for predicting household appliance energy consumption 2 hours ahead using time-series forecasting and XGBoost quantile regression.

![App Screenshot](docs/app_screenshot.png)

## 📋 Description

This project develops an intelligent forecasting system that predicts household appliance energy consumption using historical usage patterns and temporal features. Built on 4.5 months of granular energy data collected at 10-minute intervals from a residential building in Belgium, the system achieves:

- **26.86 Wh MAE** on test data (27.5% of mean consumption)
- **90% accuracy** within ±50 Wh tolerance
- **58% accuracy** within ±10 Wh for typical usage patterns
- Real-time predictions suitable for smart grid integration and energy management

The model leverages 55 engineered features including exponential moving averages, rolling statistics, lag features, and usage regime classification to capture complex temporal patterns while remaining computationally efficient for production deployment.

## 🛠️ Tech Stack

- **Python 3.8+**
- **XGBoost** - Gradient boosting with quantile regression
- **Pandas & NumPy** - Data manipulation and numerical operations
- **Scikit-learn** - Model evaluation metrics
- **Gradio** - Interactive web interface
- **Matplotlib/Seaborn** - Visualization (notebooks)

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download data (if not included)**

```bash
# Place your dataset in data/raw/
# Expected format: KAG_energydata_complete.csv with 'date' and 'Appliances' columns
```

## ⚙️ Configuration

Update `config.py` with your settings:

```python
# config.py
DATA_PATH = "data/raw/KAG_energydata_complete.csv"
MODEL_PATH = "models/xgboost_model.json"
HORIZON_HOURS = 2
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.85
RANDOM_SEED = 42
```

## 🚀 Usage

### Training the Model

```bash
python src/model_training.py
```

### Running the Gradio App

```bash
python app.py
```

Then navigate to `http://localhost:7860` in your browser.

### Using Notebooks

```bash
jupyter notebook
# Open notebooks in notebooks/ directory
```

## 📁 Folder Structure

```
energy-forecasting/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py                          # Configuration settings
├── app.py                             # Gradio web application
├── data/
│   ├── raw/
│   │   └── KAG_energydata_complete.csv
│   └── processed/
│       └── engineered_features.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and statistical analysis
│   ├── 02_spike_analysis_eda.ipynb    # Spike behavior analysis
│   ├── 03_feature_engineering.ipynb   # Feature creation process
│   ├── 04_model_training.ipynb        # Model development
│   └── 05_evaluation.ipynb            # Performance evaluation
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py         # Feature creation functions
│   ├── model_training.py              # XGBoost training pipeline
│   └── evaluation.py                  # Metrics and analysis
├── models/
│   └── xgboost_model.json             # Trained model (generated)
└── docs/
    ├── app_screenshot.png
    ├── interpretability_insights.pdf
    ├── scalability_production.pdf
    └── architecture.md
```

## 🎯 Key Features

- **55 Engineered Features**: Temporal patterns, rolling statistics, lag features, usage regimes
- **No Data Leakage**: Proper temporal split with train-time computed historical averages
- **Quantile Regression**: Robust to outliers and skewed distributions
- **Fast Inference**: <5ms per prediction, suitable for real-time applications
- **Interpretable**: Feature importance analysis reveals exponential moving averages and recent lags dominate predictions

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 26.86 Wh |
| RMSE | 72.23 Wh |
| R² Score | 0.3684 |
| MAPE | 18.54% |
| Median Error | 8.12 Wh |

### Performance by Usage Level:

- **Low usage (0-100 Wh)**: 9.01 Wh MAE - covers 73% of cases
- **Medium usage (100-200 Wh)**: 25.02 Wh MAE
- **High usage (200-300 Wh)**: 76.24 Wh MAE
- **Very high usage (>300 Wh)**: 275.03 Wh MAE



## 👤 Author

**Lekshmi J**
