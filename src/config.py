"""
Configuration Module for Energy Consumption Forecasting Project
================================================================
Central configuration file for all paths, parameters, and settings.

Author: Lekshmi
Date: 2025
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


# =============================================================================
# PROJECT ROOT AND BASE DIRECTORIES
# =============================================================================

# Assuming config.py is inside src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


# =============================================================================
# DATA PATHS
# =============================================================================

@dataclass
class DataPaths:
    """All data-related paths"""

    # Raw data
    raw_dir: Path = RAW_DATA_DIR
    raw_data: Path = raw_dir / "KAG_energydata_complete.csv"

    # Processed data
    processed_dir: Path = PROCESSED_DATA_DIR
    cleaned_data: Path = processed_dir / "cleaned_data.csv"
    engineered_features: Path = processed_dir / "engineered_features.csv"


# =============================================================================
# MODEL PATHS
# =============================================================================

@dataclass
class ModelPaths:
    """All model-related paths"""

    models_dir: Path = MODELS_DIR

    # Model artifacts
    xgboost_model: Path = models_dir / "xgboost_model.json"
    xgboost_pickle: Path = models_dir / "xgboost_model.pkl"

    # Metadata
    training_metadata: Path = models_dir / "training_metadata.json"
    split_info: Path = models_dir / "split_info.json"
    feature_importance: Path = models_dir / "feature_importance.csv"

    # Predictions
    predictions_dir: Path = models_dir / "predictions"
    train_predictions: Path = predictions_dir / "train_predictions.csv"
    val_predictions: Path = predictions_dir / "val_predictions.csv"
    test_predictions: Path = predictions_dir / "test_predictions.csv"
    quick_check_plot: Path = predictions_dir / "quick_check_plot.png"

    # Evaluation outputs
    evaluation_dir: Path = models_dir / "evaluation"
    evaluation_report: Path = evaluation_dir / "evaluation_report.json"
    timeseries_plot: Path = evaluation_dir / "timeseries_comparison.png"
    scatter_plot: Path = evaluation_dir / "scatter_plot.png"
    error_dist_plot: Path = evaluation_dir / "error_distribution.png"
    residual_plot: Path = evaluation_dir / "residual_plot.png"


# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature engineering parameters"""

    horizon_hours: int = 2

    time_features: List[str] = None
    lag_features: List[str] = None
    rolling_features: List[str] = None
    momentum_features: List[str] = None
    regime_features: List[str] = None
    context_features: List[str] = None

    def __post_init__(self):
        self.time_features = [
            "hour_sin", "hour_cos", "target_hour_sin",
            "dow_sin", "month_sin",
            "day_of_week", "target_dow", "month"
        ]

        self.lag_features = [
            "lag_1h", "lag_2h", "lag_3h",
            "lag_6h", "lag_12h", "lag_24h", "lag_168h"
        ]

        self.rolling_features = [
            "roll_3h_mean", "roll_6h_mean", "roll_12h_mean",
            "roll_24h_mean", "roll_168h_mean",
            "roll_6h_max", "roll_6h_min",
            "roll_12h_max", "roll_12h_min",
            "roll_24h_min", "roll_168h_min",
            "roll_168h_median", "roll_168h_q25", "roll_168h_q75",
            "roll_24h_median", "range_24h"
        ]

        self.momentum_features = [
            "momentum_1h", "momentum_6h", "momentum_24h",
            "dist_from_24h_min", "dist_from_6h_min",
            "rel_to_24h_mean", "rel_to_6h_mean",
            "zscore_168h", "zscore_6h",
            "ema_3h", "ema_6h", "trend_6h"
        ]

        self.regime_features = [
            "usage_regime", "spike_intensity_24h", "avg_this_hour"
        ]

        self.context_features = [
            "is_night", "is_morning", "is_evening",
            "is_weekend", "target_is_peak",
            "is_local_peak", "is_local_trough",
            "evening_x_level", "weekend_x_level"
        ]

    def get_all_features(self) -> List[str]:
        return (
            self.time_features +
            self.lag_features +
            self.rolling_features +
            self.momentum_features +
            self.regime_features +
            self.context_features
        )


# =============================================================================
# MODEL TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Model training parameters"""

    random_seed: int = 42

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    num_boost_round: int = 3000
    early_stopping_rounds: int = 150
    verbose_eval: int = 100

    use_smoothing: bool = True

    xgb_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "max_depth": 6,
                "min_child_weight": 3,
                "gamma": 0.2,
                "eta": 0.01,
                "reg_alpha": 0.3,
                "reg_lambda": 2.5,
                "subsample": 0.75,
                "colsample_bytree": 0.7,
                "objective": "reg:quantileerror",
                "quantile_alpha": 0.5,
                "tree_method": "hist",
                "seed": self.random_seed,
            }


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Model evaluation parameters"""

    n_plot_samples: int = 1000
    max_scatter_points: int = 5000

    tolerance_levels: List[int] = None
    error_ranges: List[int] = None

    def __post_init__(self):
        self.tolerance_levels = [5, 10, 15, 20, 30, 50, 100, 150, 200]
        self.error_ranges = [5, 10, 20, 30, 50, 100, 200]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create required directories if missing"""
    dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR / "predictions",
        MODELS_DIR / "evaluation",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("✓ Project directories verified")


# =============================================================================
# GLOBAL CONFIG INSTANCES
# =============================================================================

data_paths = DataPaths()
model_paths = ModelPaths()
feature_config = FeatureConfig()
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Testing configuration...")
    create_directories()
    print(f"Total features: {len(feature_config.get_all_features())}")
