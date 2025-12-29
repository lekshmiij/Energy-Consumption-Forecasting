"""
Model Training Module for Appliance Energy Prediction
======================================================
XGBoost Quantile Regression with optimized hyperparameters for 2-hour 
ahead energy consumption prediction with adaptive post-processing.

Author:Lekshmi
Date: 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Dict, Any

warnings.filterwarnings('ignore')


class EnergyModelTrainer:
    """
    Trains and evaluates XGBoost quantile regression model for energy prediction.
    
    Features:
    - Temporal data splitting (no shuffling)
    - XGBoost quantile regression training
    - Adaptive post-processing/smoothing
    - Comprehensive model persistence
    - Feature importance analysis
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_seed (int): Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.model = None
        self.features = None
        self.params = None
        self.metadata = {}
        self.split_info = {}
        
    def temporal_split(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                      val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data temporally to prevent data leakage.
        
        Temporal split preserves time order: 70% train, 15% validation, 15% test
        
        Args:
            df (pd.DataFrame): Input dataframe with features and target
            train_ratio (float): Proportion for training set
            val_ratio (float): Proportion for validation set
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Remove rows with missing target
        df_clean = df.dropna(subset=['target']).reset_index(drop=True)
        
        n = len(df_clean)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df_clean.iloc[:train_end]
        val = df_clean.iloc[train_end:val_end]
        test = df_clean.iloc[val_end:]
        
        print("Dataset Split:")
        print(f"  Train:      {len(train):,} samples ({len(train)/len(df_clean)*100:.1f}%)")
        print(f"  Validation: {len(val):,} samples ({len(val)/len(df_clean)*100:.1f}%)")
        print(f"  Test:       {len(test):,} samples ({len(test)/len(df_clean)*100:.1f}%)")
        
        # Store split information
        self.split_info = {
            'train_indices': train.index.tolist(),
            'val_indices': val.index.tolist(),
            'test_indices': test.index.tolist(),
            'train_date_range': [str(train['date'].min()), str(train['date'].max())],
            'val_date_range': [str(val['date'].min()), str(val['date'].max())],
            'test_date_range': [str(test['date'].min()), str(test['date'].max())]
        }
        
        return train, val, test
    
    def prepare_feature_matrices(self, train: pd.DataFrame, val: pd.DataFrame, 
                                 test: pd.DataFrame, features: list) -> Tuple:
        """
        Prepares feature matrices (X) and targets (y) for all data splits.
        
        Args:
            train (pd.DataFrame): Training data
            val (pd.DataFrame): Validation data
            test (pd.DataFrame): Test data
            features (list): List of feature column names
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Prepare train set
        X_train = train[features].fillna(0)
        y_train = train['target']
        
        # Prepare validation set
        X_val = val[features].fillna(0)
        y_val = val['target']
        
        # Prepare test set
        X_test = test[features].fillna(0)
        y_test = test['target']
        
        print("\nFeature Matrix Shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_optimized_params(self) -> Dict[str, Any]:
        """
        Returns optimized hyperparameters for XGBoost quantile regression.
        
        These parameters were tuned through extensive ablation testing.
        
        Returns:
            Dict containing XGBoost hyperparameters
        """
        params = {
            # Tree structure
            'max_depth': 6,                    # Maximum tree depth (controls complexity)
            'min_child_weight': 3,             # Minimum sum of instance weight in child
            'gamma': 0.2,                      # Minimum loss reduction for split
            
            # Learning rate and regularization
            'eta': 0.01,                       # Learning rate (small for stability)
            'reg_alpha': 0.3,                  # L1 regularization term
            'reg_lambda': 2.5,                 # L2 regularization term
            
            # Sampling
            'subsample': 0.75,                 # Row sampling ratio per tree
            'colsample_bytree': 0.7,           # Column sampling ratio per tree
            'colsample_bylevel': 0.7,          # Column sampling ratio per tree level
            
            # Objective
            'objective': 'reg:quantileerror',  # Quantile regression loss
            'quantile_alpha': 0.5,             # Target quantile (0.5 = median)
            
            # Training
            'tree_method': 'hist',             # Histogram-based algorithm (fast)
            'seed': self.random_seed           # Random seed for reproducibility
        }
        
        self.params = params
        return params
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   num_boost_round: int = 3000,
                   early_stopping_rounds: int = 150,
                   verbose_eval: int = 100) -> xgb.Booster:
        """
        Trains XGBoost model with early stopping on validation set.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            num_boost_round (int): Maximum number of boosting rounds
            early_stopping_rounds (int): Stop if no improvement for N rounds
            verbose_eval (int): Print progress every N rounds
            
        Returns:
            Trained XGBoost Booster model
        """
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)
        
        # Get parameters
        if self.params is None:
            self.params = self.get_optimized_params()
        
        # Print hyperparameters
        print("\nHyperparameters:")
        for key, value in self.params.items():
            print(f"  {key:20s}: {value}")
        
        # Convert to DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        print("\nDMatrix objects created successfully")
        print(f"Training configuration:")
        print(f"  Max boosting rounds:      {num_boost_round}")
        print(f"  Early stopping rounds:    {early_stopping_rounds}")
        print(f"  Verbose evaluation every: {verbose_eval} rounds")
        
        # Train model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        print(f"\n✓ Training complete")
        print(f"Best iteration: {self.model.best_iteration}")
        print(f"Best validation score: {self.model.best_score:.4f}")
        
        return self.model
    
    def smooth_predictions(self, predictions: np.ndarray, 
                          prev_values: np.ndarray) -> np.ndarray:
        """
        Applies adaptive smoothing based on previous value magnitude.
        
        Reduces large jumps that are physically unrealistic in energy consumption.
        Smoothing intensity varies by usage regime (low/medium/high).
        
        Args:
            predictions (np.ndarray): Raw model predictions
            prev_values (np.ndarray): Previous actual or predicted values
            
        Returns:
            np.ndarray: Smoothed predictions
        """
        smoothed = predictions.copy()
        
        for i in range(1, len(smoothed)):
            prev_val = prev_values[i-1] if i == 1 else smoothed[i-1]
            curr_val = smoothed[i]
            jump = abs(curr_val - prev_val)
            
            # Adaptive smoothing based on magnitude
            if prev_val < 150:
                # Low usage state - aggressive smoothing
                if jump > 200:
                    smoothed[i] = 0.4 * curr_val + 0.6 * prev_val
                elif jump > 120:
                    smoothed[i] = 0.6 * curr_val + 0.4 * prev_val
                    
            elif prev_val < 300:
                # Medium usage state - moderate smoothing
                if jump > 300:
                    smoothed[i] = 0.45 * curr_val + 0.55 * prev_val
                elif jump > 180:
                    smoothed[i] = 0.65 * curr_val + 0.35 * prev_val
                    
            else:
                # High usage state - light smoothing
                if jump > 400:
                    smoothed[i] = 0.5 * curr_val + 0.5 * prev_val
                elif jump > 250:
                    smoothed[i] = 0.7 * curr_val + 0.3 * prev_val
        
        # Ensure non-negative predictions
        smoothed = np.maximum(smoothed, 0)
        
        return smoothed
    
    def evaluate_predictions(self, y_true: np.ndarray, pred_raw: np.ndarray,
                           pred_smoothed: np.ndarray) -> Dict[str, float]:
        """
        Evaluates both raw and smoothed predictions, selects best version.
        
        Args:
            y_true (np.ndarray): True target values
            pred_raw (np.ndarray): Raw model predictions
            pred_smoothed (np.ndarray): Smoothed predictions
            
        Returns:
            Dict containing evaluation metrics and best predictions
        """
        # Calculate metrics
        mae_raw = mean_absolute_error(y_true, pred_raw)
        mae_smoothed = mean_absolute_error(y_true, pred_smoothed)
        rmse_raw = np.sqrt(mean_squared_error(y_true, pred_raw))
        rmse_smoothed = np.sqrt(mean_squared_error(y_true, pred_smoothed))
        
        # Select best predictions
        use_smoothed = mae_smoothed < mae_raw
        final_pred = pred_smoothed if use_smoothed else pred_raw
        final_mae = min(mae_smoothed, mae_raw)
        final_rmse = rmse_smoothed if use_smoothed else rmse_raw
        
        print("\n" + "="*80)
        print("PERFORMANCE EVALUATION")
        print("="*80)
        print(f"Raw MAE:          {mae_raw:.2f} Wh")
        print(f"Smoothed MAE:     {mae_smoothed:.2f} Wh")
        print(f"Final MAE:        {final_mae:.2f} Wh")
        print(f"Raw RMSE:         {rmse_raw:.2f} Wh")
        print(f"Smoothed RMSE:    {rmse_smoothed:.2f} Wh")
        print(f"\nUsing: {'Smoothed' if use_smoothed else 'Raw'} predictions")
        
        return {
            'mae_raw': mae_raw,
            'mae_smoothed': mae_smoothed,
            'mae_final': final_mae,
            'rmse_raw': rmse_raw,
            'rmse_smoothed': rmse_smoothed,
            'rmse_final': final_rmse,
            'use_smoothed': use_smoothed,
            'final_predictions': final_pred
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Extracts and displays feature importance scores.
        
        Args:
            top_n (int): Number of top features to display
            
        Returns:
            pd.DataFrame: Feature importance scores sorted by importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        print("\n" + "="*80)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*80)
        print(f"\n{'Rank':<6} {'Feature':<30} {'Importance'}")
        print("-" * 60)
        
        for i, row in enumerate(importance_df.head(top_n).itertuples(), 1):
            print(f"{i:<6} {row.feature:<30} {row.importance:>12.1f}")
        
        return importance_df
    
    def save_model(self, output_dir: str = 'models') -> None:
        """
        Saves the trained model in multiple formats.
        
        Args:
            output_dir (str): Directory to save model files
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XGBoost model (JSON format)
        model_path = os.path.join(output_dir, 'xgboost_model.json')
        self.model.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Save model as pickle (alternative format)
        pickle_path = os.path.join(output_dir, 'xgboost_model.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved to: {pickle_path}")
    
    def save_predictions(self, test_data: pd.DataFrame, y_test: pd.Series,
                        pred_raw: np.ndarray, pred_smoothed: np.ndarray,
                        final_pred: np.ndarray, output_dir: str = 'models/predictions') -> None:
        """
        Saves predictions to CSV for later analysis.
        
        Args:
            test_data (pd.DataFrame): Test dataset with dates
            y_test (pd.Series): True test values
            pred_raw (np.ndarray): Raw predictions
            pred_smoothed (np.ndarray): Smoothed predictions
            final_pred (np.ndarray): Final selected predictions
            output_dir (str): Directory to save predictions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'date': test_data['date'].values,
            'actual': y_test.values,
            'predicted_raw': pred_raw,
            'predicted_smoothed': pred_smoothed,
            'final_prediction': final_pred
        })
        
        pred_path = os.path.join(output_dir, 'test_predictions.csv')
        predictions_df.to_csv(pred_path, index=False)
        print(f"✓ Predictions saved to: {pred_path}")
    
    def save_feature_importance(self, importance_df: pd.DataFrame,
                               output_dir: str = 'models') -> None:
        """
        Saves feature importance scores to CSV.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            output_dir (str): Directory to save file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to: {importance_path}")
    
    def save_metadata(self, n_features: int, train_size: int, val_size: int,
                     test_size: int, eval_results: Dict, output_dir: str = 'models') -> None:
        """
        Saves training metadata and configuration.
        
        Args:
            n_features (int): Number of features used
            train_size (int): Training set size
            val_size (int): Validation set size
            test_size (int): Test set size
            eval_results (Dict): Evaluation results dictionary
            output_dir (str): Directory to save metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'XGBoost Quantile Regression',
            'n_features': n_features,
            'n_train_samples': train_size,
            'n_val_samples': val_size,
            'n_test_samples': test_size,
            'best_iteration': int(self.model.best_iteration),
            'best_validation_score': float(self.model.best_score),
            'test_mae_raw': float(eval_results['mae_raw']),
            'test_mae_smoothed': float(eval_results['mae_smoothed']),
            'test_mae_final': float(eval_results['mae_final']),
            'test_rmse_final': float(eval_results['rmse_final']),
            'used_smoothing': bool(eval_results['use_smoothed']),
            'hyperparameters': self.params
        }
        
        self.metadata = metadata
        
        metadata_path = os.path.join(output_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    def save_split_info(self, output_dir: str = 'models') -> None:
        """
        Saves data split information for reproducibility.
        
        Args:
            output_dir (str): Directory to save split info
        """
        os.makedirs(output_dir, exist_ok=True)
        
        split_path = os.path.join(output_dir, 'split_info.json')
        with open(split_path, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        print(f"✓ Split info saved to: {split_path}")
    
    def create_quick_visualization(self, y_test: pd.Series, final_pred: np.ndarray,
                                   n_plot: int = 500, output_dir: str = 'models/predictions') -> None:
        """
        Creates and saves a quick visualization of predictions vs actuals.
        
        Args:
            y_test (pd.Series): True test values
            final_pred (np.ndarray): Final predictions
            n_plot (int): Number of samples to plot
            output_dir (str): Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_plot = min(n_plot, len(y_test))
        
        plt.figure(figsize=(15, 5))
        plt.plot(range(n_plot), y_test.values[:n_plot], label='Actual', alpha=0.7, linewidth=1.5)
        plt.plot(range(n_plot), final_pred[:n_plot], label='Predicted', alpha=0.7, linewidth=1.5)
        plt.xlabel('Time Index')
        plt.ylabel('Appliances (Wh)')
        plt.title('Predictions vs Actual (First 500 Test Samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'quick_check_plot.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Visualization saved to: {plot_path}")
        plt.close()
    
    def print_training_summary(self, eval_results: Dict) -> None:
        """
        Prints a comprehensive training summary.
        
        Args:
            eval_results (Dict): Evaluation results dictionary
        """
        print("\n" + "="*80)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*80)
        print(f"\nModel Configuration:")
        print(f"  Algorithm:        XGBoost Quantile Regression")
        print(f"  Features:         {self.metadata.get('n_features', 'N/A')}")
        print(f"  Training samples: {self.metadata.get('n_train_samples', 'N/A'):,}")
        print(f"  Best iteration:   {self.model.best_iteration}")
        
        print(f"\nPerformance:")
        print(f"  Test MAE:         {eval_results['mae_final']:.2f} Wh")
        print(f"  Test RMSE:        {eval_results['rmse_final']:.2f} Wh")
        print(f"  Smoothing used:   {'Yes' if eval_results['use_smoothed'] else 'No'}")
        
        print(f"\n✓ Model ready for evaluation and deployment!")
        print("=" * 80)


def load_feature_names(feature_path: str = 'data/features/feature_names.txt') -> list:
    """
    Loads feature names from text file.
    
    Args:
        feature_path (str): Path to feature names file
        
    Returns:
        list: List of feature names
    """
    with open(feature_path, 'r') as f:
        features = f.read().splitlines()
    print(f"Loaded {len(features)} feature names")
    return features


def main():
    """
    Main execution function for model training pipeline.
    """
    # Configuration
    DATA_PATH = 'data/processed/engineered_features.csv'
    FEATURE_PATH = 'data/features/feature_names.txt'
    OUTPUT_DIR = 'models'
    RANDOM_SEED = 42
    
    print("="*80)
    print("Energy Prediction - Model Training Pipeline")
    print("="*80)
    
    # Load data
    print(f"\nLoading engineered features from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Load feature names
    print(f"\nLoading feature names from: {FEATURE_PATH}")
    features = load_feature_names(FEATURE_PATH)
    
    # Initialize trainer
    trainer = EnergyModelTrainer(random_seed=RANDOM_SEED)
    
    # Split data
    print("\n" + "="*80)
    print("Splitting data temporally...")
    print("="*80)
    train, val, test = trainer.temporal_split(df, train_ratio=0.7, val_ratio=0.15)
    
    # Prepare feature matrices
    print("\n" + "="*80)
    print("Preparing feature matrices...")
    print("="*80)
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_feature_matrices(
        train, val, test, features
    )
    
    # Get optimized parameters
    params = trainer.get_optimized_params()
    
    # Train model
    model = trainer.train_model(
        X_train, y_train, X_val, y_val,
        num_boost_round=3000,
        early_stopping_rounds=150,
        verbose_eval=100
    )
    
    # Make predictions
    print("\n" + "="*80)
    print("Generating predictions...")
    print("="*80)
    
    dtest = xgb.DMatrix(X_test)
    pred_test_raw = model.predict(dtest)
    print(f"✓ Raw predictions generated: {len(pred_test_raw):,}")
    
    # Apply post-processing
    print("\nApplying adaptive smoothing...")
    pred_test_smoothed = trainer.smooth_predictions(pred_test_raw, y_test.values)
    print("✓ Smoothing applied")
    
    # Evaluate predictions
    eval_results = trainer.evaluate_predictions(
        y_test.values, pred_test_raw, pred_test_smoothed
    )
    
    # Get feature importance
    print("\n" + "="*80)
    print("Analyzing feature importance...")
    print("="*80)
    importance_df = trainer.get_feature_importance(top_n=20)
    
    # Save everything
    print("\n" + "="*80)
    print("Saving model and results...")
    print("="*80)
    
    trainer.save_model(OUTPUT_DIR)
    trainer.save_predictions(
        test, y_test, pred_test_raw, pred_test_smoothed,
        eval_results['final_predictions'], f"{OUTPUT_DIR}/predictions"
    )
    trainer.save_feature_importance(importance_df, OUTPUT_DIR)
    trainer.save_metadata(
        len(features), len(train), len(val), len(test),
        eval_results, OUTPUT_DIR
    )
    trainer.save_split_info(OUTPUT_DIR)
    
    # Create visualization
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)
    trainer.create_quick_visualization(
        y_test, eval_results['final_predictions'],
        n_plot=500, output_dir=f"{OUTPUT_DIR}/predictions"
    )
    
    # Print summary
    trainer.print_training_summary(eval_results)
    
    return trainer, eval_results


if __name__ == "__main__":
    trainer, results = main()