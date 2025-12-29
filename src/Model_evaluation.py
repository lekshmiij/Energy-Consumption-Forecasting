"""
Model Evaluation Module for Appliance Energy Prediction
========================================================
Comprehensive performance analysis and visualization of trained models.

Features:
- Core metrics calculation (MAE, RMSE, R², MAPE)
- Error distribution analysis
- Performance by usage level and time of day
- Comprehensive visualizations
- Detailed evaluation reports

Author: Lekshmi 
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import json
import os
from typing import Dict, List, Tuple, Any
import warnings


warnings.filterwarnings('ignore')
from config import DataPaths,ModelPaths
data_paths = DataPaths()
paths = ModelPaths()


class EnergyModelEvaluator:
    """
    Comprehensive evaluation suite for energy prediction models.
    
    Provides detailed performance analysis including:
    - Multiple evaluation metrics
    - Error distribution analysis
    - Performance by usage levels and time periods
    - Visual analysis tools
    - Report generation
    """
    
    def __init__(self, predictions_path: str, metadata_path: str):
        """
        Initialize the evaluator.
        
        Args:
            predictions_path (str): Path to predictions CSV file
            metadata_path (str): Path to training metadata JSON file
        """
        self.predictions_path = predictions_path
        self.metadata_path = metadata_path
        self.predictions_df = None
        self.metadata = None
        self.y_true = None
        self.y_pred_raw = None
        self.y_pred_smoothed = None
        self.y_pred_final = None
        self.dates = None
        self.errors = None
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def load_data(self) -> None:
        """
        Loads predictions and metadata from files.
        """
        print("="*80)
        print("Loading evaluation data...")
        print("="*80)
        
        # Load predictions
        self.predictions_df = pd.read_csv(self.predictions_path)
        print(f"✓ Predictions loaded: {len(self.predictions_df):,} samples")
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print("\nTraining Metadata:")
        print(f"  Training date:    {self.metadata['training_date']}")
        print(f"  Model type:       {self.metadata['model_type']}")
        print(f"  Features:         {self.metadata['n_features']}")
        print(f"  Best iteration:   {self.metadata['best_iteration']}")
        print(f"  Used smoothing:   {self.metadata['used_smoothing']}")
        
        # Extract arrays
        self.y_true = self.predictions_df['actual'].values
        self.y_pred_raw = self.predictions_df['predicted_raw'].values
        self.y_pred_smoothed = self.predictions_df['predicted_smoothed'].values
        self.y_pred_final = self.predictions_df['final_prediction'].values
        self.dates = pd.to_datetime(self.predictions_df['date'])
        self.errors = np.abs(self.y_true - self.y_pred_final)
        
        print(f"\nData shapes:")
        print(f"  Actual:            {self.y_true.shape}")
        print(f"  Predicted (raw):   {self.y_pred_raw.shape}")
        print(f"  Predicted (final): {self.y_pred_final.shape}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         name: str = "Model") -> Dict[str, Any]:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            name (str): Name identifier for this prediction set
            
        Returns:
            Dict containing all metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'name': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def compare_all_metrics(self) -> pd.DataFrame:
        """
        Calculates and compares metrics for all prediction types.
        
        Returns:
            pd.DataFrame: Comparison table of all metrics
        """
        metrics_raw = self.calculate_metrics(self.y_true, self.y_pred_raw, "Raw Predictions")
        metrics_smoothed = self.calculate_metrics(self.y_true, self.y_pred_smoothed, "Smoothed Predictions")
        metrics_final = self.calculate_metrics(self.y_true, self.y_pred_final, "Final Predictions")
        
        metrics_df = pd.DataFrame([metrics_raw, metrics_smoothed, metrics_final])
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS COMPARISON")
        print("="*80)
        print(f"\n{metrics_df.to_string(index=False)}")
        print("\n" + "="*80)
        
        return metrics_df
    
    def analyze_error_distribution(self) -> pd.DataFrame:
        """
        Analyzes prediction errors across different ranges.
        
        Returns:
            pd.DataFrame: Error distribution statistics
        """
        print("\n" + "="*80)
        print("ERROR DISTRIBUTION")
        print("="*80)
        
        # Define error ranges
        ranges = [5, 10, 20, 30, 50, 100, 200]
        print(f"\n{'Error Range':<15} {'Count':<10} {'Percentage':<12} {'Cumulative'}")
        print("-" * 60)
        
        distribution_data = []
        cumsum = 0
        
        for i, threshold in enumerate(ranges):
            if i == 0:
                count = (self.errors <= threshold).sum()
                label = f"0-{threshold}"
            else:
                count = ((self.errors > ranges[i-1]) & (self.errors <= threshold)).sum()
                label = f"{ranges[i-1]}-{threshold}"
            
            pct = count / len(self.errors) * 100
            cumsum += pct
            print(f"{label:<15} {count:<10} {pct:>10.2f}%    {cumsum:>10.2f}%")
            
            distribution_data.append({
                'range': label,
                'count': count,
                'percentage': pct,
                'cumulative': cumsum
            })
        
        # Large errors
        count = (self.errors > ranges[-1]).sum()
        pct = count / len(self.errors) * 100
        cumsum += pct
        print(f"{f'>{ranges[-1]}':<15} {count:<10} {pct:>10.2f}%    {cumsum:>10.2f}%")
        
        distribution_data.append({
            'range': f'>{ranges[-1]}',
            'count': count,
            'percentage': pct,
            'cumulative': cumsum
        })
        
        return pd.DataFrame(distribution_data)
    
    def calculate_accuracy_by_tolerance(self) -> pd.DataFrame:
        """
        Calculates percentage of predictions within acceptable error margins.
        
        Returns:
            pd.DataFrame: Accuracy results for different tolerance levels
        """
        print("\n" + "="*80)
        print("ACCURACY BY TOLERANCE")
        print("="*80)
        print(f"\n{'Tolerance':<15} {'Within Count':<15} {'Percentage'}")
        print("-" * 50)
        
        tolerances = [5, 10, 15, 20, 30, 50, 100, 150, 200]
        accuracy_results = []
        
        for tol in tolerances:
            within = (self.errors <= tol).sum()
            pct = within / len(self.errors) * 100
            accuracy_results.append({'tolerance': tol, 'count': within, 'percentage': pct})
            print(f"±{tol:<14} {within:<15} {pct:>10.1f}%")
        
        return pd.DataFrame(accuracy_results)
    
    def calculate_error_statistics(self) -> Dict[str, float]:
        """
        Calculates comprehensive error statistics.
        
        Returns:
            Dict containing error statistics
        """
        print("\n" + "="*80)
        print("ERROR STATISTICS")
        print("="*80)
        
        stats = {
            'mean_error': self.errors.mean(),
            'median_error': np.median(self.errors),
            'std_error': self.errors.std(),
            'min_error': self.errors.min(),
            'max_error': self.errors.max(),
            'q25_error': np.percentile(self.errors, 25),
            'q75_error': np.percentile(self.errors, 75),
            'p95_error': np.percentile(self.errors, 95),
            'p99_error': np.percentile(self.errors, 99)
        }
        
        print(f"\nMean Error:       {stats['mean_error']:.2f} Wh")
        print(f"Median Error:     {stats['median_error']:.2f} Wh")
        print(f"Std Dev:          {stats['std_error']:.2f} Wh")
        print(f"Min Error:        {stats['min_error']:.2f} Wh")
        print(f"Max Error:        {stats['max_error']:.2f} Wh")
        print(f"25th Percentile:  {stats['q25_error']:.2f} Wh")
        print(f"75th Percentile:  {stats['q75_error']:.2f} Wh")
        print(f"95th Percentile:  {stats['p95_error']:.2f} Wh")
        print(f"99th Percentile:  {stats['p99_error']:.2f} Wh")
        
        return stats
    
    def analyze_by_usage_level(self) -> pd.DataFrame:
        """
        Analyzes performance across different usage levels.
        
        Returns:
            pd.DataFrame: Performance metrics by usage level
        """
        # Define usage levels
        levels = [
            (0, 100, "Low (0-100)"),
            (100, 200, "Medium (100-200)"),
            (200, 300, "High (200-300)"),
            (300, np.inf, "Very High (>300)")
        ]
        
        results = []
        for min_val, max_val, label in levels:
            mask = (self.y_true >= min_val) & (self.y_true < max_val)
            if mask.sum() > 0:
                level_errors = self.errors[mask]
                results.append({
                    'Level': label,
                    'Count': mask.sum(),
                    'MAE': level_errors.mean(),
                    'Median Error': np.median(level_errors),
                    'Max Error': level_errors.max()
                })
        
        usage_analysis = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("PERFORMANCE BY USAGE LEVEL")
        print("="*80)
        print(f"\n{usage_analysis.to_string(index=False)}")
        
        return usage_analysis
    
    def analyze_by_hour(self) -> pd.DataFrame:
        """
        Analyzes performance by hour of day.
        
        Returns:
            pd.DataFrame: Hourly performance statistics
        """
        df = self.predictions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['error'] = np.abs(df['actual'] - df['final_prediction'])
        
        hourly_stats = (
            df.groupby('hour')['error']
            .agg(
                Mean_Error='mean',
                Median_Error='median',
                Mode_Error=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                Count='count'
            )
            .reset_index()
            .rename(columns={'hour': 'Hour'})
        )
        
        print("\n" + "="*80)
        print("PERFORMANCE BY HOUR OF DAY")
        print("="*80)
        print(hourly_stats.to_string(index=False))
        
        return hourly_stats
    
    def plot_timeseries_comparison(self, output_dir: str = 'models/evaluation',
                                   n_plot: int = 1000) -> None:
        """
        Creates time series plot of actual vs predicted values.
        
        Args:
            output_dir (str): Directory to save plot
            n_plot (int): Number of points to plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_plot = min(n_plot, len(self.y_true))
        indices = np.linspace(0, len(self.y_true)-1, n_plot, dtype=int)
        
        plt.figure(figsize=(15, 6))
        plt.plot(indices, self.y_true[indices], label='Actual', alpha=0.7, 
                linewidth=1.5, color='blue')
        plt.plot(indices, self.y_pred_final[indices], label='Predicted', alpha=0.7, 
                linewidth=1.5, color='red')
        plt.xlabel('Time Index')
        plt.ylabel('Appliances (Wh)')
        plt.title(f'Actual vs Predicted Values (Sample of {n_plot} points)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir/ 'timeseries_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Time series plot saved to: {save_path}")
    
    def plot_scatter(self, output_dir: str = 'models/evaluation',
                    max_points: int = 5000) -> None:
        """
        Creates scatter plot of actual vs predicted values.
        
        Args:
            output_dir (str): Directory to save plot
            max_points (int): Maximum points to plot (for clarity)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample for clarity if dataset is large
        if len(self.y_true) > max_points:
            sample_idx = np.random.choice(len(self.y_true), max_points, replace=False)
            y_true_plot = self.y_true[sample_idx]
            y_pred_plot = self.y_pred_final[sample_idx]
        else:
            y_true_plot = self.y_true
            y_pred_plot = self.y_pred_final
        
        # Calculate R² for title
        r2 = r2_score(self.y_true, self.y_pred_final)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)
        plt.plot([self.y_true.min(), self.y_true.max()], 
                [self.y_true.min(), self.y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Values (Wh)')
        plt.ylabel('Predicted Values (Wh)')
        plt.title(f'Actual vs Predicted Scatter Plot (R² = {r2:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir/ 'scatter_plot.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scatter plot saved to: {save_path}")
    
    def plot_error_distribution(self, output_dir: str = 'models/evaluation') -> None:
        """
        Creates histogram of error distribution.
        
        Args:
            output_dir (str): Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.hist(self.errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.errors.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.errors.mean():.2f} Wh')
        plt.axvline(np.median(self.errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(self.errors):.2f} Wh')
        plt.xlabel('Absolute Error (Wh)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir/ 'error_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error distribution plot saved to: {save_path}")
    
    def plot_residuals(self, output_dir: str = 'models/evaluation') -> None:
        """
        Creates residual plot to check for bias.
        
        Args:
            output_dir (str): Directory to save plot
        """
        os.makedirs(output_dir, exist_ok=True)
        
        residuals = self.y_true - self.y_pred_final
        
        plt.figure(figsize=(12, 6))
        plt.scatter(self.y_pred_final, residuals, alpha=0.3, s=10)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values (Wh)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir/ 'residual_plot.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Residual plot saved to: {save_path}")
    
    def generate_evaluation_report(self, metrics_df: pd.DataFrame,
                                  accuracy_df: pd.DataFrame,
                                  usage_analysis: pd.DataFrame,
                                  error_stats: Dict[str, float],
                                  output_dir: str = 'models/evaluation') -> None:
        """
        Generates comprehensive evaluation report in JSON format.
        
        Args:
            metrics_df (pd.DataFrame): Metrics comparison dataframe
            accuracy_df (pd.DataFrame): Accuracy by tolerance dataframe
            usage_analysis (pd.DataFrame): Performance by usage level
            error_stats (Dict): Error statistics dictionary
            output_dir (str): Directory to save report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get final metrics
        metrics_final = metrics_df[metrics_df['name'] == 'Final Predictions'].iloc[0]
        metrics_raw = metrics_df[metrics_df['name'] == 'Raw Predictions'].iloc[0]
        metrics_smoothed = metrics_df[metrics_df['name'] == 'Smoothed Predictions'].iloc[0]
        
        report = {
            'model_info': {
                'model_type': self.metadata['model_type'],
                'training_date': self.metadata['training_date'],
                'n_features': self.metadata['n_features'],
                'best_iteration': self.metadata['best_iteration']
            },
            'dataset_info': {
                'n_train': self.metadata['n_train_samples'],
                'n_val': self.metadata['n_val_samples'],
                'n_test': self.metadata['n_test_samples']
            },
            'performance_metrics': {
                'mae': float(metrics_final['MAE']),
                'rmse': float(metrics_final['RMSE']),
                'r2': float(metrics_final['R2']),
                'mape': float(metrics_final['MAPE'])
            },
            'error_statistics': {
                'mean_error': float(error_stats['mean_error']),
                'median_error': float(error_stats['median_error']),
                'std_error': float(error_stats['std_error']),
                'max_error': float(error_stats['max_error']),
                'p95_error': float(error_stats['p95_error'])
            },
            'accuracy_by_tolerance': {
                f'within_{int(row.tolerance)}': float(row.percentage)
                for row in accuracy_df.itertuples()
            },
            'performance_by_usage': usage_analysis.to_dict('records'),
            'smoothing_impact': {
                'raw_mae': float(metrics_raw['MAE']),
                'smoothed_mae': float(metrics_smoothed['MAE']),
                'improvement': float(metrics_raw['MAE'] - metrics_smoothed['MAE'])
            }
        }
        
        report_path = os.path.join(output_dir/'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Evaluation report saved to: {report_path}")
    
    def print_final_summary(self, metrics_df: pd.DataFrame, 
                           accuracy_df: pd.DataFrame) -> None:
        """
        Prints final evaluation summary.
        
        Args:
            metrics_df (pd.DataFrame): Metrics comparison dataframe
            accuracy_df (pd.DataFrame): Accuracy by tolerance dataframe
        """
        metrics_final = metrics_df[metrics_df['name'] == 'Final Predictions'].iloc[0]
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE - FINAL SUMMARY")
        print("="*80)
        
        print(f"\n{'METRIC':<20} {'VALUE'}")
        print("-" * 50)
        print(f"{'MAE':<20} {metrics_final['MAE']:.2f} Wh")
        print(f"{'RMSE':<20} {metrics_final['RMSE']:.2f} Wh")
        print(f"{'R² Score':<20} {metrics_final['R2']:.4f}")
        print(f"{'MAPE':<20} {metrics_final['MAPE']:.2f}%")
        
        print(f"\n{'ACCURACY':<20} {'VALUE'}")
        print("-" * 50)
        print(f"{'Within ±10 Wh':<20} {accuracy_df[accuracy_df['tolerance']==10]['percentage'].values[0]:.1f}%")
        print(f"{'Within ±20 Wh':<20} {accuracy_df[accuracy_df['tolerance']==20]['percentage'].values[0]:.1f}%")
        print(f"{'Within ±50 Wh':<20} {accuracy_df[accuracy_df['tolerance']==50]['percentage'].values[0]:.1f}%")
        
        print(f"\n{'ERROR STATS':<20} {'VALUE'}")
        print("-" * 50)
        print(f"{'Mean Error':<20} {self.errors.mean():.2f} Wh")
        print(f"{'Median Error':<20} {np.median(self.errors):.2f} Wh")
        print(f"{'95th Percentile':<20} {np.percentile(self.errors, 95):.2f} Wh")
        
        print("\n" + "="*80)
        print("✓ All evaluation complete!")
        print("="*80)
    
    def run_full_evaluation(self, output_dir: str = 'models/evaluation') -> Dict[str, Any]:
        """
        Runs complete evaluation pipeline.
        
        Args:
            output_dir (str): Directory to save all outputs
            
        Returns:
            Dict containing all evaluation results
        """
        # Load data
        self.load_data()
        
        # Calculate metrics
        metrics_df = self.compare_all_metrics()
        
        # Error analysis
        distribution_df = self.analyze_error_distribution()
        accuracy_df = self.calculate_accuracy_by_tolerance()
        error_stats = self.calculate_error_statistics()
        
        # Performance by segments
        usage_analysis = self.analyze_by_usage_level()
        hourly_analysis = self.analyze_by_hour()
        
        # Create visualizations
        print("\n" + "="*80)
        print("Creating visualizations...")
        print("="*80)
        self.plot_timeseries_comparison(output_dir)
        self.plot_scatter(output_dir)
        self.plot_error_distribution(output_dir)
        self.plot_residuals(output_dir)
        
        # Generate report
        print("\n" + "="*80)
        print("Generating evaluation report...")
        print("="*80)
        self.generate_evaluation_report(metrics_df, accuracy_df, usage_analysis, 
                                       error_stats, output_dir)
        
        # Print summary
        self.print_final_summary(metrics_df, accuracy_df)
        
        return {
            'metrics': metrics_df,
            'accuracy': accuracy_df,
            'usage_analysis': usage_analysis,
            'hourly_analysis': hourly_analysis,
            'error_stats': error_stats,
            'distribution': distribution_df
        }


def main():
    """
    Main execution function for model evaluation.
    """
    # Configuration
    PREDICTIONS_PATH = paths.test_predictions          # points to models/predictions/test_predictions.csv
    METADATA_PATH = paths.training_metadata            # points to models/training_metadata.json
    OUTPUT_DIR = paths.evaluation_dir                 # points to models/evaluation
        
    print("="*80)
    print("Energy Prediction - Model Evaluation Pipeline")
    print("="*80)
    
    # Initialize evaluator
    evaluator = EnergyModelEvaluator(PREDICTIONS_PATH, METADATA_PATH)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(OUTPUT_DIR)
    
    return evaluator, results


if __name__ == "__main__":
    evaluator, results = main()