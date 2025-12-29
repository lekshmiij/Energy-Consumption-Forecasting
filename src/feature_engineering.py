"""
Feature Engineering and Preprocessing Module for Energy Prediction
===================================================================
This module contains all feature engineering functions for the appliances 
energy prediction project. 

Author: Lekshmi
Date: 2025
"""
import os
print("Current working directory:", os.getcwd())

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List
from config import data_paths



warnings.filterwarnings('ignore')


class EnergyFeatureEngineering:
    """
    A comprehensive feature engineering pipeline for energy consumption prediction.
    
    This class handles the creation of features across multiple categories:
    - Time-based features (8)
    - Lag features (7)
    - Rolling statistics (5 means, 6 extremes, 4 percentiles)
    - Momentum features (3)
    - Relative position features (6)
    - Volatility features (1)
    - Exponential moving averages (2)
    - Usage regime (1)
    - Context flags (5)
    - Spike detection (3)
    - Interaction features (2)
    - Historical patterns (1)
    - Trend features (1)
    """
    
    def __init__(self, horizon_hours: int = 2):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            horizon_hours (int): Prediction horizon in hours (default: 2)
        """
        self.horizon_hours = horizon_hours
        self.feature_names = self._get_feature_names()
    
    def create_essential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates base features including target variable and time components.
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' and 'Appliances' columns
            
        Returns:
            pd.DataFrame: DataFrame with base features added
        """
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Create target variable (future appliance usage)
        data['target'] = data['Appliances'].shift(-self.horizon_hours)
        
        # Extract base time components
        data['hour'] = data['date'].dt.hour
        data['day_of_week'] = data['date'].dt.dayofweek
        data['month'] = data['date'].dt.month
        
        # Calculate target timestamp features
        data['target_date'] = data['date'] + pd.Timedelta(hours=self.horizon_hours)
        data['target_hour'] = data['target_date'].dt.hour
        data['target_dow'] = data['target_date'].dt.dayofweek
        
        return data
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cyclical time encodings and discrete time components (8 features).
        
        Cyclical encoding ensures continuity (e.g., 23:00 and 00:00 are close).
        
        Args:
            data (pd.DataFrame): Input data with time columns
            
        Returns:
            pd.DataFrame: Data with time features added
        """
        # Hour cyclical encoding (captures 24-hour cycle)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # Target hour cyclical encoding
        data['target_hour_sin'] = np.sin(2 * np.pi * data['target_hour'] / 24)
        
        # Day of week cyclical encoding (captures weekly patterns)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        
        # Month cyclical encoding (captures seasonal patterns)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        
        return data
    
    def add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag features at strategic time intervals (7 features).
        
        Historical values at key intervals capture temporal dependencies.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with lag features added
        """
        # Short-term lags (immediate past influence)
        data['lag_1h'] = data['Appliances'].shift(1)
        data['lag_2h'] = data['Appliances'].shift(2)
        data['lag_3h'] = data['Appliances'].shift(3)
        
        # Medium-term lags (recent patterns)
        data['lag_6h'] = data['Appliances'].shift(6)
        data['lag_12h'] = data['Appliances'].shift(12)
        data['lag_24h'] = data['Appliances'].shift(24)  # Yesterday same hour
        
        # Long-term lag (weekly pattern)
        data['lag_168h'] = data['Appliances'].shift(168)  # Last week same hour
        
        return data
    
    def add_rolling_means(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling averages over multiple time windows (5 features).
        
        Moving averages smooth noise and capture trend levels.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with rolling mean features added
        """
        # 3-hour rolling mean (very short-term trend)
        data['roll_3h_mean'] = data['Appliances'].shift(1).rolling(3, min_periods=1).mean()
        
        # 6-hour rolling mean (short-term trend)
        data['roll_6h_mean'] = data['Appliances'].shift(1).rolling(6, min_periods=2).mean()
        
        # 12-hour rolling mean (half-day trend)
        data['roll_12h_mean'] = data['Appliances'].shift(1).rolling(12, min_periods=4).mean()
        
        # 24-hour rolling mean (daily baseline)
        data['roll_24h_mean'] = data['Appliances'].shift(1).rolling(24, min_periods=8).mean()
        
        # 168-hour rolling mean (weekly baseline)
        data['roll_168h_mean'] = data['Appliances'].shift(1).rolling(168, min_periods=56).mean()
        
        return data
    
    def add_rolling_extremes(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, object, object, object]:
        """
        Extracts min and max values from rolling windows (6 features).
        
        Min/max values identify volatility ranges and boundaries.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Tuple containing:
                - pd.DataFrame: Data with extreme features added
                - rolled_6: 6-hour rolling window object
                - rolled_24: 24-hour rolling window object
                - rolled_168: 168-hour rolling window object
        """
        # Create rolling window objects
        rolled_6 = data['Appliances'].shift(1).rolling(6, min_periods=2)
        rolled_12 = data['Appliances'].shift(1).rolling(12, min_periods=4)
        rolled_24 = data['Appliances'].shift(1).rolling(24, min_periods=8)
        rolled_168 = data['Appliances'].shift(1).rolling(168, min_periods=56)
        
        # 6-hour extremes (short-term range)
        data['roll_6h_max'] = rolled_6.max()
        data['roll_6h_min'] = rolled_6.min()
        
        # 12-hour extremes
        data['roll_12h_max'] = rolled_12.max()
        data['roll_12h_min'] = rolled_12.min()
        
        # 24-hour and 168-hour minimums (baseline floors)
        data['roll_24h_min'] = rolled_24.min()
        data['roll_168h_min'] = rolled_168.min()
        
        return data, rolled_6, rolled_24, rolled_168
    
    def add_rolling_percentiles(self, data: pd.DataFrame, rolled_24: object, 
                                rolled_168: object) -> pd.DataFrame:
        """
        Calculates percentiles from rolling distributions (4 features).
        
        Quantiles capture distribution shape and outlier context.
        
        Args:
            data (pd.DataFrame): Input data
            rolled_24: 24-hour rolling window object
            rolled_168: 168-hour rolling window object
            
        Returns:
            pd.DataFrame: Data with percentile features added
        """
        # Weekly percentiles (long-term distribution)
        data['roll_168h_median'] = rolled_168.median()
        data['roll_168h_q25'] = rolled_168.quantile(0.25)
        data['roll_168h_q75'] = rolled_168.quantile(0.75)
        
        # Daily median (short-term center)
        data['roll_24h_median'] = rolled_24.median()
        
        return data
    
    def add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes change rates over different time scales (3 features).
        
        Rate of change indicates acceleration or deceleration trends.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with momentum features added
        """
        # 1-hour momentum (immediate change)
        data['momentum_1h'] = data['Appliances'] - data['lag_1h']
        
        # 6-hour momentum (short-term change)
        data['momentum_6h'] = data['Appliances'] - data['lag_6h']
        
        # 24-hour momentum (daily change)
        data['momentum_24h'] = data['Appliances'] - data['lag_24h']
        
        return data
    
    def add_relative_position(self, data: pd.DataFrame, rolled_6: object, 
                             rolled_168: object) -> pd.DataFrame:
        """
        Creates relative positioning and z-score features (6 features).
        
        Normalized metrics show current value context within distributions.
        
        Args:
            data (pd.DataFrame): Input data
            rolled_6: 6-hour rolling window object
            rolled_168: 168-hour rolling window object
            
        Returns:
            pd.DataFrame: Data with relative position features added
        """
        # Distance from minimums (how far above baseline)
        data['dist_from_24h_min'] = data['Appliances'] - data['roll_24h_min']
        data['dist_from_6h_min'] = data['Appliances'] - data['roll_6h_min']
        
        # Ratio to means (relative magnitude)
        data['rel_to_24h_mean'] = data['Appliances'] / (data['roll_24h_mean'] + 1)
        data['rel_to_6h_mean'] = data['Appliances'] / (data['roll_6h_mean'] + 1)
        
        # Z-scores (standardized deviations)
        data['zscore_168h'] = (data['Appliances'] - data['roll_168h_mean']) / (rolled_168.std() + 1)
        data['zscore_6h'] = (data['Appliances'] - data['roll_6h_mean']) / (rolled_6.std() + 1)
        
        return data
    
    def add_volatility(self, data: pd.DataFrame, rolled_24: object) -> pd.DataFrame:
        """
        Calculates range as volatility measure (1 feature).
        
        Range captures variability over time window.
        
        Args:
            data (pd.DataFrame): Input data
            rolled_24: 24-hour rolling window object
            
        Returns:
            pd.DataFrame: Data with volatility feature added
        """
        # 24-hour range (max - min = daily volatility)
        data['range_24h'] = rolled_24.max() - data['roll_24h_min']
        
        return data
    
    def add_ema_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates exponentially weighted moving averages (2 features).
        
        EMAs weight recent values more heavily than simple moving averages.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with EMA features added
        """
        # 3-hour EMA (fast-reacting trend)
        data['ema_3h'] = data['Appliances'].ewm(span=3, adjust=False).mean().shift(1)
        
        # 6-hour EMA (balanced trend)
        data['ema_6h'] = data['Appliances'].ewm(span=6, adjust=False).mean().shift(1)
        
        return data
    
    def add_usage_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Bins appliance usage into discrete regimes (1 feature).
        
        Categorical bins identify low/medium/high/very-high usage states.
        This is the 2nd most important feature!
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with usage regime feature added
        """
        # Categorize into 4 usage levels
        data['usage_regime'] = pd.cut(data['Appliances'], 
                                       bins=[0, 100, 200, 300, np.inf],
                                       labels=[0, 1, 2, 3]).astype(int)
        
        return data
    
    def add_context_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates binary flags for time-of-day and weekend periods (5 features).
        
        Binary indicators for specific time periods and conditions.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with context flag features added
        """
        # Time of day flags
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 5)).astype(int)
        data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] <= 9)).astype(int)
        data['is_evening'] = ((data['hour'] >= 17) & (data['hour'] <= 21)).astype(int)
        
        # Weekend flag
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # Target hour peak flag (high usage evening hours)
        data['target_is_peak'] = ((data['target_hour'] >= 18) & (data['target_hour'] <= 20)).astype(int)
        
        return data
    
    def add_spike_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detects local extrema and spike intensity (3 features).
        
        Identify local peaks, troughs, and anomalous intensity levels.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with spike detection features added
        """
        # Local peak (higher than neighbors)
        data['is_local_peak'] = ((data['Appliances'] > data['lag_1h']) & 
                                  (data['Appliances'] > data['Appliances'].shift(-1))).astype(int)
        
        # Local trough (lower than neighbors)
        data['is_local_trough'] = ((data['Appliances'] < data['lag_1h']) & 
                                    (data['Appliances'] < data['Appliances'].shift(-1))).astype(int)
        
        # Spike intensity (ratio to 24h baseline)
        data['spike_intensity_24h'] = data['Appliances'] / (data['roll_24h_mean'] + 1)
        
        return data
    
    def add_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates interaction terms between categorical and continuous features (2 features).
        
        Multiplicative features capture combined effects.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with interaction features added
        """
        # Evening period weighted by usage level
        data['evening_x_level'] = data['is_evening'] * data['Appliances']
        
        # Weekend weighted by usage level
        data['weekend_x_level'] = data['is_weekend'] * data['Appliances']
        
        return data
    
    def add_historical_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes historical average for target hour (1 feature).
        
        Average usage at the same hour across previous weeks.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with historical pattern feature added
        """
        # Rolling average of this hour across past weeks
        data['avg_this_hour'] = data.groupby('target_hour')['Appliances'].transform(
            lambda x: x.shift(1).rolling(168, min_periods=24).mean()
        )
        
        return data
    
    def add_trend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 6-hour linear trend slope (1 feature).
        
        Linear slope captures directional movement.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with trend feature added
        """
        # Trend direction over recent 6 hours
        data['trend_6h'] = data['Appliances'].shift(1).rolling(6).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 6 else 0, 
            raw=True
        )
        
        return data
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master function that orchestrates all feature engineering steps.
        
        Creates all 49 engineered features in the correct order.
        
        Args:
            df (pd.DataFrame): DataFrame with 'date' and 'Appliances' columns
            
        Returns:
            pd.DataFrame: DataFrame with 49 engineered features + target
        """
        print("Starting feature engineering pipeline...")
        
        # Initialize base features
        data = self.create_essential_features(df)
        print("✓ Base features created")
        
        # Add feature groups sequentially
        data = self.add_time_features(data)
        print("✓ Time features added (8)")
        
        data = self.add_lag_features(data)
        print("✓ Lag features added (7)")
        
        data = self.add_rolling_means(data)
        print("✓ Rolling means added (5)")
        
        data, rolled_6, rolled_24, rolled_168 = self.add_rolling_extremes(data)
        print("✓ Rolling extremes added (6)")
        
        data = self.add_rolling_percentiles(data, rolled_24, rolled_168)
        print("✓ Rolling percentiles added (4)")
        
        data = self.add_momentum_features(data)
        print("✓ Momentum features added (3)")
        
        data = self.add_relative_position(data, rolled_6, rolled_168)
        print("✓ Relative position features added (6)")
        
        data = self.add_volatility(data, rolled_24)
        print("✓ Volatility feature added (1)")
        
        data = self.add_ema_features(data)
        print("✓ EMA features added (2)")
        
        data = self.add_usage_regime(data)
        print("✓ Usage regime added (1)")
        
        data = self.add_context_flags(data)
        print("✓ Context flags added (5)")
        
        data = self.add_spike_detection(data)
        print("✓ Spike detection features added (3)")
        
        data = self.add_interactions(data)
        print("✓ Interaction features added (2)")
        
        data = self.add_historical_patterns(data)
        print("✓ Historical pattern added (1)")
        
        data = self.add_trend(data)
        print("✓ Trend feature added (1)")
        
        print(f"\n✓ Feature engineering complete! Total features: {len(self.feature_names)}")
        
        return data
    
    def _get_feature_names(self) -> List[str]:
        """
        Returns ordered list of all 49 feature names for model training.
        
        Returns:
            List[str]: List of feature names
        """
        features = [
            # Time features (8)
            'hour_sin', 'hour_cos', 'target_hour_sin', 'dow_sin', 'month_sin',
            'day_of_week', 'target_dow', 'month',
            
            # Lag features (7)
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_12h', 'lag_24h', 'lag_168h',
            
            # Rolling means (5)
            'roll_3h_mean', 'roll_6h_mean', 'roll_12h_mean', 'roll_24h_mean', 'roll_168h_mean',
            
            # Rolling extremes (6)
            'roll_6h_max', 'roll_6h_min', 'roll_12h_max', 'roll_12h_min', 
            'roll_24h_min', 'roll_168h_min',
            
            # Percentiles (4)
            'roll_168h_median', 'roll_168h_q25', 'roll_168h_q75', 'roll_24h_median',
            
            # Momentum (3)
            'momentum_1h', 'momentum_6h', 'momentum_24h',
            
            # Relative position (6)
            'dist_from_24h_min', 'dist_from_6h_min', 'rel_to_24h_mean', 
            'rel_to_6h_mean', 'zscore_168h', 'zscore_6h',
            
            # Volatility (1)
            'range_24h',
            
            # EMAs (2)
            'ema_3h', 'ema_6h',
            
            # Regime (1)
            'usage_regime',
            
            # Context flags (5)
            'is_night', 'is_morning', 'is_evening', 'is_weekend', 'target_is_peak',
            
            # Spike detection (3)
            'is_local_peak', 'is_local_trough', 'spike_intensity_24h',
            
            # Interactions (2)
            'evening_x_level', 'weekend_x_level',
            
            # Historical patterns (1)
            'avg_this_hour',
            
            # Trend (1)
            'trend_6h'
        ]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Public method to get feature names.
        
        Returns:
            List[str]: List of all 49 feature names
        """
        return self.feature_names
    
    def prepare_final_dataset(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the final dataset with proper column ordering and cleaning.
        
        Args:
            df_features (pd.DataFrame): DataFrame with engineered features
            
        Returns:
            pd.DataFrame: Clean final dataset ready for modeling
        """
        # Extract features and target
        X = df_features[self.feature_names].fillna(0)
        y = df_features['target']
        
        # Combine features and target for saving
        final_data = X.copy()
        final_data['target'] = y
        final_data['date'] = df_features['date']
        
        # Reorder columns (date first, target last)
        cols = ['date'] + self.feature_names + ['target']
        final_data = final_data[cols]
        
        # Remove rows with missing target values
        final_data = final_data.dropna()
        
        print(f"\nFinal dataset shape: {final_data.shape}")
        print(f"Missing values: {final_data.isnull().sum().sum()}")
        
        return final_data


def save_engineered_features(final_data: pd.DataFrame, feature_names: List[str]) -> None:

    """
    Saves the engineered features and feature names to files.
    
    Args:
        final_data (pd.DataFrame): Final dataset with all features
        feature_names (List[str]): List of feature names
        output_dir (str): Directory to save processed data
        features_dir (str): Directory to save feature names
    """
    # Create directories if they don't exist
    data_paths.processed_dir.mkdir(parents=True, exist_ok=True)

    
    # Save complete engineered dataset
    final_data.to_csv(data_paths.engineered_features, index=False)
    print(f"✓ Saved processed features to: {data_paths.engineered_features}")

    
    # Save feature names
    feature_names_path = data_paths.processed_dir / "feature_names.txt"
    with open(feature_names_path, "w") as f:
        f.write("\n".join(feature_names))
    print(f"✓ Saved feature names to: {feature_names_path}")



def main():
    """
    Main execution function for the feature engineering pipeline.
    """
    # Configuration
    INPUT_PATH = data_paths.raw_data
    #INPUT_PATH = 'C:/Users/lekshmi/Desktop/ml projects/appliances energy prediction/KAG_energydata_complete.csv'
    HORIZON_HOURS = 2
    
    print("="*70)
    print("Energy Prediction - Feature Engineering Pipeline")
    print("="*70)
    
    # Load raw data
    print(f"\nLoading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize feature engineering pipeline
    fe = EnergyFeatureEngineering(horizon_hours=HORIZON_HOURS)
    
    # Apply feature engineering
    print("\n" + "="*70)
    df_features = fe.engineer_all_features(df[['date', 'Appliances']])
    print(f"Engineered data shape: {df_features.shape}")
    
    # Prepare final dataset
    print("\n" + "="*70)
    print("Preparing final dataset...")
    final_data = fe.prepare_final_dataset(df_features)
    
    # Save results
    print("\n" + "="*70)
    print("Saving results...")
    save_engineered_features(final_data, fe.get_feature_names())
    
    print("\n" + "="*70)
    print("✓ Feature engineering pipeline completed successfully!")
    print("="*70)
    
    return final_data, fe


if __name__ == "__main__":
    final_data, feature_engineer = main()