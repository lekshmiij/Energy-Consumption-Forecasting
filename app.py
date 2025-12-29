"""
Gradio Deployment for Appliance Energy Prediction
==================================================
Interactive web interface for predicting appliance energy consumption
with spike risk assessment.

Features:
- Date-time picker constrained to test data range
- Energy consumption prediction
- Spike risk classification (High/Moderate/Low)
- Visual risk indicators and detailed explanations

Author: Lekshmi
Date: 2025
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
import warnings
from src.config import DataPaths, ModelPaths


warnings.filterwarnings('ignore')


class EnergyPredictionApp:
    """
    Gradio application for energy prediction with spike risk assessment.
    """
    
    def __init__(self, model_path: str, test_data_path: str, 
                 feature_names_path: str):
        """
        Initialize the prediction application.
        
        Args:
            model_path (str): Path to trained model pickle file
            test_data_path (str): Path to test dataset CSV
            feature_names_path (str): Path to feature names text file
        """
        self.model = None
        self.test_data = None
        self.feature_names = None
        self.available_dates = None
        self.date_range = None
        
        # Load all required data
        self._load_model(model_path)
        self._load_test_data(test_data_path)
        self._load_feature_names(feature_names_path)
        
        # Define spike risk windows
        self._define_risk_windows()
    
    def _load_model(self, model_path: str) -> None:
        """Load the trained XGBoost model."""
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully")
    
    def _load_test_data(self, test_data_path: str) -> None:
        """Load test dataset and extract available date-time ranges."""
        print(f"Loading test data from: {test_data_path}")
        self.test_data = pd.read_csv(test_data_path)
        self.test_data['date'] = pd.to_datetime(self.test_data['date'])
        
        # Extract available dates
        self.available_dates = self.test_data['date'].dt.date.unique()
        self.date_range = {
            'min': self.test_data['date'].min(),
            'max': self.test_data['date'].max()
        }
        
        print(f"✓ Test data loaded: {len(self.test_data)} samples")
        print(f"  Date range: {self.date_range['min']} to {self.date_range['max']}")
    
    def _load_feature_names(self, feature_names_path: str) -> None:
        """Load feature names."""
        print(f"Loading feature names from: {feature_names_path}")
        with open(feature_names_path, 'r') as f:
            self.feature_names = f.read().splitlines()
        print(f"✓ Loaded {len(self.feature_names)} features")
    
    def _define_risk_windows(self) -> None:
        """Define spike risk windows based on EDA patterns."""
        # High-Risk Windows (Extreme volatility)
        self.high_risk_windows = [
            {'day': 'Friday', 'hours': range(8, 17), 
             'reason': 'Highest daytime intensity among weekdays (10-12% changes ≥200 Wh)'},
            {'day': 'Saturday', 'hours': range(8, 17), 
             'reason': 'Most volatile period overall, event-driven usage with >1050 Wh spikes'},
            {'day': 'Monday', 'hours': range(12, 21), 
             'reason': 'Peak appliance usage window (up to 600 Wh), 33.3% changes in 400-600 Wh range'},
            {'day': 'Thursday', 'hours': range(16, 21), 
             'reason': 'Widest spread including rare 900-1200 Wh spikes'}
        ]
        
        # Moderate-Risk Windows (Structured volatility)
        self.moderate_risk_windows = [
            {'day': 'Tuesday', 'hours': range(16, 21), 
             'reason': '~17% changes ≥100 Wh, controlled mid-to-high usage (300-600 Wh)'},
            {'day': 'Wednesday', 'hours': list(range(8, 13)) + list(range(16, 21)), 
             'reason': 'Highest mid-to-large fluctuations among weekdays (150-450 Wh)'},
            {'day': 'Friday', 'hours': range(16, 21), 
             'reason': 'Active but not extreme, sustained elevated usage'},
            {'day': 'Sunday', 'hours': range(8, 21), 
             'reason': 'Sustained variability, moderate usage (150-450 Wh)'}
        ]
        
        # Low-Risk Windows (Stable, predictable)
        self.low_risk_windows = [
            {'day': 'All', 'hours': range(0, 9), 
             'reason': '≈98% usage ≤25 Wh baseline (night hours)'},
            {'day': 'Monday', 'hours': range(20, 25), 
             'reason': 'Sharp drop back to low usage after peak'},
            {'day': 'Wednesday', 'hours': range(0, 5), 
             'reason': 'Most stable early morning, zero activity beyond baseline'},
            {'day': 'Sunday', 'hours': range(20, 25), 
             'reason': 'Settle earlier, no extreme spikes'},
            {'day': 'Tuesday', 'hours': range(0, 9), 
             'reason': 'Stable morning period'},
            {'day': 'Thursday', 'hours': range(0, 9), 
             'reason': 'Stable morning period'}
        ]
    
    def get_available_times(self, selected_date: str) -> list:
        """
        Get available times for a selected date.
        
        Args:
            selected_date (str): Date in YYYY-MM-DD format
            
        Returns:
            list: Available time strings in HH:MM format
        """
        if selected_date is None:
            return []
        
        date_obj = pd.to_datetime(selected_date).date()
        
        # Filter test data for this date
        day_data = self.test_data[self.test_data['date'].dt.date == date_obj]
        
        if len(day_data) == 0:
            return []
        
        # Extract unique times
        times = day_data['date'].dt.strftime('%H:%M').unique().tolist()
        return sorted(times)
    
    def find_nearest_data_point(self, target_datetime: datetime) -> pd.Series:
        """
        Find the nearest data point in test set to the target datetime.
        
        Args:
            target_datetime (datetime): Target date and time
            
        Returns:
            pd.Series: Row from test data closest to target
        """
        # Calculate time differences
        time_diffs = abs(self.test_data['date'] - target_datetime)
        nearest_idx = time_diffs.idxmin()
        
        return self.test_data.loc[nearest_idx]
    
    def assess_spike_risk(self, timestamp: datetime) -> Tuple[str, str, str]:
        """
        Assess spike risk based on day and hour.
        
        Args:
            timestamp (datetime): Target datetime
            
        Returns:
            Tuple of (risk_level, badge, reason)
        """
        day_name = timestamp.strftime('%A')
        hour = timestamp.hour
        
        # Check high-risk windows
        for window in self.high_risk_windows:
            if (window['day'] == day_name or window['day'] == 'All') and hour in window['hours']:
                return (
                    'High',
                    '🔴 HIGH RISK',
                    f"**High Risk Period:** {window['reason']}"
                )
        
        # Check moderate-risk windows
        for window in self.moderate_risk_windows:
            if (window['day'] == day_name or window['day'] == 'All') and hour in window['hours']:
                return (
                    'Moderate',
                    '🟡 MODERATE RISK',
                    f"**Moderate Risk Period:** {window['reason']}"
                )
        
        # Check low-risk windows
        for window in self.low_risk_windows:
            if (window['day'] == day_name or window['day'] == 'All') and hour in window['hours']:
                return (
                    'Low',
                    '🟢 LOW RISK',
                    f"**Low Risk Period:** {window['reason']}"
                )
        
        # Default to moderate if not explicitly categorized
        return (
            'Moderate',
            '🟡 MODERATE RISK',
            '**Moderate Risk Period:** Standard operational hours with typical variability'
        )
    
    def predict_energy(self, date_str: str, time_str: str) -> Tuple[str, str, str]:
        """
        Predict energy consumption and assess spike risk.
        
        Args:
            date_str (str): Date in YYYY-MM-DD format
            time_str (str): Time in HH:MM format
            
        Returns:
            Tuple of (forecast_html, risk_html, details_html)
        """
        try:
            # Validate inputs
            if not date_str or not time_str:
                return (
                    "⚠️ Please select both date and time",
                    "",
                    ""
                )
            
            # Parse datetime
            datetime_str = f"{date_str} {time_str}"
            target_datetime = pd.to_datetime(datetime_str)
            
            # Find nearest data point
            data_point = self.find_nearest_data_point(target_datetime)
            
            # Extract features for prediction as DataFrame (XGBoost needs column names)
            feature_values = pd.DataFrame(
                [data_point[self.feature_names].values],
                columns=self.feature_names
            )
            
            # Make prediction
            import xgboost as xgb
            dmatrix = xgb.DMatrix(feature_values, feature_names=self.feature_names)
            prediction = self.model.predict(dmatrix)[0]
            
            # Ensure non-negative
            prediction = max(0, prediction)
            
            # Get actual value if available
            actual_value = data_point.get('target', None)
            
            # Assess spike risk
            risk_level, risk_badge, risk_reason = self.assess_spike_risk(target_datetime)
            
            # Format forecast output
            forecast_html = f"""
            <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center;">
                <h2 style="margin: 0 0 10px 0;">⚡ Energy Forecast</h2>
                <div style="font-size: 48px; font-weight: bold; margin: 15px 0;">
                    {prediction:.1f} Wh
                </div>
                <div style="font-size: 14px; opacity: 0.9;">
                    📅 {target_datetime.strftime('%A, %B %d, %Y at %H:%M')}
                </div>
            </div>
            """
            
            # Format risk output
            risk_color = {
                'High': '#ff4444',
                'Moderate': '#ffbb33',
                'Low': '#00C851'
            }[risk_level]
            
            risk_html = f"""
            <div style="padding: 20px; border-radius: 10px; background: {risk_color}; color: white; text-align: center;">
                <h2 style="margin: 0 0 10px 0;">Spike Risk Assessment</h2>
                <div style="font-size: 36px; font-weight: bold; margin: 15px 0;">
                    {risk_badge}
                </div>
                <div style="font-size: 14px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-top: 10px; text-align: left;">
                    {risk_reason}
                </div>
            </div>
            """
            
            # Format details
            day_name = target_datetime.strftime('%A')
            hour = target_datetime.hour
            
            details_html = f"""
            <div style="padding: 20px; border-radius: 10px; background: #f8f9fa; border: 1px solid #dee2e6;">
                <h3 style="margin-top: 0; color: #333;">📊 Prediction Details</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Day of Week:</td>
                        <td style="padding: 8px;">{day_name}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Hour:</td>
                        <td style="padding: 8px;">{hour}:00</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Is Weekend:</td>
                        <td style="padding: 8px;">{"Yes" if target_datetime.weekday() >= 5 else "No"}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Predicted Value:</td>
                        <td style="padding: 8px; color: #667eea; font-weight: bold;">{prediction:.2f} Wh</td>
                    </tr>
            """
            
            if actual_value is not None and not pd.isna(actual_value):
                error = abs(prediction - actual_value)
                error_pct = (error / actual_value * 100) if actual_value > 0 else 0
                details_html += f"""
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Actual Value:</td>
                        <td style="padding: 8px;">{actual_value:.2f} Wh</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #dee2e6;">
                        <td style="padding: 8px; font-weight: bold;">Absolute Error:</td>
                        <td style="padding: 8px;">{error:.2f} Wh ({error_pct:.1f}%)</td>
                    </tr>
                """
            
            details_html += """
                </table>
                <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 5px; font-size: 13px;">
                    <strong>ℹ️ Note:</strong> This prediction is based on historical patterns and 49 engineered features 
                    including temporal patterns, lag values, rolling statistics, and usage regime indicators.
                </div>
            </div>
            """
            
            return forecast_html, risk_html, details_html
            
        except Exception as e:
            error_msg = f"❌ Error during prediction: {str(e)}"
            return error_msg, "", ""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(
            title="⚡ Energy Prediction Dashboard",
            theme=gr.themes.Soft()
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # ⚡ Appliance Energy Prediction Dashboard
                
                Predict household appliance energy consumption with intelligent spike risk assessment.
                
                **How to use:**
                1. Select a date from the test dataset range
                2. Choose a specific time
                3. Click "Predict Energy" to get forecast and risk analysis
                """
            )
            
            # Input section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📅 Select Date & Time")
                    
                    date_input = gr.Dropdown(
                        choices=[str(d) for d in sorted(self.available_dates)],
                        label="Date",
                        info=f"Available dates: {self.date_range['min'].date()} to {self.date_range['max'].date()}",
                        interactive=True
                    )
                    
                    time_input = gr.Dropdown(
                        choices=[],
                        label="Time",
                        info="Select a date first to see available times",
                        interactive=True
                    )
                    
                    # Update times when date changes
                    def update_times(date):
                        if date:
                            times = self.get_available_times(date)
                            return gr.Dropdown(choices=times, value=times[0] if times else None)
                        return gr.Dropdown(choices=[])
                    
                    date_input.change(
                        fn=update_times,
                        inputs=date_input,
                        outputs=time_input
                    )
                    
                    predict_btn = gr.Button(
                        "🔮 Predict Energy",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown(
                        """
                        ---
                        **📊 Model Information**
                        - Algorithm: XGBoost Quantile Regression
                        - Features: 49 engineered features
                        - Horizon: 2 hours ahead
                        - Training: Temporal split (no data leakage)
                        """
                    )
            
            # Output section
            with gr.Row():
                with gr.Column(scale=1):
                    forecast_output = gr.HTML(label="Energy Forecast")
                    
                with gr.Column(scale=1):
                    risk_output = gr.HTML(label="Spike Risk")
            
            with gr.Row():
                details_output = gr.HTML(label="Prediction Details")
            
            # Risk classification guide
            with gr.Accordion("📖 Spike Risk Classification Guide", open=False):
                gr.Markdown(
                    """
                    ### 🔴 High-Risk Windows (Extreme volatility, large jumps common)
                    - **Friday 08-16:** Highest daytime intensity among weekdays (10-12% changes ≥200 Wh)
                    - **Saturday 08-16:** Most volatile period overall, event-driven usage, presence of >1050 Wh spikes
                    - **Monday 12-20:** Peak appliance usage window (up to 600 Wh), 33.3% of changes in 400-600 Wh range
                    - **Thursday 16-20:** Widest spread including rare 900-1200 Wh spikes
                    
                    ### 🟡 Moderate-Risk Windows (Structured volatility, predictable patterns)
                    - **Tuesday 16-20:** ~17% changes ≥100 Wh, controlled mid-to-high usage (300-600 Wh)
                    - **Wednesday 08-12 & 16-20:** Highest mid-to-large fluctuations among weekdays (150-450 Wh)
                    - **Friday 16-20:** Active but not extreme, sustained elevated usage
                    - **Sunday 08-20:** Sustained variability, moderate usage (150-450 Wh), smoother than Saturday
                    
                    ### 🟢 Low-Risk Windows (Stable, predictable)
                    - **All days 00-08:** ≈98% usage ≤25 Wh baseline (night hours)
                    - **Monday 20-24:** Sharp drop back to low usage after peak
                    - **Wednesday 00-04:** Most stable early morning, zero activity beyond baseline
                    - **Sunday evenings:** Settle earlier, no extreme spikes
                    - **Tuesday/Thursday 00-08:** Stable morning periods
                    """
                )
            
            # Connect prediction function
            predict_btn.click(
                fn=self.predict_energy,
                inputs=[date_input, time_input],
                outputs=[forecast_output, risk_output, details_output]
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                <div style="text-align: center; color: #666; font-size: 12px;">
                    Built with XGBoost, Gradio, and 49 engineered features | 
                    Risk classification based on comprehensive EDA patterns
                </div>
                """
            )
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Arguments to pass to gr.Blocks.launch()
        """
        interface = self.create_interface()
        interface.launch(**kwargs)


def main():
    """
    Main function to launch the application.
    """
    # Configuration
    data_paths = DataPaths()
    model_paths = ModelPaths()
    MODEL_PATH = model_paths.xgboost_pickle
    TEST_DATA_PATH = data_paths.engineered_features
    FEATURE_NAMES_PATH = data_paths.processed_dir / "feature_names.txt"   
    
    print("="*80)
    print("⚡ Appliance Energy Prediction Dashboard")
    print("="*80)
    print("\nInitializing application...")
    
    # Create and launch app
    app = EnergyPredictionApp(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        feature_names_path=FEATURE_NAMES_PATH
    )
    
    print("\n" + "="*80)
    print("✓ Application ready! Launching Gradio interface...")
    print("="*80)
    
    # Launch with public sharing (set share=False for local only)
    app.launch(
        share=True,  # Set to False for local development
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        show_error=True
    )


if __name__ == "__main__":
    main()