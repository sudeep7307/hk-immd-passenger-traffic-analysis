"""
Data preprocessing module for HK immigration passenger traffic data.
Updated for the actual dataset format.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from typing import Tuple, List

class DataPreprocessor:
    """Preprocess HK immigration passenger traffic data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data file
        """
        self.data_path = data_path
        self.hk_holidays = holidays.HK()
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV with proper parsing."""
        print("Loading data...")
        
        # Load with correct date format
        self.df = pd.read_csv(
            self.data_path, 
            parse_dates=['Date'],
            dayfirst=True  # Important: date format is DD-MM-YYYY
        )
        
        # Rename columns for easier access
        self.df.columns = [
            'date', 'control_point', 'arrival_departure', 
            'hk_residents', 'mainland_visitors', 'other_visitors', 
            'total'
        ]
        
        print(f"Data shape: {self.df.shape}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Control Points: {self.df['control_point'].unique()}")
        
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the loaded data."""
        print("Cleaning data...")
        
        # Check for missing values
        print(f"Missing values before cleaning:")
        print(self.df.isnull().sum())
        
        # Handle zeros in total (some entries might be legitimately zero)
        # Keep them for now, but we'll analyze separately
        
        # Remove rows where all passenger counts are NaN
        passenger_cols = ['hk_residents', 'mainland_visitors', 'other_visitors', 'total']
        self.df.dropna(subset=passenger_cols, how='all', inplace=True)
        
        # Fill any remaining NaN in passenger counts with 0
        self.df[passenger_cols] = self.df[passenger_cols].fillna(0)
        
        # Ensure numeric types
        for col in passenger_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Data shape after cleaning: {self.df.shape}")
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """Create new features from the data."""
        print("Engineering features...")
        
        # Temporal features
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.dayofweek  # Monday=0, Sunday=6
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        
        # Holiday features
        self.df['is_holiday'] = self.df['date'].apply(
            lambda x: x in self.hk_holidays
        ).astype(int)
        
        # COVID-19 period indicator (adjust based on your data range)
        # Assuming 2021 was during COVID restrictions
        self.df['is_covid_period'] = (self.df['year'] == 2021).astype(int)
        
        # Create combined features
        self.df['hk_vs_visitors_ratio'] = np.where(
            (self.df['mainland_visitors'] + self.df['other_visitors']) > 0,
            self.df['hk_residents'] / (self.df['mainland_visitors'] + self.df['other_visitors']),
            0
        )
        
        # Arrival vs Departure
        self.df['is_arrival'] = (self.df['arrival_departure'] == 'Arrival').astype(int)
        
        # Control point features (one-hot encoding ready)
        self.df['is_airport'] = (self.df['control_point'] == 'Airport').astype(int)
        self.df['is_land_crossing'] = self.df['control_point'].isin([
            'Lo Wu', 'Lok Ma Chau', 'Shenzhen Bay', 'Man Kam To'
        ]).astype(int)
        
        # Create daily aggregates (for overall daily traffic analysis)
        daily_data = self.df.groupby('date').agg({
            'total': 'sum',
            'hk_residents': 'sum',
            'mainland_visitors': 'sum',
            'other_visitors': 'sum'
        }).reset_index()
        
        # Add features to daily data
        daily_data['year'] = daily_data['date'].dt.year
        daily_data['month'] = daily_data['date'].dt.month
        daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
        daily_data['is_weekend'] = (daily_data['day_of_week'] >= 5).astype(int)
        
        # Rolling statistics for daily data
        daily_data['rolling_7day_mean'] = daily_data['total'].rolling(window=7, min_periods=1).mean()
        daily_data['rolling_30day_mean'] = daily_data['total'].rolling(window=30, min_periods=1).mean()
        
        self.daily_data = daily_data
        
        return self.df
    
    def prepare_for_ml(self, analysis_type: str = 'daily_total') -> Tuple[pd.DataFrame, List[str], str]:
        """
        Prepare data for machine learning.
        
        Parameters:
        -----------
        analysis_type : str
            Options: 'daily_total', 'by_control_point', 'arrival_vs_departure'
        """
        print(f"Preparing for ML analysis: {analysis_type}...")
        
        if analysis_type == 'daily_total':
            # Use aggregated daily data
            df = self.daily_data.copy()
            target = 'total'
            
            # Features for daily prediction
            features = [
                'year', 'month', 'day_of_week', 'is_weekend',
                'rolling_7day_mean', 'rolling_30day_mean'
            ]
            
        elif analysis_type == 'by_control_point':
            # Use granular data by control point
            df = self.df.copy()
            target = 'total'
            
            # Filter for specific control point or use all
            df = df[df['control_point'] == 'Airport']  # Example: Airport only
            
            features = [
                'year', 'month', 'day_of_week', 'is_weekend',
                'is_arrival', 'is_holiday'
            ]
            
        elif analysis_type == 'arrival_vs_departure':
            # Compare arrival vs departure patterns
            df = self.df.copy()
            target = 'total'
            
            features = [
                'year', 'month', 'day_of_week', 'is_weekend',
                'is_holiday', 'is_arrival', 'is_airport', 'is_land_crossing'
            ]
        
        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")
        
        # Create classification target (high vs low traffic)
        threshold = df[target].quantile(0.75)
        df['traffic_level'] = (df[target] > threshold).astype(int)
        
        # Keep only features that exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        
        print(f"Available features: {available_features}")
        print(f"Target variable: {target}")
        print(f"Data shape: {df.shape}")
        
        return df, available_features, target
    
    def analyze_data_distribution(self):
        """Analyze basic statistics of the data."""
        print("\n=== Data Distribution Analysis ===")
        
        # Basic statistics
        print(f"\nTotal records: {len(self.df)}")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        # Control point distribution
        print("\nControl Point Distribution:")
        cp_dist = self.df['control_point'].value_counts()
        print(cp_dist)
        
        # Arrival vs Departure
        print("\nArrival vs Departure:")
        ad_dist = self.df['arrival_departure'].value_counts()
        print(ad_dist)
        
        # Passenger statistics
        print("\nPassenger Statistics:")
        passenger_stats = self.df[['hk_residents', 'mainland_visitors', 'other_visitors', 'total']].describe()
        print(passenger_stats)
        
        # Daily total statistics
        if hasattr(self, 'daily_data'):
            print("\nDaily Total Statistics:")
            daily_stats = self.daily_data['total'].describe()
            print(daily_stats)
            
            # Top 10 busiest days
            top_days = self.daily_data.nlargest(10, 'total')[['date', 'total']]
            print("\nTop 10 Busiest Days:")
            print(top_days)
        
        return self.df
    
    def run_pipeline(self, analysis_type: str = 'daily_total'):
        """Run the complete preprocessing pipeline."""
        print("=" * 50)
        print("Starting Data Preprocessing Pipeline")
        print("=" * 50)
        
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.analyze_data_distribution()
        
        df, features, target = self.prepare_for_ml(analysis_type)
        
        print("\n" + "=" * 50)
        print("Preprocessing Completed!")
        print("=" * 50)
        print(f"Final dataset shape: {df.shape}")
        print(f"Features for ML: {features}")
        print(f"Target variable: {target}")
        
        return df, features, target