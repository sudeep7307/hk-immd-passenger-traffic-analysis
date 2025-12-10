"""
Data preprocessing module for HK immigration passenger traffic data.
Provides a runnable CLI to produce processed CSVs under data/processed/.
"""
from typing import Tuple, List
import os
import sys
import argparse
import pandas as pd
import numpy as np
import holidays
from datetime import datetime

class DataPreprocessor:
    """Preprocess HK immigration passenger traffic data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor.
        data_path : str - Path to the raw data file
        """
        self.data_path = data_path
        self.hk_holidays = holidays.HK()
        self.df = pd.DataFrame()
        self.daily_data = pd.DataFrame()
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV with robust parsing and header normalization."""
        print("Loading data from:", self.data_path)
        self.df = pd.read_csv(self.data_path, skipinitialspace=True, dtype=str)
        
        # Drop completely empty / unnamed columns (trailing commas create these)
        self.df.dropna(axis=1, how='all', inplace=True)
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', na=False)]
        
        # Normalize header names
        col_map = {c: c.strip().lower() for c in self.df.columns}
        self.df.rename(columns=col_map, inplace=True)
        
        # Map to canonical internal names
        name_map = {
            'date': 'date',
            'control point': 'control_point',
            'control_point': 'control_point',
            'arrival / departure': 'arrival_departure',
            'arrival/departure': 'arrival_departure',
            'arrival_departure': 'arrival_departure',
            'hong kong residents': 'hk_residents',
            'hong kong resident': 'hk_residents',
            'hk residents': 'hk_residents',
            'mainland visitors': 'mainland_visitors',
            'other visitors': 'other_visitors',
            'total': 'total'
        }
        rename_map = {}
        for c in self.df.columns:
            k = c.strip().lower()
            if k in name_map:
                rename_map[c] = name_map[k]
        # fallback positional mapping if header names are off
        if not rename_map and self.df.shape[1] >= 7:
            rename_map = {
                self.df.columns[0]: 'date',
                self.df.columns[1]: 'control_point',
                self.df.columns[2]: 'arrival_departure',
                self.df.columns[3]: 'hk_residents',
                self.df.columns[4]: 'mainland_visitors',
                self.df.columns[5]: 'other_visitors',
                self.df.columns[6]: 'total'
            }
        self.df.rename(columns=rename_map, inplace=True)
        
        # find date column
        date_col = None
        for c in self.df.columns:
            if 'date' == c.lower() or 'date' in c.lower():
                date_col = c
                break
        if date_col is None:
            raise ValueError("Date column not found in CSV")
        # parse date with dayfirst (your CSV uses DD-MM-YYYY)
        self.df[date_col] = pd.to_datetime(self.df[date_col], dayfirst=True, errors='coerce')
        self.df.rename(columns={date_col: 'date'}, inplace=True)
        
        # canonicalize control_point text
        if 'control_point' in self.df.columns:
            self.df['control_point'] = self.df['control_point'].astype(str).str.strip()
        
        print("Loaded rows:", len(self.df))
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean types and numeric columns."""
        if self.df is None or self.df.empty:
            raise RuntimeError("Data not loaded")
        # drop rows without date
        self.df.dropna(subset=['date'], inplace=True)
        # convert numeric columns
        for col in ['hk_residents', 'mainland_visitors', 'other_visitors', 'total']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col].replace('', np.nan), errors='coerce').fillna(0).astype(int)
        # normalize arrival_departure
        if 'arrival_departure' in self.df.columns:
            self.df['arrival_departure'] = self.df['arrival_departure'].astype(str).str.strip().str.title()
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """Add date-derived features and produce daily aggregated data."""
        df = self.df
        df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        # is holiday
        df['is_holiday'] = df['date'].dt.date.apply(lambda d: 1 if d in self.hk_holidays else 0)
        
        # create daily aggregate (sum totals across control points and arrival/departure)
        agg_cols = [c for c in ['total','hk_residents','mainland_visitors','other_visitors'] if c in df.columns]
        if 'total' in agg_cols:
            daily = df.groupby('date', as_index=False)[agg_cols].sum()
            # merge back a representative day_of_week/month/is_holiday (first)
            meta = df.groupby('date', as_index=False).agg({'day_of_week':'first','month':'first','is_weekend':'first','is_holiday':'first'})
            daily = daily.merge(meta, on='date', how='left')
            self.daily_data = daily.sort_values('date').reset_index(drop=True)
        else:
            self.daily_data = df.copy()
        return self.daily_data
    
    def prepare_for_ml(self, analysis_type: str = 'daily_total') -> Tuple[pd.DataFrame, List[str], str]:
        """Return ML-ready dataframe, feature list and target."""
        if analysis_type == 'daily_total':
            df_ml = self.daily_data.copy()
            features = [f for f in ['day_of_week','month','is_weekend','is_holiday'] if f in df_ml.columns]
            target = 'total'
        elif analysis_type == 'by_control_point':
            df_ml = self.df.copy()
            features = [f for f in ['control_point','arrival_departure','day_of_week','month'] if f in df_ml.columns]
            # convert categorical control_point to code for ML caller
            if 'control_point' in df_ml.columns:
                df_ml['control_point'] = df_ml['control_point'].astype('category').cat.codes
            target = 'total' if 'total' in df_ml.columns else (df_ml.columns[-1])
        else:
            df_ml = self.daily_data.copy()
            features = [f for f in ['day_of_week','month'] if f in df_ml.columns]
            target = 'total'
        return df_ml, features, target
    
    def analyze_data_distribution(self):
        """Print simple diagnostics."""
        print("Data diagnostics:")
        print(" - Rows (raw):", len(self.df))
        if not self.df.empty:
            print(self.df[['date']].drop_duplicates().shape[0], "unique dates")
        if hasattr(self, 'daily_data') and not self.daily_data.empty:
            print(" - Daily data rows:", len(self.daily_data))
            if 'total' in self.daily_data.columns:
                print(" - total stats:", self.daily_data['total'].describe().to_dict())
        nulls = self.df.isnull().sum()
        print(" - Null counts (raw):")
        print(nulls[nulls>0].to_dict())
    
    def run_pipeline(self, analysis_type: str = 'daily_total', save_processed: bool = True, processed_dir: str = 'data/processed'):
        """Run full pipeline and optionally save outputs to processed_dir."""
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.analyze_data_distribution()
        
        df_ml, features, target = self.prepare_for_ml(analysis_type)
        
        if save_processed:
            os.makedirs(processed_dir, exist_ok=True)
            granular_path = os.path.join(processed_dir, 'passenger_data_granular.csv')
            daily_path = os.path.join(processed_dir, 'passenger_data_daily.csv')
            try:
                self.df.to_csv(granular_path, index=False)
                print("Saved granular processed file:", granular_path)
            except Exception as e:
                print("Failed to save granular CSV:", e)
            try:
                self.daily_data.to_csv(daily_path, index=False)
                print("Saved daily aggregated file:", daily_path)
            except Exception as e:
                print("Failed to save daily CSV:", e)
        
        return df_ml, features, target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing for HK passenger traffic")
    parser.add_argument("--input","-i", default=os.path.join("data","raw","statistics_on_daily_passenger_traffic.csv"), help="Path to raw CSV")
    parser.add_argument("--processed-dir","-o", default=os.path.join("data","processed"), help="Directory to write processed outputs")
    parser.add_argument("--analysis","-a", default="daily_total", choices=["daily_total","by_control_point","arrival_vs_departure"], help="Analysis type")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print("Input file not found:", args.input)
        sys.exit(1)
    
    dp = DataPreprocessor(args.input)
    df_ml, features, target = dp.run_pipeline(analysis_type=args.analysis, save_processed=True, processed_dir=args.processed_dir)
    print("Done. Processed files written to:", args.processed_dir)
    print("ML dataframe sample:")
    print(df_ml.head().to_string(index=False))