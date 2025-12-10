"""
Utility functions for the project.
"""
import os
import json
import yaml
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

def create_project_structure():
    """Create the project directory structure."""
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'scripts',
        'reports/figures',
        'models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("Project structure created successfully!")

def save_results(results, filename):
    """Save model results to file (JSON if possible, else pickle)."""
    # Ensure parent directory exists
    parent = os.path.dirname(filename) or '.'
    os.makedirs(parent, exist_ok=True)

    # Convert numpy/pandas/datetime objects for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            try:
                return obj.to_dict(orient='records') if isinstance(obj, pd.DataFrame) else obj.tolist()
            except Exception:
                return obj.tolist()
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        # numpy datetime
        try:
            if isinstance(obj, np.datetime64):
                return pd.to_datetime(obj).isoformat()
        except Exception:
            pass
        # fallback for other objects with tolist()
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()
            except Exception:
                pass
        # last resort: string
        try:
            return str(obj)
        except Exception:
            return None

    # Try JSON first
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, default=convert_for_json, indent=2)
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        # Fallback: pickle
        try:
            pfile = f"{filename}.pkl"
            with open(pfile, 'wb') as f:
                pickle.dump(results, f)
            print(f"JSON save failed ({e}). Pickle saved to {pfile}")
            return True
        except Exception as e2:
            print(f"Failed to save results: {e2}")
            return False

def load_config(config_file='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging():
    """Setup basic logging for the project."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/project_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)