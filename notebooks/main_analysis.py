# %% [markdown]
# # Hong Kong Immigration Passenger Traffic Analysis
# 
# ## 1. Project Overview
# 
# ### 1.1 Introduction
# This project analyzes daily passenger traffic at Hong Kong immigration checkpoints using various machine learning algorithms. The goal is to predict traffic patterns, classify high-traffic days, and identify clusters for operational planning.
# 
# ### 1.2 Objectives
# - Predict future passenger traffic using Linear Regression
# - Classify days as high/low traffic using Logistic Regression and SVM
# - Cluster similar traffic patterns using K-means
# - Analyze differences between control points
# - Provide actionable insights for immigration authorities
# 
# ### 1.3 Project Structure
# - Data loading and preprocessing
# - Exploratory data analysis (EDA)
# - Feature engineering
# - Machine learning modeling
# - Model evaluation and comparison
# - Visualization and reporting
# 
# ## 2. Data Loading and Exploration

# %% [markdown]
# ### 2.1 Import Libraries

# %%
# Core libraries
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Data manipulation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    silhouette_score
)

# Custom modules
# Add parent directory to path to import custom modules
sys.path.append('..')

try:
    from data_preprocessing import DataPreprocessor
    from models import TrafficModels
    from visualization import TrafficVisualizer
    print("✅ Custom modules imported successfully!")
except ImportError as e:
    print(f"⚠️  Some custom modules not available: {e}")
    print("   Continuing with standard libraries...")

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ All libraries imported successfully!")

# %% [markdown]
# ### 2.2 Load and Explore Dataset

# %%
def load_data(data_path="data/raw/passenger_traffic.csv"):
    """
    Load the passenger traffic dataset.
    Returns DataFrame and metadata about the dataset.
    """
    print(f"📊 Attempting to load data from: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"⚠️  File not found: {data_path}")
        
        # Check for processed data
        processed_path = "data/processed/passenger_data_daily.csv"
        if os.path.exists(processed_path):
            print(f"📂 Found processed data at: {processed_path}")
            data_path = processed_path
        else:
            print("❌ No data files found. Creating sample data for demonstration...")
            return create_sample_data()
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"✅ Data loaded successfully with {encoding} encoding!")
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(data_path, encoding='utf-8', errors='ignore')
            print("⚠️  Loaded with errors, check data quality")
        
        # Basic info
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "   No date column found")
        
        # Display sample
        print("\n📋 First 5 rows:")
        display(df.head())
        
        # Display info
        print("\n📊 Data Info:")
        print(df.info())
        
        # Check for missing values
        print("\n🔍 Missing Values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
        display(missing_df[missing_df['Missing Count'] > 0])
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes."""
    print("🛠️ Creating sample data...")
    
    # Generate dates for a year
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Define control points (immigration checkpoints)
    control_points = ['Airport', 'Lo Wu', 'Lok Ma Chau', 'Shenzhen Bay', 'Hong Kong-Macau Ferry', 'Man Kam To']
    
    # Generate sample data
    np.random.seed(42)
    
    data = []
    for date in dates:
        for cp in control_points:
            # Base traffic with seasonality
            base_traffic = np.random.normal(50000, 15000)
            
            # Weekend effect
            weekend_boost = 30000 if date.dayofweek >= 5 else 0
            
            # Monthly seasonality (higher in holidays)
            month_boost = 40000 if date.month in [7, 8, 12] else 0
            
            # Control point specific patterns
            cp_multiplier = {
                'Airport': 1.5,
                'Lo Wu': 2.0,
                'Lok Ma Chau': 1.8,
                'Shenzhen Bay': 1.3,
                'Hong Kong-Macau Ferry': 1.0,
                'Man Kam To': 0.8
            }
            
            total = (base_traffic + weekend_boost + month_boost) * cp_multiplier[cp]
            
            # Split by passenger type
            hk_residents = total * np.random.uniform(0.4, 0.6)
            mainland_visitors = total * np.random.uniform(0.3, 0.5)
            other_visitors = total - hk_residents - mainland_visitors
            
            data.append({
                'date': date,
                'control_point': cp,
                'total': int(total),
                'hk_residents': int(hk_residents),
                'mainland_visitors': int(mainland_visitors),
                'other_visitors': int(other_visitors),
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
                'is_holiday': 1 if date.month in [1, 7, 8, 12] and date.day in [1, 15] else 0
            })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Sample data created with {len(df)} records")
    print(f"   Control points: {control_points}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save sample data for future use
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/sample_passenger_data.csv', index=False)
    print("💾 Sample data saved to 'data/processed/sample_passenger_data.csv'")
    
    return df

# Load the data
df = load_data()

# %% [markdown]
# ## 3. Data Preprocessing and Feature Engineering

# %%
def preprocess_data(df):
    """
    Preprocess the passenger traffic data.
    """
    print("🔄 Preprocessing data...")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # 1. Convert date column to datetime
    if 'date' in df_processed.columns:
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        print("✅ Converted 'date' column to datetime")
    
    # 2. Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            print(f"✅ Filled missing values in '{col}' with median")
    
    # 3. Add temporal features
    if 'date' in df_processed.columns:
        # Basic temporal features
        df_processed['year'] = df_processed['date'].dt.year
        df_processed['month'] = df_processed['date'].dt.month
        df_processed['day'] = df_processed['date'].dt.day
        df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
        df_processed['week_of_year'] = df_processed['date'].dt.isocalendar().week
        df_processed['quarter'] = df_processed['date'].dt.quarter
        
        # Derived features
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        df_processed['is_month_start'] = df_processed['date'].dt.is_month_start.astype(int)
        df_processed['is_month_end'] = df_processed['date'].dt.is_month_end.astype(int)
        
        # Holiday seasons (simplified)
        df_processed['is_holiday_season'] = df_processed['month'].isin([1, 7, 8, 12]).astype(int)
        
        # Traffic ratios
        if all(col in df_processed.columns for col in ['hk_residents', 'mainland_visitors', 'other_visitors', 'total']):
            df_processed['mainland_ratio'] = df_processed['mainland_visitors'] / df_processed['total']
            df_processed['residents_ratio'] = df_processed['hk_residents'] / df_processed['total']
            df_processed['other_ratio'] = df_processed['other_visitors'] / df_processed['total']
        
        print("✅ Added temporal and derived features")
    
    # 4. Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].nunique() <= 10:  # Low cardinality
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            print(f"✅ Encoded '{col}' with {df_processed[col].nunique()} categories")
        else:
            print(f"⚠️  High cardinality column '{col}' ({df_processed[col].nunique()} categories) - consider one-hot encoding")
    
    # 5. Remove outliers (optional)
    # Using IQR method for numeric columns
    for col in numeric_cols:
        if col not in ['year', 'month', 'day', 'day_of_week', 'is_weekend', 'is_holiday']:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"⚠️  Found {outliers} outliers in '{col}' ({outliers/len(df_processed)*100:.1f}%)")
                # Cap outliers instead of removing
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    print(f"✅ Preprocessing completed!")
    print(f"   Final shape: {df_processed.shape}")
    print(f"   New columns: {list(df_processed.columns)}")
    
    return df_processed

# Preprocess the data
df_processed = preprocess_data(df)

# Display processed data
print("\n📋 Processed Data Sample:")
display(df_processed.head())
print(f"\n📊 Data Types:")
print(df_processed.dtypes)

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)

# %%
def perform_eda(df):
    """
    Perform exploratory data analysis and create visualizations.
    """
    print("🔍 Performing Exploratory Data Analysis...")
    
    # Create visualizer
    viz = TrafficVisualizer(df)
    
    # 1. Summary statistics
    print("\n📈 Summary Statistics:")
    
    # Select numeric columns for summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'total' in numeric_cols:
        total_stats = df['total'].describe()
        print(f"\nTotal Passenger Statistics:")
        print(f"   Mean: {total_stats['mean']:,.0f}")
        print(f"   Median: {total_stats['50%']:,.0f}")
        print(f"   Std: {total_stats['std']:,.0f}")
        print(f"   Min: {total_stats['min']:,.0f}")
        print(f"   Max: {total_stats['max']:,.0f}")
        print(f"   IQR: {total_stats['75%'] - total_stats['25%']:,.0f}")
    
    # 2. Time series analysis
    if 'date' in df.columns and 'total' in df.columns:
        print("\n📅 Generating time series plot...")
        fig1 = viz.plot_time_series(
            date_col='date',
            target_col='total',
            title='Daily Passenger Traffic Over Time'
        )
        plt.show()
    
    # 3. Distribution analysis
    if 'total' in df.columns:
        print("\n📊 Generating distribution plots...")
        fig2 = viz.plot_distribution(
            target_col='total',
            title='Passenger Traffic Distribution'
        )
        plt.show()
    
    # 4. Correlation analysis
    if len(numeric_cols) > 1:
        print("\n🔗 Generating correlation heatmap...")
        fig3 = viz.plot_correlation_heatmap(
            columns=list(numeric_cols)[:10],  # Limit to first 10 for readability
            title='Feature Correlation Heatmap'
        )
        plt.show()
    
    # 5. Seasonality analysis
    if 'date' in df.columns and 'total' in df.columns:
        print("\n📅 Generating seasonality plots...")
        fig4 = viz.plot_seasonality(
            date_col='date',
            target_col='total',
            title='Traffic Seasonality Patterns'
        )
        plt.show()
    
    # 6. Control point analysis (if control_point column exists)
    if 'control_point' in df.columns and 'total' in df.columns:
        print("\n📍 Generating control point comparison...")
        fig5 = viz.plot_control_point_comparison(
            data=df,
            control_point_col='control_point',
            target_col='total',
            title='Control Point Traffic Comparison'
        )
        plt.show()
        
        # Control point time series
        print("\n⏰ Generating control point time series...")
        fig6 = viz.plot_control_point_time_series(
            data=df,
            control_point_col='control_point',
            target_col='total',
            time_col='date',
            title='Control Point Traffic Time Series'
        )
        plt.show()
    
    # 7. Traffic composition analysis
    if all(col in df.columns for col in ['hk_residents', 'mainland_visitors', 'other_visitors']):
        print("\n👥 Analyzing passenger composition...")
        
        # Calculate composition percentages
        composition = df[['hk_residents', 'mainland_visitors', 'other_visitors']].mean()
        total = composition.sum()
        
        print(f"\nAverage Daily Passenger Composition:")
        print(f"   Hong Kong Residents: {composition['hk_residents']:,.0f} ({composition['hk_residents']/total*100:.1f}%)")
        print(f"   Mainland Visitors: {composition['mainland_visitors']:,.0f} ({composition['mainland_visitors']/total*100:.1f}%)")
        print(f"   Other Visitors: {composition['other_visitors']:,.0f} ({composition['other_visitors']/total*100:.1f}%)")
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['HK Residents', 'Mainland Visitors', 'Other Visitors']
        sizes = [composition['hk_residents'], composition['mainland_visitors'], composition['other_visitors']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Average Passenger Composition')
        plt.show()
    
    print("\n✅ EDA completed!")

# Perform EDA
perform_eda(df_processed)

# %% [markdown]
# ## 5. Machine Learning Modeling

# %%
def prepare_modeling_data(df, target_col='total', problem_type='regression'):
    """
    Prepare data for machine learning modeling.
    """
    print(f"🛠️ Preparing data for {problem_type} modeling...")
    
    # Define features
    # Exclude date columns and target column from features
    exclude_cols = ['date', target_col]
    
    # Also exclude any encoded versions if we have the original
    for col in df.columns:
        if col.endswith('_encoded') and col.replace('_encoded', '') in df.columns:
            exclude_cols.append(col.replace('_encoded', ''))
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select only numeric columns for features
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"   Target column: {target_col}")
    print(f"   Selected {len(feature_cols)} features:")
    for i, feat in enumerate(feature_cols[:10]):  # Show first 10
        print(f"     {i+1}. {feat}")
    if len(feature_cols) > 10:
        print(f"     ... and {len(feature_cols) - 10} more")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # For classification, convert continuous target to binary
    if problem_type == 'classification':
        if y.nunique() > 10:  # Likely continuous, need to bin
            threshold = y.quantile(0.75)  # Top 25% as high traffic
            y = (y > threshold).astype(int)
            print(f"   Converted continuous target to binary (threshold: {threshold:,.0f})")
            print(f"   Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    return X, y, feature_cols

# Prepare data for regression
X_reg, y_reg, features_reg = prepare_modeling_data(df_processed, 'total', 'regression')

# Prepare data for classification
X_clf, y_clf, features_clf = prepare_modeling_data(df_processed, 'total', 'classification')

print(f"\n✅ Data preparation completed!")
print(f"   Regression data: X={X_reg.shape}, y={y_reg.shape}")
print(f"   Classification data: X={X_clf.shape}, y={y_clf.shape}")

# %% [markdown]
# ### 5.1 Using Custom TrafficModels Class

# %%
# Initialize TrafficModels for regression
print("🚀 Initializing TrafficModels for regression analysis...")
tm_reg = TrafficModels(df=df_processed, features=features_reg, target='total', problem_type='regression')

# Prepare data
data_reg = tm_reg.prepare_data(test_size=0.2, random_state=42, scale=True)

# Run linear regression
print("\n" + "="*60)
print("Running Linear Regression...")
print("="*60)
linear_results = tm_reg.linear_regression(show_plots=True, save_plots=True)

# %% [markdown]
# ### 5.2 Classification Analysis

# %%
# For classification, we need to create a binary target
df_classification = df_processed.copy()

# Create binary target (high traffic vs normal traffic)
threshold = df_classification['total'].quantile(0.75)  # Top 25% as high traffic
df_classification['traffic_level'] = (df_classification['total'] > threshold).astype(int)

print(f"📊 Classification Target Created:")
print(f"   Threshold: {threshold:,.0f} passengers")
print(f"   High Traffic Days: {df_classification['traffic_level'].sum()} ({(df_classification['traffic_level'].sum()/len(df_classification)*100):.1f}%)")
print(f"   Normal Traffic Days: {(df_classification['traffic_level']==0).sum()} ({((df_classification['traffic_level']==0).sum()/len(df_classification)*100):.1f}%)")

# Initialize TrafficModels for classification
print("\n🚀 Initializing TrafficModels for classification analysis...")
tm_clf = TrafficModels(df=df_classification, features=features_clf, target='traffic_level', problem_type='classification')

# Prepare data
data_clf = tm_clf.prepare_data(test_size=0.2, random_state=42, scale=True)

# Run logistic regression
print("\n" + "="*60)
print("Running Logistic Regression...")
print("="*60)
logistic_results = tm_clf.logistic_regression(show_plots=True, save_plots=True)

# Run SVM
print("\n" + "="*60)
print("Running SVM Classification...")
print("="*60)
svm_results = tm_clf.svm_classification(kernel='rbf', C=1.0, show_plots=True, save_plots=True)

# %% [markdown]
# ### 5.3 Clustering Analysis

# %%
print("🔍 Running Clustering Analysis...")

# Initialize TrafficModels for clustering (use regression mode)
tm_cluster = TrafficModels(df=df_processed, features=features_reg, target='total', problem_type='regression')

# Run K-means clustering
print("\n" + "="*60)
print("Running K-means Clustering...")
print("="*60)
kmeans_results = tm_cluster.kmeans_clustering(
    n_clusters=4,  # Try 4 clusters
    feature_subset=['total', 'hk_residents', 'mainland_visitors', 'day_of_week', 'month'],
    show_plots=True,
    save_plots=True
)

# Display cluster statistics
if kmeans_results:
    print("\n📊 Cluster Statistics:")
    for cluster, stats in kmeans_results['cluster_stats'].items():
        print(f"\nCluster {cluster}:")
        print(f"  Size: {stats['count']} samples ({stats['percentage']:.1f}%)")
        if not np.isnan(stats['mean_total']):
            print(f"  Avg Total Traffic: {stats['mean_total']:,.0f} ± {stats['std_total']:,.0f}")
        if not np.isnan(stats['avg_day_of_week']):
            print(f"  Avg Day of Week: {stats['avg_day_of_week']:.2f}")

# %% [markdown]
# ### 5.4 Control Point Comparison Analysis

# %%
# Check if control_point column exists
if 'control_point' in df_processed.columns:
    print("📍 Running Control Point Comparison Analysis...")
    
    # Use the TrafficModels instance we already have
    print("\n" + "="*60)
    print("Comparing Control Point Traffic Patterns...")
    print("="*60)
    
    cp_results = tm_reg.compare_control_points(
        control_point_col='control_point',
        target_col='total',
        time_col='date',
        show_plots=True,
        save_plots=True
    )
    
    if cp_results:
        print("\n🏆 Top Control Points by Traffic Volume:")
        comparison_df = cp_results['comparison_df']
        display(comparison_df.head(10))
        
        # Display insights
        print("\n💡 Key Insights:")
        print(f"   Total control points analyzed: {cp_results['total_control_points']}")
        print(f"   Busiest control point: {cp_results['top_control_point']}")
        print(f"   Quietest control point: {cp_results['bottom_control_point']}")
        print(f"   Traffic ratio (busiest/quietest): {cp_results['traffic_ratio']:.1f}x")
else:
    print("⚠️  'control_point' column not found. Skipping control point analysis.")

# %% [markdown]
# ### 5.5 Model Comparison

# %%
print("📊 Comparing All Models...")

# Compare all models
comparison_results = tm_reg.compare_models(show_plots=True)

# Display comparison table
if comparison_results is not None:
    print("\n🏆 Model Performance Comparison:")
    display(comparison_results)
    
    # Determine best model for each task
    print("\n🎯 Best Performing Models:")
    
    # Regression models
    reg_models = comparison_results[comparison_results['Type'] == 'Regression']
    if not reg_models.empty:
        best_reg = reg_models.loc[reg_models['R²'].idxmax()]
        print(f"   Regression: {best_reg.name} (R²: {best_reg['R²']:.4f})")
    
    # Classification models
    clf_models = comparison_results[comparison_results['Type'] == 'Classification']
    if not clf_models.empty:
        best_clf = clf_models.loc[clf_models['Accuracy'].idxmax()]
        print(f"   Classification: {best_clf.name} (Accuracy: {best_clf['Accuracy']:.4f})")
    
    # Clustering models
    cluster_models = comparison_results[comparison_results['Type'] == 'Clustering']
    if not cluster_models.empty:
        # For clustering, higher silhouette is better
        best_cluster = cluster_models.loc[cluster_models['Silhouette Score'].idxmax()]
        print(f"   Clustering: {best_cluster.name} (Silhouette: {best_cluster['Silhouette Score']:.4f})")

# %% [markdown]
# ## 6. Comprehensive Pipeline

# %%
def run_comprehensive_pipeline(df):
    """
    Run the complete machine learning pipeline.
    """
    print("🚀 Starting Comprehensive Machine Learning Pipeline")
    print("="*70)
    
    # Define features for modeling
    exclude_cols = ['date', 'traffic_level']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"📊 Using {len(feature_cols)} features for modeling")
    print(f"🎯 Target: 'total' (regression) and derived 'traffic_level' (classification)")
    
    # Initialize TrafficModels
    tm = TrafficModels(df=df, features=feature_cols, target='total', problem_type='regression')
    
    # Run complete pipeline
    pipeline_results = tm.run_all_models(
        regression=True,
        classification=True,
        clustering=True,
        control_point_analysis='control_point' in df.columns,
        show_plots=True,
        save_plots=True
    )
    
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE PIPELINE COMPLETED!")
    print("="*70)
    
    # Save results
    save_pipeline_results(pipeline_results)
    
    return pipeline_results

def save_pipeline_results(results):
    """Save pipeline results to files."""
    import json
    import joblib
    from datetime import datetime
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results summary
    summary = {
        'timestamp': timestamp,
        'models_trained': list(results.get('results', {}).keys()),
        'comparison_available': results.get('comparison') is not None
    }
    
    # Save summary as JSON
    summary_path = f'{results_dir}/pipeline_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save comparison DataFrame as CSV
    if results.get('comparison') is not None:
        comparison_path = f'{results_dir}/model_comparison_{timestamp}.csv'
        results['comparison'].to_csv(comparison_path)
    
    # Save models (if available in tm instance)
    models_dir = f'{results_dir}/models_{timestamp}'
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"💾 Results saved to '{results_dir}/'")
    print(f"   Summary: {summary_path}")
    if results.get('comparison') is not None:
        print(f"   Comparison: {comparison_path}")
    print(f"   Models directory: {models_dir}")
    
    return summary_path

# Run comprehensive pipeline
print("🎯 Running Complete Analysis Pipeline...")
pipeline_results = run_comprehensive_pipeline(df_processed)

# %% [markdown]
# ## 7. Results Summary and Insights

# %%
def generate_insights(df, model_results):
    """
    Generate actionable insights from the analysis.
    """
    print("💡 Generating Insights and Recommendations...")
    print("="*60)
    
    insights = []
    
    # 1. Traffic patterns
    if 'total' in df.columns:
        avg_traffic = df['total'].mean()
        max_traffic = df['total'].max()
        min_traffic = df['total'].min()
        
        insights.append(f"📊 Traffic Patterns:")
        insights.append(f"   • Average daily traffic: {avg_traffic:,.0f} passengers")
        insights.append(f"   • Peak traffic: {max_traffic:,.0f} passengers")
        insights.append(f"   • Minimum traffic: {min_traffic:,.0f} passengers")
        insights.append(f"   • Daily variation: ±{(df['total'].std()/avg_traffic*100):.1f}%")
    
    # 2. Weekly patterns
    if 'day_of_week' in df.columns and 'total' in df.columns:
        weekly_patterns = df.groupby('day_of_week')['total'].mean()
        busiest_day = weekly_patterns.idxmax()
        quietest_day = weekly_patterns.idxmin()
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        insights.append(f"\n📅 Weekly Patterns:")
        insights.append(f"   • Busiest day: {day_names[busiest_day]} ({weekly_patterns[busiest_day]:,.0f} passengers)")
        insights.append(f"   • Quietest day: {day_names[quietest_day]} ({weekly_patterns[quietest_day]:,.0f} passengers)")
        insights.append(f"   • Weekend vs weekday ratio: {(weekly_patterns[[5,6]].mean()/weekly_patterns[:5].mean()):.2f}x")
    
    # 3. Monthly seasonality
    if 'month' in df.columns and 'total' in df.columns:
        monthly_patterns = df.groupby('month')['total'].mean()
        busiest_month = monthly_patterns.idxmax()
        quietest_month = monthly_patterns.idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        insights.append(f"\n🌤️ Monthly Seasonality:")
        insights.append(f"   • Peak month: {month_names[busiest_month-1]} ({monthly_patterns[busiest_month]:,.0f} passengers)")
        insights.append(f"   • Low season: {month_names[quietest_month-1]} ({monthly_patterns[quietest_month]:,.0f} passengers)")
        insights.append(f"   • Seasonal variation: {(monthly_patterns.max()/monthly_patterns.min()):.1f}x")
    
    # 4. Model performance insights
    if model_results and 'comparison' in model_results and model_results['comparison'] is not None:
        comp_df = model_results['comparison']
        
        # Regression insights
        reg_models = comp_df[comp_df['Type'] == 'Regression']
        if not reg_models.empty:
            best_reg = reg_models.loc[reg_models['R²'].idxmax()]
            insights.append(f"\n🎯 Model Performance:")
            insights.append(f"   • Best regression model: {best_reg.name} (R²: {best_reg['R²']:.3f})")
            insights.append(f"   • Prediction accuracy: ±{np.sqrt(reg_models['RMSE'].min()):,.0f} passengers")
        
        # Classification insights
        clf_models = comp_df[comp_df['Type'] == 'Classification']
        if not clf_models.empty:
            best_clf = clf_models.loc[clf_models['Accuracy'].idxmax()]
            insights.append(f"   • Best classification model: {best_clf.name} (Accuracy: {best_clf['Accuracy']:.1%})")
    
    # 5. Control point insights (if available)
    if 'control_point' in df.columns:
        cp_stats = df.groupby('control_point')['total'].agg(['mean', 'std', 'count'])
        cp_stats = cp_stats.sort_values('mean', ascending=False)
        
        insights.append(f"\n📍 Control Point Analysis:")
        insights.append(f"   • Busiest checkpoint: {cp_stats.index[0]} ({cp_stats.iloc[0]['mean']:,.0f} passengers/day)")
        insights.append(f"   • Total checkpoints analyzed: {len(cp_stats)}")
        
        if len(cp_stats) > 1:
            traffic_ratio = cp_stats.iloc[0]['mean'] / cp_stats.iloc[-1]['mean']
            insights.append(f"   • Busiest/quietest ratio: {traffic_ratio:.1f}x")
    
    # 6. Operational recommendations
    insights.append(f"\n🚀 Operational Recommendations:")
    insights.append(f"   1. Staff allocation: Increase resources on weekends and peak months")
    insights.append(f"   2. Infrastructure: Focus on busiest control points for upgrades")
    insights.append(f"   3. Prediction: Use {best_reg.name if 'best_reg' in locals() else 'regression models'} for traffic forecasting")
    insights.append(f"   4. Monitoring: Implement real-time classification for high-traffic alerts")
    
    # Print all insights
    for insight in insights:
        print(insight)
    
    # Save insights to file
    insights_dir = 'reports'
    os.makedirs(insights_dir, exist_ok=True)
    
    insights_path = f'{insights_dir}/insights_summary.txt'
    with open(insights_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('HONG KONG IMMIGRATION TRAFFIC ANALYSIS - INSIGHTS\n')
        f.write('='*60 + '\n\n')
        for insight in insights:
            f.write(insight + '\n')
    
    print(f"\n💾 Insights saved to: {insights_path}")
    
    return insights

# Generate insights
print("\n" + "="*60)
insights = generate_insights(df_processed, pipeline_results)

# %% [markdown]
# ## 8. Export Results and Create Final Report

# %%
def create_final_report(df, pipeline_results, insights):
    """
    Create a comprehensive final report.
    """
    print("📋 Creating Final Report...")
    
    from datetime import datetime
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Create reports directory
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'{reports_dir}/final_report_{timestamp}.pdf'
    
    # Create PDF report
    with PdfPages(report_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        title_text = "Hong Kong Immigration Passenger Traffic Analysis\nFinal Report"
        ax.text(0.5, 0.7, title_text, ha='center', va='center', fontsize=20, fontweight='bold')
        
        info_text = (
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Data period: {df['date'].min().date()} to {df['date'].max().date()}\n"
            f"Total records: {len(df):,}\n"
            f"Author: Luk Ka Chun\n"
            f"Course: DIT 5412 Data Science, THEi FDE"
        )
        ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Executive Summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        summary_text = (
            "EXECUTIVE SUMMARY\n\n"
            "This report presents a comprehensive analysis of passenger traffic at "
            "Hong Kong immigration checkpoints using machine learning techniques.\n\n"
            "Key Findings:\n"
            "1. Significant seasonal and weekly patterns detected\n"
            "2. Machine learning models accurately predict traffic patterns\n"
            "3. Clear differences between control points identified\n"
            "4. Actionable insights for operational planning provided\n\n"
            "The analysis employs regression, classification, and clustering "
            "algorithms to understand traffic patterns and provide predictive "
            "capabilities for resource planning."
        )
        ax.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Data Overview
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        data_text = (
            "DATA OVERVIEW\n\n"
            f"Total Records: {len(df):,}\n"
            f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}\n"
            f"Control Points Analyzed: {df['control_point'].nunique() if 'control_point' in df.columns else 'N/A'}\n\n"
            "Variables Analyzed:\n"
        )
        
        # List key variables
        key_vars = ['total', 'hk_residents', 'mainland_visitors', 'other_visitors', 
                   'day_of_week', 'month', 'is_weekend']
        for var in key_vars:
            if var in df.columns:
                if pd.api.types.is_numeric_dtype(df[var]):
                    data_text += f"• {var}: Mean={df[var].mean():,.0f}, Std={df[var].std():,.0f}\n"
                else:
                    data_text += f"• {var}: {df[var].nunique()} unique values\n"
        
        ax.text(0.1, 0.9, data_text, ha='left', va='top', fontsize=12)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Model Results
        if pipeline_results and 'comparison' in pipeline_results:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            model_text = "MODEL PERFORMANCE SUMMARY\n\n"
            
            comp_df = pipeline_results['comparison']
            
            # Regression models
            reg_models = comp_df[comp_df['Type'] == 'Regression']
            if not reg_models.empty:
                model_text += "REGRESSION MODELS:\n"
                for idx, row in reg_models.iterrows():
                    model_text += f"• {idx}: R²={row['R²']:.3f}, RMSE={row['RMSE']:,.0f}\n"
                model_text += "\n"
            
            # Classification models
            clf_models = comp_df[comp_df['Type'] == 'Classification']
            if not clf_models.empty:
                model_text += "CLASSIFICATION MODELS:\n"
                for idx, row in clf_models.iterrows():
                    model_text += f"• {idx}: Accuracy={row['Accuracy']:.3f}, F1={row.get('F1-Score', 'N/A')}\n"
                model_text += "\n"
            
            # Clustering models
            cluster_models = comp_df[comp_df['Type'] == 'Clustering']
            if not cluster_models.empty:
                model_text += "CLUSTERING MODELS:\n"
                for idx, row in cluster_models.iterrows():
                    model_text += f"• {idx}: Silhouette={row.get('Silhouette Score', 'N/A'):.3f}\n"
            
            ax.text(0.1, 0.9, model_text, ha='left', va='top', fontsize=11)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Key Insights
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        insights_text = "KEY INSIGHTS AND RECOMMENDATIONS\n\n"
        for insight in insights:
            if insight.strip():  # Skip empty lines
                insights_text += insight + "\n"
        
        ax.text(0.1, 0.9, insights_text, ha='left', va='top', fontsize=10)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Final report saved to: {report_path}")
    
    # Also create a simple text report
    text_report_path = f'{reports_dir}/report_summary_{timestamp}.txt'
    with open(text_report_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('HONG KONG IMMIGRATION TRAFFIC ANALYSIS - FINAL REPORT\n')
        f.write('='*60 + '\n\n')
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Period: {df['date'].min().date()} to {df['date'].max().date()}\n")
        f.write(f"Total Records: {len(df):,}\n\n")
        
        f.write("KEY FINDINGS:\n")
        for insight in insights[:10]:  # First 10 insights
            if insight.strip():
                f.write(insight + '\n')
    
    print(f"✅ Text summary saved to: {text_report_path}")
    
    return report_path

# Create final report
print("\n" + "="*60)
report_path = create_final_report(df_processed, pipeline_results, insights)

# %% [markdown]
# ## 9. Conclusion

# %%
print("🎯 ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nSummary of Outputs Generated:")

# List generated files and directories
outputs = [
    ("📊 Visualizations", "reports/figures/", "PNG/PDF plots"),
    ("📋 Results", "results/", "Model outputs and comparisons"),
    ("📄 Reports", "reports/", "PDF and text reports"),
    ("💾 Processed Data", "data/processed/", "Cleaned and feature-engineered data"),
    ("📓 Notebook", "main_analysis.ipynb", "Complete analysis notebook")
]

for name, path, description in outputs:
    if os.path.exists(path.split('/')[0]):  # Check if directory exists
        print(f"  ✅ {name}: {description}")
        print(f"     Location: {path}")
    else:
        print(f"  ⚠️  {name}: Not generated (check path)")

print("\n🔑 Key Files to Review:")
key_files = [
    "reports/figures/ - All visualization plots",
    "results/model_comparison_*.csv - Model performance comparison",
    "reports/final_report_*.pdf - Comprehensive PDF report",
    "reports/insights_summary.txt - Actionable insights"
]

for file_desc in key_files:
    print(f"  • {file_desc}")

print("\n🚀 Next Steps:")
next_steps = [
    "1. Review the generated visualizations in 'reports/figures/'",
    "2. Check model performance in 'results/model_comparison_*.csv'",
    "3. Read actionable insights in 'reports/insights_summary.txt'",
    "4. Share the final report 'reports/final_report_*.pdf' with stakeholders",
    "5. Consider deploying the best-performing model for real-time predictions"
]

for step in next_steps:
    print(f"  {step}")

print("\n" + "="*60)
print("✅ PROJECT COMPLETED!")
print("="*60)

# %% [markdown]
# ---
# 
# ## Appendix: Quick Access Functions
# 
# The following functions can be reused for future analyses:

# %%
# Quick analysis function
def quick_analysis(data_path="data/processed/passenger_data_daily.csv"):
    """Run a quick analysis on the data."""
    print("🚀 Starting Quick Analysis...")
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Run EDA
    perform_eda(df_processed)
    
    # Run models
    tm = TrafficModels(df=df_processed, 
                      features=[col for col in df_processed.columns if col not in ['date', 'total'] and pd.api.types.is_numeric_dtype(df_processed[col])], 
                      target='total', 
                      problem_type='regression')
    
    results = tm.run_all_models(
        regression=True,
        classification=True,
        clustering=True,
        control_point_analysis='control_point' in df_processed.columns,
        show_plots=True,
        save_plots=True
    )
    
    return df_processed, results

# Function to update analysis with new data
def update_analysis(new_data_path, output_dir="reports/updated_analysis"):
    """Update analysis with new data."""
    import shutil
    from datetime import datetime
    
    print(f"🔄 Updating analysis with new data: {new_data_path}")
    
    # Create new output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(new_output_dir, exist_ok=True)
    
    # Run analysis
    df_processed, results = quick_analysis(new_data_path)
    
    # Move results to new directory
    for item in ['reports/figures', 'results', 'reports']:
        if os.path.exists(item):
            dest = os.path.join(new_output_dir, os.path.basename(item))
            shutil.copytree(item, dest, dirs_exist_ok=True)
    
    print(f"✅ Updated analysis saved to: {new_output_dir}")
    return new_output_dir

print("\n📌 Quick Analysis Functions Available:")
print("   • quick_analysis(data_path) - Run complete analysis")
print("   • update_analysis(new_data_path) - Update with new data")