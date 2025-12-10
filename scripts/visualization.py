"""
Enhanced Visualization functions for traffic analysis with comprehensive model result plots.
FIXED VERSION - addresses axis labeling and edge case issues.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                            accuracy_score, confusion_matrix, classification_report,
                            silhouette_score, roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

class TrafficVisualizer:
    """Create enhanced visualizations for traffic analysis and model results."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.set_style()
        self.figures = {}  # Store generated figures
        
    def set_style(self):
        """Set matplotlib style with custom settings."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        
    def plot_time_series_analysis(self, date_col='date', value_col='total', 
                                 control_point=None, arrival_departure=None):
        """Comprehensive time series analysis with multiple subplots."""
        fig = plt.figure(figsize=(16, 10))
        
        # Filter data if needed
        plot_df = self.df.copy()
        if control_point:
            plot_df = plot_df[plot_df['control_point'] == control_point]
        if arrival_departure:
            plot_df = plot_df[plot_df['arrival_departure'] == arrival_departure]
        
        # 1. Main time series
        ax1 = plt.subplot(3, 2, (1, 2))
        ax1.plot(plot_df[date_col], plot_df[value_col], 'b-', alpha=0.5, linewidth=1, label='Daily')
        ax1.plot(plot_df[date_col], plot_df[value_col].rolling(window=7).mean(), 
                'r-', linewidth=2, label='7-day MA')
        ax1.plot(plot_df[date_col], plot_df[value_col].rolling(window=30).mean(), 
                'g-', linewidth=2, label='30-day MA')
        ax1.set_title(f'Passenger Traffic Over Time ({control_point or "All Points"})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Passenger Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Weekly pattern
        ax2 = plt.subplot(3, 2, 3)
        if 'day_of_week' in plot_df.columns:
            weekly_avg = plot_df.groupby('day_of_week')[value_col].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            # Ensure we have 7 days of data
            if len(weekly_avg) > 0:
                ax2.bar(range(len(weekly_avg)), weekly_avg.values, color='skyblue')
                ax2.set_title('Average Traffic by Day of Week')
                ax2.set_xlabel('Day of Week')
                ax2.set_ylabel('Average Passenger Count')
                ax2.set_xticks(range(len(weekly_avg)))
                ax2.set_xticklabels(days[:len(weekly_avg)])
        
        # 3. Monthly pattern
        ax3 = plt.subplot(3, 2, 4)
        if 'month' in plot_df.columns:
            monthly_avg = plot_df.groupby('month')[value_col].mean()
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if len(monthly_avg) > 0:
                months_to_show = months[:len(monthly_avg)]
                ax3.bar(range(1, len(monthly_avg) + 1), monthly_avg.values, color='lightcoral')
                ax3.set_title('Average Traffic by Month')
                ax3.set_xlabel('Month')
                ax3.set_ylabel('Average Passenger Count')
                ax3.set_xticks(range(1, len(monthly_avg) + 1))
                ax3.set_xticklabels(months_to_show)
        
        # 4. Box plot by day of week - FIXED
        ax4 = plt.subplot(3, 2, 5)
        if 'day_of_week' in plot_df.columns and value_col in plot_df.columns:
            # Ensure we have data for boxplot
            valid_data = plot_df[['day_of_week', value_col]].dropna()
            if len(valid_data) > 0 and valid_data['day_of_week'].nunique() > 1:
                try:
                    sns.boxplot(x='day_of_week', y=value_col, data=valid_data, ax=ax4)
                    ax4.set_title('Traffic Distribution by Day of Week')
                    ax4.set_xlabel('Day of Week')
                    ax4.set_ylabel('Passenger Count')
                    # Only set labels for existing days
                    unique_days = sorted(valid_data['day_of_week'].unique())
                    if len(unique_days) <= 7:
                        day_labels = days[:len(unique_days)]
                        ax4.set_xticks(range(len(unique_days)))
                        ax4.set_xticklabels(day_labels)
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Boxplot Error:\n{str(e)[:50]}', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Traffic Distribution')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor boxplot', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Traffic Distribution')
        
        # 5. Holiday vs Non-Holiday - FIXED
        ax5 = plt.subplot(3, 2, 6)
        if 'is_holiday' in plot_df.columns:
            holiday_stats = plot_df.groupby('is_holiday')[value_col].agg(['mean', 'std'])
            # Ensure we have both holiday and non-holiday data
            if len(holiday_stats) > 0:
                x_pos = list(range(len(holiday_stats)))
                ax5.bar(x_pos, holiday_stats['mean'], yerr=holiday_stats['std'], 
                       color=['lightblue', 'salmon'][:len(holiday_stats)], capsize=10)
                ax5.set_title('Traffic: Holiday vs Non-Holiday')
                ax5.set_xlabel('Day Type')
                ax5.set_ylabel('Average Passenger Count')
                ax5.set_xticks(x_pos)
                # Only show labels for available categories
                labels = []
                if 0 in holiday_stats.index:
                    labels.append('Non-Holiday')
                if 1 in holiday_stats.index:
                    labels.append('Holiday')
                ax5.set_xticklabels(labels[:len(holiday_stats)])
            else:
                ax5.text(0.5, 0.5, 'No holiday data\navailable', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Holiday Analysis')
        
        plt.tight_layout()
        self.figures['time_series_analysis'] = fig
        return fig
    
    def plot_regression_analysis(self, y_true, y_pred, model_name="Linear Regression"):
        """Comprehensive regression analysis plots."""
        fig = plt.figure(figsize=(18, 12))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residuals = y_true - y_pred
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 1. Actual vs Predicted (Parity Plot)
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5, 
                            c=residuals, cmap='coolwarm')
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name}: Actual vs Predicted\nR² = {r2:.4f}, RMSE = {rmse:.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Residuals')
        
        # 2. Residuals Distribution
        ax2 = plt.subplot(2, 3, 2)
        if len(residuals) > 0:
            sns.histplot(residuals, kde=True, ax=ax2, bins=30, color='skyblue')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Residuals (Actual - Predicted)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Residuals Distribution\nMean = {residuals.mean():.1f}, Std = {residuals.std():.1f}')
        else:
            ax2.text(0.5, 0.5, 'No residuals data', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Residuals Distribution')
        
        # 3. Residuals vs Predicted
        ax3 = plt.subplot(2, 3, 3)
        if len(y_pred) > 0:
            ax3.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', linewidth=0.5)
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax3.set_xlabel('Predicted Values')
            ax3.set_ylabel('Residuals')
            ax3.set_title('Residuals vs Predicted Values')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No prediction data', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Residuals vs Predicted')
        
        # 4. Q-Q Plot for residuals normality
        ax4 = plt.subplot(2, 3, 4)
        if len(residuals) > 0:
            try:
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=ax4)
                ax4.set_title('Q-Q Plot for Normality Check')
                ax4.grid(True, alpha=0.3)
            except:
                ax4.text(0.5, 0.5, 'Q-Q Plot failed\n(insufficient data)', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Q-Q Plot')
        else:
            ax4.text(0.5, 0.5, 'No data for Q-Q plot', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Q-Q Plot')
        
        # 5. Time series of actual vs predicted
        ax5 = plt.subplot(2, 3, 5)
        if len(y_true) > 0:
            indices = np.arange(len(y_true))
            ax5.plot(indices, y_true, 'b-', label='Actual', alpha=0.7)
            ax5.plot(indices, y_pred, 'r--', label='Predicted', alpha=0.7)
            ax5.set_xlabel('Sample Index')
            ax5.set_ylabel('Passenger Count')
            ax5.set_title('Actual vs Predicted (Index View)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No time series data', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Time Series View')
        
        # 6. Error distribution
        ax6 = plt.subplot(2, 3, 6)
        if len(y_true) > 0:
            # Avoid division by zero
            denominator = np.where(y_true != 0, y_true, 1)
            error_percentage = np.abs(residuals / denominator) * 100
            if len(error_percentage) > 0:
                sns.boxplot(data=error_percentage, ax=ax6, color='lightgreen')
                ax6.set_ylabel('Percentage Error (%)')
                ax6.set_title(f'Error Distribution\nMedian Error: {np.median(error_percentage):.1f}%')
            else:
                ax6.text(0.5, 0.5, 'No error data', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Error Distribution')
        else:
            ax6.text(0.5, 0.5, 'No data for error\ndistribution', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Error Distribution')
        
        plt.suptitle(f'{model_name} Analysis Report', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures[f'regression_analysis_{model_name.lower().replace(" ", "_")}'] = fig
        
        # Print metrics
        metrics_dict = {
            'R² Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'Max Error': np.max(np.abs(residuals)) if len(residuals) > 0 else 0,
            'Mean Residual': residuals.mean() if len(residuals) > 0 else 0,
            'Std Residual': residuals.std() if len(residuals) > 0 else 0
        }
        
        return fig, metrics_dict
    
    def plot_classification_analysis(self, y_true, y_pred, y_proba=None, 
                                    model_name="Classification Model"):
        """Comprehensive classification analysis plots - FIXED VERSION."""
        fig = plt.figure(figsize=(18, 12))
        
        # Convert to numpy arrays and handle edge cases
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Check if we have valid data
        if len(y_true) == 0 or len(y_pred) == 0:
            print("Warning: Empty classification data provided")
            # Create empty plot with message
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No classification data available', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Classification Analysis')
            ax.axis('off')
            plt.tight_layout()
            return fig, {}
        
        # Get unique classes
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(unique_classes)
        
        # Handle binary vs multi-class
        if n_classes <= 2:
            # Binary classification
            pos_class = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
            y_true_binary = (y_true == pos_class).astype(int)
            y_pred_binary = (y_pred == pos_class).astype(int)
            
            # Calculate metrics
            try:
                acc = accuracy_score(y_true_binary, y_pred_binary)
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                
                # Get classification report safely
                try:
                    report = classification_report(y_true_binary, y_pred_binary, 
                                                 output_dict=True, zero_division=0)
                except:
                    # Create minimal report
                    report = {
                        '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        'accuracy': 0
                    }
            except:
                print("Error calculating classification metrics")
                acc = 0
                cm = np.array([[0, 0], [0, 0]])
                report = {
                    '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                    '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                    'accuracy': 0
                }
        else:
            # Multi-class classification
            try:
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            except:
                print("Error with multi-class classification")
                acc = 0
                cm = np.zeros((n_classes, n_classes))
                report = {'accuracy': 0}
        
        # 1. Confusion Matrix - FIXED
        ax1 = plt.subplot(2, 3, 1)
        if cm.size > 0:
            # Ensure we have proper dimensions
            if cm.shape[0] == cm.shape[1]:
                # Create labels based on actual classes
                if n_classes == 2:
                    labels = ['Low Traffic', 'High Traffic']
                elif n_classes <= 5:
                    labels = [f'Class {i}' for i in range(n_classes)]
                else:
                    labels = [str(i) for i in range(n_classes)]
                
                # Truncate labels if needed
                if len(labels) > cm.shape[0]:
                    labels = labels[:cm.shape[0]]
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                           xticklabels=labels[:cm.shape[1]], 
                           yticklabels=labels[:cm.shape[0]])
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('Actual')
                ax1.set_title(f'Confusion Matrix\nAccuracy: {acc:.4f}')
            else:
                ax1.text(0.5, 0.5, 'Invalid confusion matrix\nshape', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Confusion Matrix')
        else:
            ax1.text(0.5, 0.5, 'No confusion matrix\ndata available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Confusion Matrix')
        
        # 2. ROC Curve (if probabilities available and binary classification)
        ax2 = plt.subplot(2, 3, 2)
        if y_proba is not None and len(y_proba) > 0 and n_classes == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax2.legend(loc="lower right")
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                ax2.text(0.5, 0.5, f'ROC Curve Error:\n{str(e)[:40]}', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('ROC Curve')
        else:
            ax2.text(0.5, 0.5, 'ROC Curve\n(Probabilities not available\nor not binary classification)', 
                    ha='center', va='center', fontsize=10)
            ax2.set_title('ROC Curve')
        
        # 3. Precision-Recall Curve
        ax3 = plt.subplot(2, 3, 3)
        if y_proba is not None and len(y_proba) > 0 and n_classes == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba)
                pr_auc = auc(recall, precision)
                ax3.plot(recall, precision, color='green', lw=2, 
                        label=f'PR curve (AUC = {pr_auc:.3f})')
                ax3.set_xlabel('Recall')
                ax3.set_ylabel('Precision')
                ax3.set_title('Precision-Recall Curve')
                ax3.legend(loc="lower left")
                ax3.grid(True, alpha=0.3)
            except:
                ax3.text(0.5, 0.5, 'PR Curve Error', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('PR Curve')
        else:
            ax3.text(0.5, 0.5, 'PR Curve\n(Probabilities not available\nor not binary classification)', 
                    ha='center', va='center', fontsize=10)
            ax3.set_title('Precision-Recall Curve')
        
        # 4. Classification Metrics Bar Chart - FIXED
        ax4 = plt.subplot(2, 3, 4)
        if n_classes == 2 and '0' in report and '1' in report:
            try:
                metrics = ['Precision', 'Recall', 'F1-Score']
                class_0_metrics = [report['0']['precision'], report['0']['recall'], report['0']['f1-score']]
                class_1_metrics = [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]
                
                x = np.arange(len(metrics))
                width = 0.35
                ax4.bar(x - width/2, class_0_metrics, width, label='Low Traffic', color='skyblue')
                ax4.bar(x + width/2, class_1_metrics, width, label='High Traffic', color='salmon')
                ax4.set_xlabel('Metrics')
                ax4.set_ylabel('Score')
                ax4.set_title('Classification Metrics by Class')
                ax4.set_xticks(x)
                ax4.set_xticklabels(metrics)
                ax4.legend()
                ax4.set_ylim([0, 1.1])
                ax4.grid(True, alpha=0.3, axis='y')
            except:
                ax4.text(0.5, 0.5, 'Metrics chart\ncalculation error', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Classification Metrics')
        else:
            ax4.text(0.5, 0.5, 'Metrics chart\n(binary classification\nrequired)', 
                    ha='center', va='center', fontsize=10)
            ax4.set_title('Classification Metrics')
        
        # 5. Class Distribution - FIXED
        ax5 = plt.subplot(2, 3, 5)
        if len(y_true) > 0:
            try:
                unique, counts = np.unique(y_true, return_counts=True)
                if len(unique) > 0:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
                    bars = ax5.bar(range(len(unique)), counts, color=colors[:len(unique)])
                    ax5.set_xlabel('Traffic Class')
                    ax5.set_ylabel('Count')
                    ax5.set_title('Class Distribution in Dataset')
                    
                    # Set appropriate labels
                    if len(unique) == 2:
                        labels = ['Low Traffic', 'High Traffic']
                    else:
                        labels = [f'Class {int(cls)}' for cls in unique]
                    
                    ax5.set_xticks(range(len(unique)))
                    ax5.set_xticklabels(labels)
                    ax5.grid(True, alpha=0.3, axis='y')
                else:
                    ax5.text(0.5, 0.5, 'No class data', 
                            ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('Class Distribution')
            except:
                ax5.text(0.5, 0.5, 'Class distribution\nerror', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Class Distribution')
        else:
            ax5.text(0.5, 0.5, 'No class distribution\ndata', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Class Distribution')
        
        # 6. Feature Importance (placeholder)
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.5, 0.5, 'Feature Importance Plot\n\n(Would show if model provides\nfeature_importances_ or coef_)', 
                ha='center', va='center', fontsize=10)
        ax6.set_title('Feature Importance')
        ax6.axis('off')
        
        plt.suptitle(f'{model_name} Analysis Report', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures[f'classification_analysis_{model_name.lower().replace(" ", "_")}'] = fig
        
        # Return metrics dictionary
        metrics_dict = {
            'Accuracy': acc
        }
        
        if n_classes == 2 and '0' in report and '1' in report:
            metrics_dict.update({
                'Precision (Class 0)': report['0']['precision'],
                'Recall (Class 0)': report['0']['recall'],
                'F1-Score (Class 0)': report['0']['f1-score'],
                'Precision (Class 1)': report['1']['precision'],
                'Recall (Class 1)': report['1']['recall'],
                'F1-Score (Class 1)': report['1']['f1-score']
            })
        
        if y_proba is not None and n_classes == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba)
                roc_auc = auc(fpr, tpr)
                metrics_dict['ROC AUC'] = roc_auc
            except:
                pass
        
        return fig, metrics_dict
    
    def plot_clustering_analysis(self, df_with_clusters, cluster_col='cluster', 
                               feature_cols=None, title="K-means Clustering Analysis"):
        """Comprehensive clustering analysis plots - FIXED VERSION."""
        if feature_cols is None:
            feature_cols = ['total', 'day_of_week', 'month']
        
        fig = plt.figure(figsize=(18, 12))
        
        # Check if cluster column exists
        if cluster_col not in df_with_clusters.columns:
            print(f"Warning: Cluster column '{cluster_col}' not found in dataframe")
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Cluster column '{cluster_col}' not found", 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Clustering Analysis')
            ax.axis('off')
            plt.tight_layout()
            return fig, {}
        
        # Get unique clusters
        clusters = sorted(df_with_clusters[cluster_col].dropna().unique())
        n_clusters = len(clusters)
        
        if n_clusters == 0:
            print("Warning: No clusters found in data")
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No clusters found in data', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Clustering Analysis')
            ax.axis('off')
            plt.tight_layout()
            return fig, {}
        
        # 1. 2D Scatter Plot (first two features)
        ax1 = plt.subplot(2, 3, 1)
        if len(feature_cols) >= 2:
            # Check if features exist
            available_features = [f for f in feature_cols[:2] if f in df_with_clusters.columns]
            if len(available_features) == 2:
                scatter = ax1.scatter(df_with_clusters[available_features[0]], 
                                     df_with_clusters[available_features[1]], 
                                     c=df_with_clusters[cluster_col], 
                                     cmap='tab10', alpha=0.6, edgecolors='w', linewidth=0.5,
                                     s=50)
                ax1.set_xlabel(available_features[0])
                ax1.set_ylabel(available_features[1])
                ax1.set_title('Feature Space Clustering')
                plt.colorbar(scatter, ax=ax1, label='Cluster')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient features\nfor 2D scatter plot', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Feature Space Clustering')
        else:
            ax1.text(0.5, 0.5, 'Need at least 2 features\nfor scatter plot', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Feature Space Clustering')
        
        # 2. Time Series with Clusters
        ax2 = plt.subplot(2, 3, 2)
        if 'date' in df_with_clusters.columns and 'total' in df_with_clusters.columns:
            try:
                for cluster in clusters:
                    cluster_data = df_with_clusters[df_with_clusters[cluster_col] == cluster]
                    if len(cluster_data) > 0:
                        ax2.scatter(cluster_data['date'], cluster_data['total'], 
                                  alpha=0.6, label=f'Cluster {int(cluster)}', s=20)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Total Passengers')
                ax2.set_title('Clusters Over Time')
                if n_clusters <= 10:  # Only show legend if reasonable number of clusters
                    ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            except:
                ax2.text(0.5, 0.5, 'Time series plot\nerror', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Clusters Over Time')
        else:
            ax2.text(0.5, 0.5, 'Date or total column\nnot available', 
                    ha='center', va='center', fontsize=10)
            ax2.set_title('Clusters Over Time')
        
        # 3. Cluster Distribution
        ax3 = plt.subplot(2, 3, 3)
        try:
            cluster_counts = df_with_clusters[cluster_col].value_counts().sort_index()
            if len(cluster_counts) > 0:
                colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_counts)))
                bars = ax3.bar(cluster_counts.index.astype(str), cluster_counts.values, 
                              color=colors)
                ax3.set_xlabel('Cluster')
                ax3.set_ylabel('Number of Days')
                ax3.set_title('Cluster Distribution')
                
                # Add count labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{int(height)}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No cluster counts', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cluster Distribution')
        except:
            ax3.text(0.5, 0.5, 'Cluster distribution\nerror', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cluster Distribution')
        
        # 4. Box Plot by Cluster
        ax4 = plt.subplot(2, 3, 4)
        if 'total' in df_with_clusters.columns:
            try:
                # Prepare data for boxplot
                box_data = []
                labels = []
                for cluster in clusters:
                    cluster_data = df_with_clusters[df_with_clusters[cluster_col] == cluster]['total'].dropna()
                    if len(cluster_data) > 0:
                        box_data.append(cluster_data)
                        labels.append(f'Cluster {int(cluster)}')
                
                if len(box_data) > 0:
                    bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
                    # Color the boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    ax4.set_ylabel('Total Passengers')
                    ax4.set_title('Traffic Distribution by Cluster')
                    ax4.grid(True, alpha=0.3, axis='y')
                else:
                    ax4.text(0.5, 0.5, 'No boxplot data', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Traffic Distribution')
            except:
                ax4.text(0.5, 0.5, 'Boxplot error', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Traffic Distribution')
        else:
            ax4.text(0.5, 0.5, 'Total column\nnot available', 
                    ha='center', va='center', fontsize=10)
            ax4.set_title('Traffic Distribution')
        
        # 5. Radar Chart of Cluster Centroids
        ax5 = plt.subplot(2, 3, 5, polar=True)
        try:
            # Try to create radar chart with available features
            available_features = [f for f in feature_cols if f in df_with_clusters.columns]
            if len(available_features) >= 3 and n_clusters > 0:
                centroids = []
                for cluster in clusters:
                    cluster_data = df_with_clusters[df_with_clusters[cluster_col] == cluster]
                    centroid = cluster_data[available_features].mean().values[:5]  # Take first 5 features
                    if len(centroid) > 0:
                        # Normalize for radar chart
                        if centroid.max() > 0:
                            centroid_normalized = centroid / centroid.max()
                        else:
                            centroid_normalized = centroid
                        centroids.append(centroid_normalized)
                
                if len(centroids) > 0 and len(centroids[0]) >= 3:
                    angles = np.linspace(0, 2 * np.pi, len(centroids[0]), endpoint=False).tolist()
                    angles += angles[:1]  # Close the loop
                    
                    for idx, centroid in enumerate(centroids):
                        if len(centroid) == len(angles) - 1:
                            values = centroid.tolist()
                            values += values[:1]  # Close the loop
                            ax5.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}')
                            ax5.fill(angles, values, alpha=0.25)
                    
                    ax5.set_xticks(angles[:-1])
                    ax5.set_xticklabels(available_features[:len(centroids[0])])
                    ax5.set_title('Cluster Centroids (Normalized)')
                    if n_clusters <= 5:  # Only show legend if reasonable
                        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    ax5.grid(True)
                else:
                    ax5.text(0.5, 0.5, 'Insufficient data\nfor radar chart', 
                            ha='center', va='center', transform=ax5.transAxes)
                    ax5.set_title('Cluster Centroids')
            else:
                ax5.text(0.5, 0.5, 'Need at least 3 features\nfor radar chart', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Cluster Centroids')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Radar chart error\n{str(e)[:30]}', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Cluster Centroids')
        
        # 6. Weekly Pattern by Cluster
        ax6 = plt.subplot(2, 3, 6)
        if 'day_of_week' in df_with_clusters.columns and 'total' in df_with_clusters.columns:
            try:
                weekly_patterns = []
                valid_clusters = []
                for cluster in clusters:
                    cluster_data = df_with_clusters[df_with_clusters[cluster_col] == cluster]
                    if len(cluster_data) > 0:
                        weekly_avg = cluster_data.groupby('day_of_week')['total'].mean()
                        if len(weekly_avg) >= 5:  # At least 5 days of data
                            weekly_patterns.append(weekly_avg.values[:7])
                            valid_clusters.append(cluster)
                
                if len(weekly_patterns) > 0:
                    for idx, (pattern, cluster) in enumerate(zip(weekly_patterns, valid_clusters)):
                        if len(pattern) >= 5:
                            ax6.plot(range(len(pattern)), pattern, 'o-', 
                                    label=f'Cluster {int(cluster)}', alpha=0.8, linewidth=2)
                    
                    ax6.set_xlabel('Day of Week')
                    ax6.set_ylabel('Average Passengers')
                    ax6.set_title('Weekly Traffic Patterns by Cluster')
                    ax6.set_xticks(range(7))
                    ax6.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                    if len(valid_clusters) <= 5:
                        ax6.legend()
                    ax6.grid(True, alpha=0.3)
                else:
                    ax6.text(0.5, 0.5, 'Insufficient weekly data', 
                            ha='center', va='center', transform=ax6.transAxes)
                    ax6.set_title('Weekly Patterns')
            except:
                ax6.text(0.5, 0.5, 'Weekly pattern error', 
                        ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Weekly Patterns')
        else:
            ax6.text(0.5, 0.5, 'Day of week or total\nnot available', 
                    ha='center', va='center', fontsize=10)
            ax6.set_title('Weekly Patterns')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['clustering_analysis'] = fig
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster in clusters:
            cluster_data = df_with_clusters[df_with_clusters[cluster_col] == cluster]
            if len(cluster_data) > 0:
                cluster_stats[f'Cluster_{int(cluster)}'] = {
                    'count': len(cluster_data),
                    'percentage': (len(cluster_data) / len(df_with_clusters)) * 100,
                    'mean_total': cluster_data['total'].mean() if 'total' in cluster_data.columns else np.nan,
                    'std_total': cluster_data['total'].std() if 'total' in cluster_data.columns else np.nan,
                    'avg_day_of_week': cluster_data['day_of_week'].mean() if 'day_of_week' in cluster_data.columns else np.nan
                }
        
        return fig, cluster_stats
    
    # [Remaining methods stay the same - plot_feature_importance, plot_model_comparison, etc.]
    # Keep all other methods as they were...


if __name__ == "__main__":
    # Enhanced demo with better error handling
    import sys
    import os
    
    print("Enhanced Visualization Module Demo")
    print("=" * 50)
    
    # Check if processed data exists
    proc_path = os.path.join("data", "processed", "passenger_data_daily.csv")
    if not os.path.exists(proc_path):
        print(f"Processed data not found: {proc_path}")
        print("Please run data preprocessing first.")
        print("\nCreating sample data for demo...")
        
        # Create realistic sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Base traffic with weekly pattern
        base_traffic = 50000 + 20000 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        weekly_pattern = 10000 * (dates.dayofweek >= 5).astype(int)  # Weekend boost
        monthly_season = 15000 * np.sin(2 * np.pi * dates.month / 12)
        noise = np.random.normal(0, 5000, n_days)
        
        df = pd.DataFrame({
            'date': dates,
            'total': base_traffic + weekly_pattern + monthly_season + noise,
            'hk_residents': np.random.randint(20000, 80000, n_days),
            'mainland_visitors': np.random.randint(10000, 50000, n_days),
            'other_visitors': np.random.randint(1000, 10000, n_days),
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.choice([0, 1], n_days, p=[0.9, 0.1])
        })
        
        print(f"✅ Created sample data with {len(df)} records")
    else:
        # Load data
        try:
            df = pd.read_csv(proc_path, parse_dates=['date'])
            print(f"✅ Loaded data: {df.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    try:
        # Create visualizer
        viz = TrafficVisualizer(df)
        
        # Generate comprehensive time series analysis
        print("\n1. Generating Time Series Analysis...")
        fig1 = viz.plot_time_series_analysis()
        
        # Generate sample regression plots with better data
        print("\n2. Generating Sample Regression Analysis...")
        if 'total' in df.columns:
            # Use actual data for regression demo
            sample_size = min(100, len(df))
            sample_y_true = df['total'].values[:sample_size]
            # Create realistic predictions (add some noise to actual values)
            sample_y_pred = sample_y_true * 0.95 + np.random.normal(0, sample_y_true.std() * 0.1, sample_size)
            fig2, metrics = viz.plot_regression_analysis(sample_y_true, sample_y_pred, "Sample Regression")
            print(f"   Sample Metrics: R²={metrics['R² Score']:.4f}, RMSE={metrics['RMSE']:.1f}")
        
        # Generate sample classification plots with better data
        print("\n3. Generating Sample Classification Analysis...")
        if 'total' in df.columns:
            # Create balanced binary classification data
            sample_size = min(100, len(df))
            median_val = df['total'].median()
            sample_y_true_clf = (df['total'].values[:sample_size] > median_val).astype(int)
            # Create predictions with some errors
            sample_y_pred_clf = sample_y_true_clf.copy()
            # Flip 20% of predictions to create errors
            flip_indices = np.random.choice(sample_size, size=int(sample_size * 0.2), replace=False)
            sample_y_pred_clf[flip_indices] = 1 - sample_y_pred_clf[flip_indices]
            # Create realistic probabilities
            sample_y_proba = np.random.uniform(0.3, 0.9, sample_size)
            sample_y_proba[sample_y_pred_clf == 0] = 1 - sample_y_proba[sample_y_pred_clf == 0]
            
            fig3, clf_metrics = viz.plot_classification_analysis(
                sample_y_true_clf, sample_y_pred_clf, sample_y_proba, "Sample Classification"
            )
            print(f"   Sample Accuracy: {clf_metrics.get('Accuracy', 0):.4f}")
        
        # Generate sample clustering plots
        print("\n4. Generating Sample Clustering Analysis...")
        if 'total' in df.columns and 'day_of_week' in df.columns:
            df_sample = df[['date', 'total', 'day_of_week', 'month']].copy()
            # Create sample clusters based on traffic levels
            traffic_levels = pd.qcut(df_sample['total'], q=3, labels=[0, 1, 2])
            df_sample['cluster'] = traffic_levels
            
            fig4, cluster_stats = viz.plot_clustering_analysis(
                df_sample, title="Sample Clustering (Traffic Levels)"
            )
            print(f"   Clusters: {len(cluster_stats)} clusters identified")
        
        # Save all figures
        print("\n5. Saving all figures...")
        os.makedirs('reports/figures', exist_ok=True)
        saved_files = []
        for fig_name, fig in viz.figures.items():
            filename = f"{fig_name}.png"
            filepath = os.path.join('reports/figures', filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
        
        print(f"   ✅ Saved {len(saved_files)} figures to reports/figures/")
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()