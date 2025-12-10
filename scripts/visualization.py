"""
Enhanced visualization module for passenger traffic analysis.
Provides comprehensive plotting capabilities for traffic data analysis and model results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TrafficVisualizer:
    """Enhanced visualization class for traffic data analysis and model results."""
    
    def __init__(self, data: pd.DataFrame = None):
        """
        Initialize TrafficVisualizer with optional data.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing traffic data
        """
        self.data = data
        self.figures = {}
        
    def set_data(self, data: pd.DataFrame):
        """Set or update the data for visualization."""
        self.data = data.copy()
        
    def plot_time_series(self, 
                        date_col: str = 'date',
                        target_col: str = 'total',
                        title: str = 'Passenger Traffic Over Time',
                        figsize: Tuple[int, int] = (15, 6),
                        show_rolling: bool = True,
                        rolling_window: int = 7) -> plt.Figure:
        """
        Plot time series of passenger traffic.
        
        Parameters:
        -----------
        date_col : str
            Column name for dates
        target_col : str
            Column name for traffic volume
        title : str
            Plot title
        figsize : tuple
            Figure size
        show_rolling : bool
            Whether to show rolling average
        rolling_window : int
            Window size for rolling average
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure date column is datetime
        data_copy = self.data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data_copy[date_col]):
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
        
        # Sort by date
        data_copy = data_copy.sort_values(date_col)
        
        # Plot raw data
        ax.plot(data_copy[date_col], data_copy[target_col], 
                label='Daily Traffic', alpha=0.5, linewidth=0.5)
        
        # Plot rolling average
        if show_rolling and len(data_copy) > rolling_window:
            rolling_avg = data_copy[target_col].rolling(window=rolling_window).mean()
            ax.plot(data_copy[date_col], rolling_avg, 
                    label=f'{rolling_window}-Day Moving Avg', 
                    linewidth=2, color='red')
        
        # Add markers for peaks and troughs
        max_idx = data_copy[target_col].idxmax()
        min_idx = data_copy[target_col].idxmin()
        
        ax.scatter(data_copy.loc[max_idx, date_col], data_copy.loc[max_idx, target_col],
                  color='red', s=100, zorder=5, label=f'Peak: {data_copy.loc[max_idx, target_col]:,.0f}')
        ax.scatter(data_copy.loc[min_idx, date_col], data_copy.loc[min_idx, target_col],
                  color='green', s=100, zorder=5, label=f'Trough: {data_copy.loc[min_idx, target_col]:,.0f}')
        
        # Formatting
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{target_col.replace("_", " ").title()}')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self.figures['time_series'] = fig
        return fig
    
    def plot_distribution(self, 
                         target_col: str = 'total',
                         bins: int = 50,
                         title: str = 'Traffic Distribution',
                         figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot distribution of traffic data.
        
        Parameters:
        -----------
        target_col : str
            Column name for traffic volume
        bins : int
            Number of bins for histogram
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        data_series = self.data[target_col].dropna()
        
        # Histogram with KDE
        axes[0].hist(data_series, bins=bins, edgecolor='black', alpha=0.7, density=True)
        sns.kdeplot(data_series, ax=axes[0], color='red', linewidth=2)
        axes[0].axvline(data_series.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {data_series.mean():,.0f}')
        axes[0].axvline(data_series.median(), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {data_series.median():,.0f}')
        axes[0].set_xlabel(f'{target_col.replace("_", " ").title()}')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Distribution of {target_col.replace("_", " ").title()}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Box plot
        box_data = axes[1].boxplot(data_series, vert=False, patch_artist=True)
        box_data['boxes'][0].set_facecolor('lightblue')
        
        # Add statistics annotations
        stats_text = (f"Min: {data_series.min():,.0f}\n"
                     f"Q1: {data_series.quantile(0.25):,.0f}\n"
                     f"Median: {data_series.median():,.0f}\n"
                     f"Q3: {data_series.quantile(0.75):,.0f}\n"
                     f"Max: {data_series.max():,.0f}\n"
                     f"IQR: {data_series.quantile(0.75) - data_series.quantile(0.25):,.0f}")
        
        axes[1].text(0.98, 0.02, stats_text, transform=axes[1].transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1].set_xlabel(f'{target_col.replace("_", " ").title()}')
        axes[1].set_title(f'Box Plot of {target_col.replace("_", " ").title()}')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self.figures['distribution'] = fig
        return fig
    
    def plot_correlation_heatmap(self,
                                columns: List[str] = None,
                                title: str = 'Feature Correlation Heatmap',
                                figsize: Tuple[int, int] = (10, 8),
                                annot: bool = True) -> plt.Figure:
        """
        Plot correlation heatmap of numerical features.
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to include in correlation
        title : str
            Plot title
        figsize : tuple
            Figure size
        annot : bool
            Whether to annotate correlation values
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        if columns is None:
            # Select numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols[:10]  # Limit to first 10 for readability
        
        # Calculate correlation matrix
        corr_matrix = self.data[columns].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, fmt='.2f',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        self.figures['correlation_heatmap'] = fig
        return fig
    
    def plot_seasonality(self,
                        date_col: str = 'date',
                        target_col: str = 'total',
                        title: str = 'Traffic Seasonality Analysis',
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot seasonality patterns in traffic data.
        
        Parameters:
        -----------
        date_col : str
            Column name for dates
        target_col : str
            Column name for traffic volume
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        # Prepare data
        data_copy = self.data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data_copy[date_col]):
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
        
        # Extract temporal features
        data_copy['year'] = data_copy[date_col].dt.year
        data_copy['month'] = data_copy[date_col].dt.month
        data_copy['day_of_week'] = data_copy[date_col].dt.dayofweek
        data_copy['hour'] = data_copy[date_col].dt.hour if 'hour' in data_copy.columns else 0
        data_copy['quarter'] = data_copy[date_col].dt.quarter
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Monthly patterns
        monthly_avg = data_copy.groupby('month')[target_col].mean()
        axes[0].bar(monthly_avg.index, monthly_avg.values, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel(f'Average {target_col.replace("_", " ").title()}')
        axes[0].set_title('Average Monthly Traffic')
        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 2. Day of week patterns
        weekday_avg = data_copy.groupby('day_of_week')[target_col].mean()
        axes[1].bar(weekday_avg.index, weekday_avg.values, color='lightgreen', edgecolor='black')
        axes[1].set_xlabel('Day of Week')
        axes[1].set_ylabel(f'Average {target_col.replace("_", " ").title()}')
        axes[1].set_title('Average Traffic by Day of Week')
        axes[1].set_xticks(range(0, 7))
        axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 3. Hourly patterns (if available)
        if 'hour' in data_copy.columns and data_copy['hour'].nunique() > 1:
            hourly_avg = data_copy.groupby('hour')[target_col].mean()
            axes[2].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
            axes[2].set_xlabel('Hour of Day')
            axes[2].set_ylabel(f'Average {target_col.replace("_", " ").title()}')
            axes[2].set_title('Average Hourly Traffic')
            axes[2].grid(True, alpha=0.3)
            axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        else:
            # Quarterly patterns as alternative
            quarterly_avg = data_copy.groupby('quarter')[target_col].mean()
            axes[2].bar(quarterly_avg.index, quarterly_avg.values, color='orange', edgecolor='black')
            axes[2].set_xlabel('Quarter')
            axes[2].set_ylabel(f'Average {target_col.replace("_", " ").title()}')
            axes[2].set_title('Average Quarterly Traffic')
            axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 4. Yearly trends
        yearly_avg = data_copy.groupby('year')[target_col].mean()
        axes[3].plot(yearly_avg.index, yearly_avg.values, marker='s', linewidth=2, color='red')
        axes[3].set_xlabel('Year')
        axes[3].set_ylabel(f'Average {target_col.replace("_", " ").title()}')
        axes[3].set_title('Yearly Traffic Trends')
        axes[3].grid(True, alpha=0.3)
        axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['seasonality'] = fig
        return fig
    
    def plot_regression_analysis(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            model_name: str = "Model",
                            figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:  # Changed return type
        """
        Plot regression analysis results.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted target values
        model_name : str
            Name of the model for title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        Tuple[plt.Figure, Dict]
            Figure object and calculated metrics
        """
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Actual vs Predicted scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
        sns.kdeplot(residuals, ax=axes[2], color='red', linewidth=2)
        axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'{model_name}: Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        # Add metrics text
        metrics_text = (f"R²: {r2:.4f}\n"
                       f"RMSE: {rmse:.2f}\n"
                       f"MAE: {mae:.2f}\n"
                       f"MAPE: {mape:.2f}%")
        
        axes[2].text(0.98, 0.98, metrics_text, transform=axes[2].transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'{model_name} - Regression Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures[f'regression_analysis_{model_name}'] = fig
        return fig  # REMOVE: , metrics from return statement
    
    def plot_classification_analysis(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_proba: np.ndarray = None,
                                model_name: str = "Model",
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:  # Changed return type
        """
        Plot feature importance/coefficients.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        coefficients : np.ndarray or dict
            Coefficients or importance values
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Convert coefficients to dictionary if not already
        if isinstance(coefficients, dict):
            coeff_dict = coefficients
        else:
            coeff_dict = dict(zip(feature_names, coefficients))
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': list(coeff_dict.keys()),
            'Coefficient': list(coeff_dict.values())
        })
        
        # Sort by absolute value
        importance_df['Abs_Coefficient'] = np.abs(importance_df['Coefficient'])
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Bar plot of coefficients
        colors = ['red' if x < 0 else 'green' for x in importance_df['Coefficient']]
        axes[0].barh(range(len(importance_df)), importance_df['Coefficient'], color=colors, edgecolor='black')
        axes[0].set_yticks(range(len(importance_df)))
        axes[0].set_yticklabels(importance_df['Feature'])
        axes[0].set_xlabel('Coefficient Value')
        axes[0].set_title('Feature Coefficients')
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        axes[0].invert_yaxis()  # Highest importance at top
        
        # Add coefficient values on bars
        for i, (coeff, color) in enumerate(zip(importance_df['Coefficient'], colors)):
            axes[0].text(coeff, i, f'{coeff:.4f}', 
                        ha='left' if coeff >= 0 else 'right',
                        va='center',
                        color='white' if abs(coeff) > np.abs(importance_df['Coefficient']).max() * 0.5 else 'black',
                        fontweight='bold')
        
        # 2. Top N features by absolute value
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n)
        
        axes[1].barh(range(top_n), top_features['Abs_Coefficient'], 
                    color='skyblue', edgecolor='black')
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(top_features['Feature'])
        axes[1].set_xlabel('Absolute Coefficient Value')
        axes[1].set_title(f'Top {top_n} Most Important Features')
        
        # Add values on bars
        for i, val in enumerate(top_features['Abs_Coefficient']):
            axes[1].text(val, i, f'{val:.4f}', 
                        ha='left', va='center',
                        fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Return only the figure
        self.figures[f'classification_analysis_{model_name}'] = fig
        return fig  # REMOVE: , metrics from return statement
    
    def plot_clustering_analysis(self,
                            data: pd.DataFrame,
                            cluster_col: str = 'cluster',
                            feature_cols: List[str] = None,
                            title: str = "Clustering Analysis",
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:  # Changed return type
        """
        Plot classification analysis results.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Predicted probabilities
        model_name : str
            Name of the model for title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        Tuple[plt.Figure, Dict]
            Figure object and calculated metrics
        """
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, confusion_matrix, roc_curve, auc, 
                                   classification_report, precision_recall_curve)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')
        
        # 2. ROC Curve
        if y_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
            
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend(loc="lower right")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No probability scores available\nfor ROC curve',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title('ROC Curve (Not Available)')
        
        # 3. Precision-Recall Curve
        if y_proba is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
            axes[2].plot(recall_vals, precision_vals, color='green', lw=2)
            axes[2].set_xlabel('Recall')
            axes[2].set_ylabel('Precision')
            axes[2].set_title('Precision-Recall Curve')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No probability scores available\nfor Precision-Recall curve',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes, fontsize=12)
            axes[2].set_title('Precision-Recall Curve (Not Available)')
        
        # 4. Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [accuracy, precision, recall, f1]
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
        
        bars = axes[3].bar(metrics_names, metrics_values, color=colors, edgecolor='black')
        axes[3].set_ylabel('Score')
        axes[3].set_title('Classification Metrics')
        axes[3].set_ylim([0, 1.1])
        axes[3].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Prediction Distribution
        pred_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Correct': y_true == y_pred
        })
        
        correct_counts = pred_df['Correct'].value_counts()
        axes[4].pie(correct_counts.values, labels=['Correct', 'Incorrect'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%',
                   startangle=90, explode=(0.1, 0))
        axes[4].set_title('Prediction Accuracy')
        
        # 6. Probability Distribution (if available)
        if y_proba is not None:
            proba_df = pd.DataFrame({
                'Probability': y_proba,
                'Actual': y_true
            })
            
            for label in [0, 1]:
                subset = proba_df[proba_df['Actual'] == label]
                axes[5].hist(subset['Probability'], bins=20, alpha=0.5, 
                            label=f'Actual Class {label}', density=True)
            
            axes[5].set_xlabel('Predicted Probability')
            axes[5].set_ylabel('Density')
            axes[5].set_title('Probability Distribution by Actual Class')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, 'No probability scores available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[5].transAxes, fontsize=12)
            axes[5].set_title('Probability Distribution (Not Available)')
        
        plt.suptitle(f'{model_name} - Classification Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Return only the figure
        self.figures['clustering_analysis'] = fig
        return fig  # REMOVE: , cluster_stats from return statement
    
    def plot_clustering_analysis(self,
                                data: pd.DataFrame,
                                cluster_col: str = 'cluster',
                                feature_cols: List[str] = None,
                                title: str = "Clustering Analysis",
                                figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, Dict]:
        """
        Plot clustering analysis results.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with cluster labels
        cluster_col : str
            Column name containing cluster labels
        feature_cols : list, optional
            Features to analyze
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        Tuple[plt.Figure, Dict]
            Figure object and cluster statistics
        """
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != cluster_col][:6]
        
        n_clusters = data[cluster_col].nunique()
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = data[data[cluster_col] == cluster]
            cluster_stats[cluster] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }
            
            for col in feature_cols:
                if col in cluster_data.columns:
                    cluster_stats[cluster][f'{col}_mean'] = cluster_data[col].mean()
                    cluster_stats[cluster][f'{col}_std'] = cluster_data[col].std()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Cluster Size Distribution
        sizes = [stats['size'] for stats in cluster_stats.values()]
        percentages = [stats['percentage'] for stats in cluster_stats.values()]
        
        bars = axes[0].bar(range(n_clusters), sizes, color='lightblue', edgecolor='black')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Cluster Size Distribution')
        axes[0].set_xticks(range(n_clusters))
        
        # Add percentage labels
        for i, (bar, percentage) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 2. Feature Means by Cluster
        if len(feature_cols) >= 1:
            feature = feature_cols[0]
            means = [stats.get(f'{feature}_mean', 0) for stats in cluster_stats.values()]
            stds = [stats.get(f'{feature}_std', 0) for stats in cluster_stats.values()]
            
            axes[1].bar(range(n_clusters), means, yerr=stds, 
                       color='lightgreen', edgecolor='black', capsize=5)
            axes[1].set_xlabel('Cluster')
            axes[1].set_ylabel(f'Mean {feature}')
            axes[1].set_title(f'{feature} by Cluster')
            axes[1].set_xticks(range(n_clusters))
        
        # 3. Scatter Plot of First Two Features
        if len(feature_cols) >= 2:
            scatter = axes[2].scatter(data[feature_cols[0]], data[feature_cols[1]],
                                     c=data[cluster_col], cmap='tab10', alpha=0.6, 
                                     edgecolors='black', linewidth=0.5)
            axes[2].set_xlabel(feature_cols[0])
            axes[2].set_ylabel(feature_cols[1])
            axes[2].set_title(f'{feature_cols[0]} vs {feature_cols[1]}')
            axes[2].grid(True, alpha=0.3)
            
            # Add legend
            handles = []
            for cluster in range(n_clusters):
                handle = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=plt.cm.tab10(cluster/n_clusters),
                                   markersize=8, label=f'Cluster {cluster}')
                handles.append(handle)
            axes[2].legend(handles=handles, title='Clusters')
        
        # 4. Parallel Coordinates Plot
        if len(feature_cols) >= 3:
            from pandas.plotting import parallel_coordinates
            
            plot_data = data[[cluster_col] + feature_cols[:4]].copy()
            plot_data[cluster_col] = plot_data[cluster_col].astype(str)
            
            parallel_coordinates(plot_data, cluster_col, ax=axes[3], alpha=0.3)
            axes[3].set_title('Parallel Coordinates Plot')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 5. Radar Chart of Cluster Centroids
        if len(feature_cols) >= 3:
            from math import pi
            
            # Normalize features for radar chart
            norm_data = data[feature_cols[:5]].copy()
            for col in norm_data.columns:
                if norm_data[col].std() > 0:
                    norm_data[col] = (norm_data[col] - norm_data[col].min()) / (norm_data[col].max() - norm_data[col].min())
            
            norm_data[cluster_col] = data[cluster_col]
            
            # Calculate cluster means
            cluster_means = norm_data.groupby(cluster_col).mean()
            
            # Set up radar chart
            categories = cluster_means.columns.tolist()
            N = len(categories)
            
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]
            
            for cluster in range(n_clusters):
                values = cluster_means.loc[cluster].tolist()
                values += values[:1]
                axes[4].plot(angles, values, linewidth=2, 
                            label=f'Cluster {cluster}', marker='o')
                axes[4].fill(angles, values, alpha=0.1)
            
            axes[4].set_xticks(angles[:-1])
            axes[4].set_xticklabels(categories)
            axes[4].set_title('Radar Chart of Cluster Profiles')
            axes[4].legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            axes[4].grid(True, alpha=0.3)
        
        # 6. Heatmap of Cluster Means
        if len(feature_cols) >= 2:
            heatmap_data = pd.DataFrame()
            for cluster in range(n_clusters):
                cluster_data = data[data[cluster_col] == cluster]
                cluster_means = {}
                for col in feature_cols[:5]:
                    if col in cluster_data.columns:
                        cluster_means[col] = cluster_data[col].mean()
                heatmap_data = pd.concat([heatmap_data, pd.DataFrame(cluster_means, index=[f'Cluster {cluster}'])])
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                           ax=axes[5], cbar_kws={'label': 'Mean Value'})
                axes[5].set_title('Cluster Feature Means')
                axes[5].set_xlabel('Cluster')
            else:
                axes[5].text(0.5, 0.5, 'Insufficient data for heatmap',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[5].transAxes, fontsize=12)
                axes[5].set_title('Heatmap (Not Available)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['clustering_analysis'] = fig
        return fig, cluster_stats
    
    def plot_control_point_comparison(self,
                                     data: pd.DataFrame,
                                     control_point_col: str = 'control_point',
                                     target_col: str = 'total',
                                     title: str = 'Control Point Traffic Comparison',
                                     figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot comparison of traffic across different control points.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Traffic data with control point information
        control_point_col : str
            Column name for control points
        target_col : str
            Column name for traffic volume
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Calculate statistics by control point
        cp_stats = data.groupby(control_point_col)[target_col].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).sort_values('mean', ascending=False)
        
        cp_stats['traffic_share'] = (cp_stats['mean'] / cp_stats['mean'].sum()) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Bar chart of mean traffic
        top_n = min(15, len(cp_stats))
        top_cps = cp_stats.head(top_n)
        
        bars = axes[0].barh(range(top_n), top_cps['mean'].values[::-1], 
                           color='skyblue', edgecolor='black')
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(top_cps.index[::-1])
        axes[0].set_xlabel(f'Average {target_col}')
        axes[0].set_title(f'Top {top_n} Control Points by Average Traffic')
        
        # Add values on bars
        for i, (bar, mean_val) in enumerate(zip(bars, top_cps['mean'].values[::-1])):
            axes[0].text(mean_val, i, f'{mean_val:,.0f}', 
                        ha='left', va='center', fontweight='bold')
        
        # 2. Traffic share pie chart
        top_share_cps = cp_stats.head(8)
        other_share = 100 - top_share_cps['traffic_share'].sum()
        
        if other_share > 0:
            share_data = pd.concat([
                top_share_cps['traffic_share'],
                pd.Series([other_share], index=['Others'])
            ])
            labels = list(top_share_cps.index) + ['Others']
        else:
            share_data = top_share_cps['traffic_share']
            labels = list(top_share_cps.index)
        
        wedges, texts, autotexts = axes[1].pie(share_data, labels=labels, autopct='%1.1f%%',
                                              startangle=90, pctdistance=0.85)
        axes[1].set_title('Traffic Share Distribution')
        
        # 3. Traffic variability (mean ± std)
        error_bars = axes[2].errorbar(range(len(cp_stats)), cp_stats['mean'], 
                                     yerr=cp_stats['std'], fmt='o', capsize=5,
                                     color='red', ecolor='gray', elinewidth=2)
        axes[2].set_xlabel('Control Point (Ranked)')
        axes[2].set_ylabel(f'Average {target_col} ± Std Dev')
        axes[2].set_title('Traffic Variability by Control Point')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Traffic distribution by control point (box plot)
        # Limit to top 10 for readability
        top_cps_box = cp_stats.head(10).index.tolist()
        box_data = []
        box_labels = []
        
        for cp in top_cps_box:
            cp_data = data[data[control_point_col] == cp][target_col]
            if len(cp_data) > 0:
                box_data.append(cp_data)
                box_labels.append(cp)
        
        if box_data:
            box_plot = axes[3].boxplot(box_data, vert=False, patch_artist=True, labels=box_labels)
            for patch in box_plot['boxes']:
                patch.set_facecolor('lightgreen')
            axes[3].set_xlabel(target_col)
            axes[3].set_title('Traffic Distribution (Top 10 Control Points)')
        else:
            axes[3].text(0.5, 0.5, 'Insufficient data for box plot',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[3].transAxes, fontsize=12)
            axes[3].set_title('Box Plot (Not Available)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['control_point_comparison'] = fig
        return fig
    
    def plot_control_point_time_series(self,
                                      data: pd.DataFrame,
                                      control_point_col: str = 'control_point',
                                      target_col: str = 'total',
                                      time_col: str = 'date',
                                      control_points: List[str] = None,
                                      title: str = 'Control Point Traffic Time Series',
                                      figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Plot time series comparison for multiple control points.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Traffic data
        control_point_col : str
            Column name for control points
        target_col : str
            Column name for traffic volume
        time_col : str
            Column name for timestamps
        control_points : list, optional
            List of control points to plot
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        if control_points is None:
            # Get top 5 control points by mean traffic
            cp_means = data.groupby(control_point_col)[target_col].mean()
            control_points = cp_means.nlargest(5).index.tolist()
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # 1. Time series plot
        colors = plt.cm.Set2(np.linspace(0, 1, len(control_points)))
        
        for cp, color in zip(control_points, colors):
            cp_data = data[data[control_point_col] == cp].copy()
            cp_data = cp_data.sort_values(time_col)
            
            if len(cp_data) > 0:
                axes[0].plot(cp_data[time_col], cp_data[target_col], 
                            label=cp, color=color, linewidth=2, alpha=0.8)
        
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(f'{target_col}')
        axes[0].set_title('Traffic Time Series by Control Point')
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # 2. Relative comparison (normalized)
        for cp, color in zip(control_points, colors):
            cp_data = data[data[control_point_col] == cp].copy()
            cp_data = cp_data.sort_values(time_col)
            
            if len(cp_data) > 0:
                # Normalize to 0-1 range for comparison
                normalized = (cp_data[target_col] - cp_data[target_col].min()) / \
                            (cp_data[target_col].max() - cp_data[target_col].min())
                axes[1].plot(cp_data[time_col], normalized, 
                            color=color, linewidth=1, alpha=0.6)
        
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Normalized Traffic')
        axes[1].set_title('Normalized Comparison')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.1])
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['control_point_time_series'] = fig
        return fig
    
    def plot_traffic_heatmap(self,
                            data: pd.DataFrame,
                            control_point_col: str = 'control_point',
                            target_col: str = 'total',
                            time_col: str = 'date',
                            title: str = 'Traffic Heatmap by Control Point and Time',
                            figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot heatmap of traffic by control point and time dimension.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Traffic data
        control_point_col : str
            Column name for control points
        target_col : str
            Column name for traffic volume
        time_col : str
            Column name for timestamps
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # Extract time components
        data_copy = data.copy()
        data_copy['hour'] = data_copy[time_col].dt.hour
        data_copy['day_of_week'] = data_copy[time_col].dt.dayofweek
        data_copy['month'] = data_copy[time_col].dt.month
        
        # Get top control points
        top_cps = data_copy[control_point_col].value_counts().head(10).index.tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Heatmap by hour and control point
        if 'hour' in data_copy.columns:
            hourly_cp = data_copy.pivot_table(
                values=target_col,
                index=control_point_col,
                columns='hour',
                aggfunc='mean'
            ).loc[top_cps]
            
            if not hourly_cp.empty:
                sns.heatmap(hourly_cp, cmap='YlOrRd', ax=axes[0], 
                           cbar_kws={'label': f'Average {target_col}'})
                axes[0].set_xlabel('Hour of Day')
                axes[0].set_ylabel('Control Point')
                axes[0].set_title('Traffic by Hour and Control Point')
            else:
                axes[0].text(0.5, 0.5, 'Insufficient data for heatmap',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[0].transAxes, fontsize=12)
                axes[0].set_title('Hourly Heatmap (Not Available)')
        
        # 2. Heatmap by day of week and control point
        if 'day_of_week' in data_copy.columns:
            daily_cp = data_copy.pivot_table(
                values=target_col,
                index=control_point_col,
                columns='day_of_week',
                aggfunc='mean'
            ).loc[top_cps]
            
            if not daily_cp.empty:
                sns.heatmap(daily_cp, cmap='YlOrRd', ax=axes[1],
                           cbar_kws={'label': f'Average {target_col}'})
                axes[1].set_xlabel('Day of Week')
                axes[1].set_ylabel('Control Point')
                axes[1].set_title('Traffic by Day of Week and Control Point')
                axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            else:
                axes[1].text(0.5, 0.5, 'Insufficient data for heatmap',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title('Daily Heatmap (Not Available)')
        
        # 3. Heatmap by month and control point
        if 'month' in data_copy.columns:
            monthly_cp = data_copy.pivot_table(
                values=target_col,
                index=control_point_col,
                columns='month',
                aggfunc='mean'
            ).loc[top_cps]
            
            if not monthly_cp.empty:
                sns.heatmap(monthly_cp, cmap='YlOrRd', ax=axes[2],
                           cbar_kws={'label': f'Average {target_col}'})
                axes[2].set_xlabel('Month')
                axes[2].set_ylabel('Control Point')
                axes[2].set_title('Traffic by Month and Control Point')
                axes[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            else:
                axes[2].text(0.5, 0.5, 'Insufficient data for heatmap',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[2].transAxes, fontsize=12)
                axes[2].set_title('Monthly Heatmap (Not Available)')
        
        # 4. Control point traffic composition
        if all(col in data_copy.columns for col in ['hk_residents', 'mainland_visitors', 'other_visitors']):
            composition = data_copy.groupby(control_point_col)[['hk_residents', 'mainland_visitors', 'other_visitors']].mean()
            composition = composition.loc[top_cps]
            
            # Normalize to percentages
            composition_pct = composition.div(composition.sum(axis=1), axis=0) * 100
            
            if not composition_pct.empty:
                composition_pct.plot(kind='bar', stacked=True, ax=axes[3],
                                   color=['lightblue', 'lightgreen', 'lightcoral'])
                axes[3].set_xlabel('Control Point')
                axes[3].set_ylabel('Percentage (%)')
                axes[3].set_title('Passenger Composition by Control Point')
                axes[3].legend(title='Passenger Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[3].tick_params(axis='x', rotation=45)
            else:
                axes[3].text(0.5, 0.5, 'Insufficient data for composition',
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[3].transAxes, fontsize=12)
                axes[3].set_title('Composition (Not Available)')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['traffic_heatmap'] = fig
        return fig
    
    def plot_control_point_boxplots(self,
                                   data: pd.DataFrame,
                                   control_point_col: str = 'control_point',
                                   target_col: str = 'total',
                                   control_points: List[str] = None,
                                   title: str = 'Control Point Traffic Distribution',
                                   figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot box plots comparing traffic distribution across control points.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Traffic data
        control_point_col : str
            Column name for control points
        target_col : str
            Column name for traffic volume
        control_points : list, optional
            List of control points to include
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if control_points is None:
            # Get top 10 control points by traffic volume
            cp_means = data.groupby(control_point_col)[target_col].mean()
            control_points = cp_means.nlargest(10).index.tolist()
        
        # Filter data for selected control points
        plot_data = data[data[control_point_col].isin(control_points)].copy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        box_data = []
        box_labels = []
        
        for cp in control_points:
            cp_data = plot_data[plot_data[control_point_col] == cp][target_col]
            if len(cp_data) > 0:
                box_data.append(cp_data)
                box_labels.append(cp)
        
        if box_data:
            box_plot = ax.boxplot(box_data, vert=True, patch_artist=True, labels=box_labels)
            
            # Color boxes differently
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add mean markers
            means = [np.mean(data) for data in box_data]
            ax.scatter(range(1, len(means) + 1), means, color='red', zorder=3, 
                      label='Mean', s=100, marker='D')
            
            ax.set_xlabel('Control Point')
            ax.set_ylabel(target_col)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for box plots',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
        
        plt.tight_layout()
        
        self.figures['control_point_boxplots'] = fig
        return fig
    
    def plot_traffic_distribution(self,
                                 comparison_df: pd.DataFrame,
                                 title: str = 'Traffic Distribution Across Control Points',
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot traffic distribution visualization.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame with control point comparison statistics
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by traffic share
        if 'Traffic Share (%)' in comparison_df.columns:
            sorted_df = comparison_df.sort_values('Traffic Share (%)', ascending=False)
            
            # Create waterfall chart
            cumulative = 0
            for idx, (cp, row) in enumerate(sorted_df.iterrows()):
                share = row['Traffic Share (%)']
                ax.bar(idx, share, bottom=cumulative, 
                      color=plt.cm.viridis(idx/len(sorted_df)), 
                      edgecolor='black', label=cp)
                
                # Add label in the middle of each bar
                ax.text(idx, cumulative + share/2, f'{cp}\n{share:.1f}%',
                       ha='center', va='center', fontsize=9, color='white')
                
                cumulative += share
            
            ax.set_xlabel('Control Points (Cumulative)')
            ax.set_ylabel('Traffic Share (%)')
            ax.set_title(title)
            ax.set_xticks([])  # Remove x-ticks since we have labels in bars
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No traffic share data available',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
        
        plt.tight_layout()
        
        self.figures['traffic_distribution'] = fig
        return fig
    
    def plot_model_comparison(self,
                             results: Dict,
                             title: str = 'Model Performance Comparison',
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot comparison of different model performances.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing model results
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Extract model metrics
        model_data = {}
        model_types = {}
        
        for model_name, result in results.items():
            if model_name == 'control_point_comparison':
                continue
                
            if 'linear' in model_name:
                model_types[model_name] = 'Regression'
                model_data[model_name] = {
                    'R²': result['metrics']['test']['R2'],
                    'RMSE': result['metrics']['test']['RMSE'],
                    'MAE': result['metrics']['test']['MAE']
                }
            elif 'logistic' in model_name or 'svm' in model_name:
                model_types[model_name] = 'Classification'
                model_data[model_name] = {
                    'Accuracy': result['metrics'].get('accuracy_test', 0),
                    'Precision': result['metrics'].get('precision', 0),
                    'Recall': result['metrics'].get('recall', 0),
                    'F1-Score': result['metrics'].get('f1_score', 0),
                    'ROC AUC': result['metrics'].get('roc_auc', 0)
                }
            elif 'kmeans' in model_name:
                model_types[model_name] = 'Clustering'
                model_data[model_name] = {
                    'Silhouette': result['metrics'].get('silhouette_score', 0),
                    'Inertia': result['metrics'].get('inertia', 0)
                }
        
        if not model_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No model results available for comparison',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return fig
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Bar chart of main metrics by model type
        regression_models = [m for m, t in model_types.items() if t == 'Regression']
        classification_models = [m for m, t in model_types.items() if t == 'Classification']
        clustering_models = [m for m, t in model_types.items() if t == 'Clustering']
        
        # Plot regression metrics
        if regression_models:
            reg_metrics = ['R²', 'RMSE', 'MAE']
            x = np.arange(len(reg_metrics))
            width = 0.8 / len(regression_models)
            
            for i, model in enumerate(regression_models):
                values = [model_data[model].get(metric, 0) for metric in reg_metrics]
                axes[0].bar(x + i*width - width*len(regression_models)/2 + width/2, 
                           values, width, label=model)
            
            axes[0].set_xlabel('Metrics')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Regression Models Comparison')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(reg_metrics)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot classification metrics
        if classification_models:
            class_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            x = np.arange(len(class_metrics))
            width = 0.8 / len(classification_models)
            
            for i, model in enumerate(classification_models):
                values = [model_data[model].get(metric, 0) for metric in class_metrics]
                axes[1].bar(x + i*width - width*len(classification_models)/2 + width/2, 
                           values, width, label=model)
            
            axes[1].set_xlabel('Metrics')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Classification Models Comparison')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(class_metrics, rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].set_ylim([0, 1.1])
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self.figures['model_comparison'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = 'reports/figures'):
        """Save all generated figures to files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for fig_name, fig in self.figures.items():
            filename = f"{output_dir}/{fig_name}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
    
    def show_all_figures(self):
        """Display all generated figures."""
        for fig_name, fig in self.figures.items():
            plt.figure(fig.number)
            plt.show()


# Helper functions for standalone use
def create_summary_report(data: pd.DataFrame, 
                         target_col: str = 'total',
                         date_col: str = 'date',
                         output_dir: str = 'reports') -> TrafficVisualizer:
    """
    Create a comprehensive summary report of traffic data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Traffic data
    target_col : str
        Target column name
    date_col : str
        Date column name
    output_dir : str
        Output directory for saving figures
        
    Returns:
    --------
    TrafficVisualizer
        Visualizer object with generated figures
    """
    viz = TrafficVisualizer(data)
    
    print("Generating comprehensive traffic analysis report...")
    
    # Generate all plots
    viz.plot_time_series(date_col=date_col, target_col=target_col)
    viz.plot_distribution(target_col=target_col)
    viz.plot_correlation_heatmap()
    
    if date_col in data.columns:
        viz.plot_seasonality(date_col=date_col, target_col=target_col)
    
    # Save all figures
    viz.save_all_figures(output_dir)
    
    print(f"Report generated and saved to {output_dir}/")
    return viz


if __name__ == "__main__":
    # Demo of the visualization module
    print("Traffic Visualization Module Demo")
    print("=" * 60)
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'total': np.random.normal(100000, 20000, len(dates)).clip(50000, 150000),
        'hk_residents': np.random.normal(60000, 10000, len(dates)).clip(30000, 90000),
        'mainland_visitors': np.random.normal(30000, 8000, len(dates)).clip(10000, 50000),
        'other_visitors': np.random.normal(10000, 3000, len(dates)).clip(5000, 20000),
        'control_point': np.random.choice(['Airport', 'Land Border', 'Sea Port', 'Rail Station'], 
                                         len(dates), p=[0.4, 0.3, 0.2, 0.1])
    })
    
    # Add some seasonality
    sample_data['total'] = sample_data['total'] * (1 + 0.3 * np.sin(2 * np.pi * sample_data['date'].dt.dayofyear / 365))
    
    print(f"Created sample data with {len(sample_data)} records")
    print(f"Columns: {list(sample_data.columns)}")
    
    # Create visualizer
    viz = TrafficVisualizer(sample_data)
    
    # Generate sample visualizations
    print("\nGenerating sample visualizations...")
    
    # Time series
    fig1 = viz.plot_time_series(title='Sample Traffic Time Series')
    plt.show()
    
    # Distribution
    fig2 = viz.plot_distribution(title='Sample Traffic Distribution')
    plt.show()
    
    # Seasonality
    fig3 = viz.plot_seasonality(title='Sample Seasonality Analysis')
    plt.show()
    
    # Control point comparison - FIXED: Pass the data parameter
    fig4 = viz.plot_control_point_comparison(
        data=sample_data,  # Add this parameter
        title='Sample Control Point Comparison'
    )
    plt.show()
    
    # Also demonstrate the control point time series
    fig5 = viz.plot_control_point_time_series(
        data=sample_data,
        title='Sample Control Point Time Series'
    )
    plt.show()
    
    print("\n✅ Visualization demo completed successfully!")