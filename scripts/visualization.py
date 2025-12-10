"""
Visualization functions for traffic analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class TrafficVisualizer:
    """Create visualizations for traffic analysis."""
    
    def __init__(self, df):
        self.df = df
        self.set_style()
        
    def set_style(self):
        """Set matplotlib style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_time_series(self, date_col='date', value_col='total', 
                        title='Daily Passenger Traffic Over Time'):
        """Plot time series of passenger traffic."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        if date_col in self.df.columns and value_col in self.df.columns:
            ax.plot(self.df[date_col], self.df[value_col], 
                   linewidth=1, alpha=0.7, label='Daily Traffic')
            
            rolling_mean = self.df[value_col].rolling(window=30).mean()
            ax.plot(self.df[date_col], rolling_mean, 
                   linewidth=2, color='red', label='30-day Moving Average')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Passenger Count', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
        return fig
    
    def plot_regression_results(self, y_true, y_pred, title='Linear Regression: Actual vs Predicted'):
        """Plot actual vs predicted values for regression."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Add perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        min_val = min(y_true.min(), y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', label='Perfect Prediction', linewidth=2)
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² text
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_classification_results(self, y_true, y_pred, model_name='Model'):
        """Plot confusion matrix for classification."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Low Traffic', 'High Traffic'],
                   yticklabels=['Low Traffic', 'High Traffic'])
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_clusters(self, features, clusters, cluster_col='cluster', value_col='total', title='K-means Clustering Results'):
        """Visualize clustering results."""
        fig = plt.figure(figsize=(15, 5))
        
        # 2D scatter plot (first two features)
        ax1 = fig.add_subplot(131)
        scatter = ax1.scatter(self.df[features[0]], self.df[features[1]], 
                            c=clusters, cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(features[1])
        ax1.set_title('Feature Space Clustering')
        plt.colorbar(scatter, ax=ax1)
        
        # Time series with clusters
        if cluster_col in self.df.columns and 'date' in self.df.columns and value_col in self.df.columns:
            ax2 = fig.add_subplot(132)
            for cluster in sorted(self.df[cluster_col].unique()):
                cluster_data = self.df[self.df[cluster_col] == cluster]
                ax2.scatter(cluster_data['date'], cluster_data[value_col], 
                          alpha=0.6, label=f'Cluster {cluster}', s=10)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Passenger Count')
            ax2.set_title('Clusters Over Time')
            ax2.legend()
            plt.xticks(rotation=45)
        
        # Cluster distribution
        ax3 = fig.add_subplot(133)
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
        ax3.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Count')
        ax3.set_title('Cluster Distribution')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, coefficients, feature_names, title='Feature Importance'):
        """Plot feature importance from linear models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feat_df = pd.DataFrame({
            'feature': feature_names,
            'coef': coefficients
        })
        feat_df['abs_coef'] = feat_df['coef'].abs()
        feat_df = feat_df.sort_values('abs_coef', ascending=True)
        
        colors = ['red' if x < 0 else 'blue' for x in feat_df['coef']]
        bars = ax.barh(feat_df['feature'], feat_df['abs_coef'], color=colors)
        
        ax.set_xlabel('Absolute Coefficient Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plot(self):
        """Create interactive Plotly visualization."""
        if 'date' in self.df.columns and 'passenger_count' in self.df.columns:
            fig = px.line(self.df, x='date', y='passenger_count',
                         title='Interactive Passenger Traffic Timeline',
                         labels={'passenger_count': 'Passenger Count', 'date': 'Date'})
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Passenger Count",
                hovermode='x unified',
                template='plotly_dark'
            )
            
            return fig