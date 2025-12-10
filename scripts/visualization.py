#!/usr/bin/env python3
"""
Enhanced visualization module for passenger traffic analysis.
Combines visualization capabilities with automatic plot saving.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TrafficVisualizer:
    """Enhanced visualization class for traffic data analysis and model results."""
    
    def __init__(self, data: pd.DataFrame = None, output_dir: str = 'reports/figures'):
        """
        Initialize TrafficVisualizer with optional data.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing traffic data
        output_dir : str
            Directory to save generated plots
        """
        self.data = data
        self.figures = {}
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def set_data(self, data: pd.DataFrame):
        """Set or update the data for visualization."""
        self.data = data.copy()
    
    def save_figure(self, fig: plt.Figure, name: str, formats: List[str] = ['png', 'pdf']):
        """
        Save figure in multiple formats.
        
        Parameters:
        -----------
        fig : plt.Figure
            Figure to save
        name : str
            Base name for the file
        formats : list
            List of formats to save (png, pdf, svg, etc.)
        """
        saved_files = []
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{name}_{self.timestamp}.{fmt}")
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            saved_files.append(filepath)
        return saved_files
    
    def plot_time_series(self, 
                        date_col: str = 'date',
                        target_col: str = 'total',
                        title: str = 'Passenger Traffic Over Time',
                        figsize: Tuple[int, int] = (15, 6),
                        show_rolling: bool = True,
                        rolling_window: int = 7,
                        save: bool = False) -> plt.Figure:
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
        save : bool
            Whether to save the figure
            
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
        
        if save:
            self.save_figure(fig, 'time_series')
        
        return fig
    
    def plot_distribution(self, 
                         target_col: str = 'total',
                         bins: int = 50,
                         title: str = 'Traffic Distribution',
                         figsize: Tuple[int, int] = (12, 5),
                         save: bool = False) -> plt.Figure:
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
        save : bool
            Whether to save the figure
            
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
        
        if save:
            self.save_figure(fig, 'distribution')
        
        return fig
    
    def plot_correlation_heatmap(self,
                                columns: List[str] = None,
                                title: str = 'Feature Correlation Heatmap',
                                figsize: Tuple[int, int] = (10, 8),
                                annot: bool = True,
                                save: bool = False) -> plt.Figure:
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
        save : bool
            Whether to save the figure
            
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
        
        if save:
            self.save_figure(fig, 'correlation_heatmap')
        
        return fig
    
    def plot_seasonality(self,
                        date_col: str = 'date',
                        target_col: str = 'total',
                        title: str = 'Traffic Seasonality Analysis',
                        figsize: Tuple[int, int] = (15, 10),
                        save: bool = False) -> plt.Figure:
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
        save : bool
            Whether to save the figure
            
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
        
        if save:
            self.save_figure(fig, 'seasonality')
        
        return fig
    
    def plot_control_point_comparison(self,
                                     control_point_col: str = 'control_point',
                                     target_col: str = 'total',
                                     title: str = 'Control Point Traffic Comparison',
                                     figsize: Tuple[int, int] = (14, 8),
                                     save: bool = False) -> plt.Figure:
        """
        Plot comparison of traffic across different control points.
        
        Parameters:
        -----------
        control_point_col : str
            Column name for control points
        target_col : str
            Column name for traffic volume
        title : str
            Plot title
        figsize : tuple
            Figure size
        save : bool
            Whether to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        if control_point_col not in self.data.columns:
            raise ValueError(f"Column '{control_point_col}' not found in data.")
        
        # Calculate statistics by control point
        cp_stats = self.data.groupby(control_point_col)[target_col].agg([
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
            cp_data = self.data[self.data[control_point_col] == cp][target_col]
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
        
        if save:
            self.save_figure(fig, 'control_point_comparison')
        
        return fig
    
    def plot_control_point_time_series(self,
                                      control_point_col: str = 'control_point',
                                      target_col: str = 'total',
                                      time_col: str = 'date',
                                      control_points: List[str] = None,
                                      title: str = 'Control Point Traffic Time Series',
                                      figsize: Tuple[int, int] = (15, 8),
                                      save: bool = False) -> plt.Figure:
        """
        Plot time series comparison for multiple control points.
        
        Parameters:
        -----------
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
        save : bool
            Whether to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        if control_point_col not in self.data.columns:
            raise ValueError(f"Column '{control_point_col}' not found in data.")
        
        data = self.data.copy()
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
        
        if save:
            self.save_figure(fig, 'control_point_time_series')
        
        return fig
    
    def plot_traffic_heatmap(self,
                            control_point_col: str = 'control_point',
                            target_col: str = 'total',
                            time_col: str = 'date',
                            title: str = 'Traffic Heatmap by Control Point and Time',
                            figsize: Tuple[int, int] = (14, 10),
                            save: bool = False) -> plt.Figure:
        """
        Plot heatmap of traffic by control point and time dimension.
        
        Parameters:
        -----------
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
        save : bool
            Whether to save the figure
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        if control_point_col not in self.data.columns:
            raise ValueError(f"Column '{control_point_col}' not found in data.")
        
        data = self.data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # Extract time components
        data['hour'] = data[time_col].dt.hour
        data['day_of_week'] = data[time_col].dt.dayofweek
        data['month'] = data[time_col].dt.month
        
        # Get top control points
        top_cps = data[control_point_col].value_counts().head(10).index.tolist()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # 1. Heatmap by hour and control point
        if 'hour' in data.columns:
            hourly_cp = data.pivot_table(
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
        if 'day_of_week' in data.columns:
            daily_cp = data.pivot_table(
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
        if 'month' in data.columns:
            monthly_cp = data.pivot_table(
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
        passenger_cols = ['hk_residents', 'mainland_visitors', 'other_visitors']
        if all(col in data.columns for col in passenger_cols):
            composition = data.groupby(control_point_col)[passenger_cols].mean()
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
        
        if save:
            self.save_figure(fig, 'traffic_heatmap')
        
        return fig
    
    def plot_regression_analysis(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                model_name: str = "Model",
                                figsize: Tuple[int, int] = (15, 5),
                                save: bool = False) -> plt.Figure:
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
        save : bool
            Whether to save the figure
            
        Returns:
        --------
        plt.Figure
            Figure object
        """
        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
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
        
        fig_name = f'regression_analysis_{model_name}'
        self.figures[fig_name] = fig
        
        if save:
            self.save_figure(fig, fig_name)
        
        return fig
    
    def generate_all_plots(self, 
                          target_col: str = 'total',
                          date_col: str = 'date',
                          control_point_col: str = 'control_point'):
        """
        Generate all standard plots for the current dataset.
        
        Parameters:
        -----------
        target_col : str
            Target column name
        date_col : str
            Date column name
        control_point_col : str
            Control point column name
            
        Returns:
        --------
        dict
            Dictionary of saved file paths
        """
        print("📊 Generating all standard plots...")
        
        saved_files = {}
        
        # Basic plots
        plots = [
            ('time_series', self.plot_time_series, 
             {'date_col': date_col, 'target_col': target_col, 'title': 'Passenger Traffic Over Time'}),
            ('distribution', self.plot_distribution,
             {'target_col': target_col, 'title': 'Traffic Distribution'}),
            ('correlation', self.plot_correlation_heatmap,
             {'title': 'Feature Correlations'}),
            ('seasonality', self.plot_seasonality,
             {'date_col': date_col, 'target_col': target_col, 'title': 'Seasonal Patterns'}),
        ]
        
        # Control point plots if column exists
        if control_point_col in self.data.columns:
            plots.extend([
                ('control_point_comparison', self.plot_control_point_comparison,
                 {'control_point_col': control_point_col, 'target_col': target_col, 
                  'title': 'Control Point Comparison'}),
                ('control_point_time_series', self.plot_control_point_time_series,
                 {'control_point_col': control_point_col, 'target_col': target_col, 
                  'time_col': date_col, 'title': 'Control Point Time Series'}),
                ('traffic_heatmap', self.plot_traffic_heatmap,
                 {'control_point_col': control_point_col, 'target_col': target_col,
                  'time_col': date_col, 'title': 'Traffic Heatmap'})
            ])
        
        # Generate all plots
        for plot_name, plot_func, kwargs in plots:
            try:
                print(f"🖼️  Generating {plot_name}...")
                fig = plot_func(**kwargs)
                saved_files[plot_name] = self.save_figure(fig, plot_name, formats=['png', 'pdf'])
                print(f"   ✅ Saved: {plot_name}")
                plt.close(fig)
            except Exception as e:
                print(f"   ❌ Failed to generate {plot_name}: {e}")
        
        # Create summary report
        summary_path = self._create_summary_report(saved_files)
        saved_files['summary'] = summary_path
        
        print(f"\n✅ All plots generated and saved to: {self.output_dir}")
        print(f"📋 Summary saved to: {summary_path}")
        
        return saved_files
    
    def _create_summary_report(self, saved_files: Dict) -> str:
        """
        Create a summary report of generated plots.
        
        Parameters:
        -----------
        saved_files : dict
            Dictionary of saved file paths
            
        Returns:
        --------
        str
            Path to the summary file
        """
        summary_path = os.path.join(self.output_dir, f'plot_summary_{self.timestamp}.md')
        
        with open(summary_path, 'w') as f:
            f.write(f"# Traffic Analysis Plot Generation Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Data shape:** {self.data.shape}\n")
            f.write(f"**Output directory:** {self.output_dir}\n\n")
            
            if hasattr(self, 'data_path'):
                f.write(f"**Data source:** {self.data_path}\n\n")
            
            f.write("## Generated Plots\n\n")
            for plot_name, files in saved_files.items():
                if plot_name != 'summary':
                    f.write(f"### {plot_name.replace('_', ' ').title()}\n")
                    for file_path in files:
                        f.write(f"- `{os.path.basename(file_path)}`\n")
                    f.write("\n")
            
            f.write("## Data Statistics\n\n")
            f.write(f"- **Total records:** {len(self.data):,}\n")
            f.write(f"- **Date range:** {self.data['date'].min().date()} to {self.data['date'].max().date()}\n")
            
            if 'total' in self.data.columns:
                f.write(f"- **Total passengers:** {self.data['total'].sum():,}\n")
                f.write(f"- **Average daily:** {self.data['total'].mean():,.0f}\n")
                f.write(f"- **Max daily:** {self.data['total'].max():,.0f}\n")
                f.write(f"- **Min daily:** {self.data['total'].min():,.0f}\n")
        
        return summary_path
    
    def save_all_figures(self):
        """Save all generated figures to files."""
        for fig_name, fig in self.figures.items():
            self.save_figure(fig, fig_name, formats=['png', 'pdf'])
    
    def show_all_figures(self):
        """Display all generated figures."""
        for fig_name, fig in self.figures.items():
            plt.figure(fig.number)
            plt.show()


def create_visualization_report(data_path: str = 'data/processed/passenger_data_daily.csv',
                               output_dir: str = 'reports/figures',
                               target_col: str = 'total',
                               date_col: str = 'date',
                               control_point_col: str = 'control_point') -> TrafficVisualizer:
    """
    Create a comprehensive visualization report from data file.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
    output_dir : str
        Output directory for plots
    target_col : str
        Target column name
    date_col : str
        Date column name
    control_point_col : str
        Control point column name
        
    Returns:
    --------
    TrafficVisualizer
        Visualizer object with generated figures
    """
    print(f"📊 Loading data from {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=[date_col])
    
    # Create visualizer
    viz = TrafficVisualizer(df, output_dir)
    viz.data_path = data_path  # Store for reference
    
    # Generate all plots
    viz.generate_all_plots(target_col=target_col, date_col=date_col, 
                          control_point_col=control_point_col)
    
    return viz


def main():
    """
    Main function for command line usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate traffic analysis visualizations')
    parser.add_argument('--data', type=str, default='data/processed/passenger_data_daily.csv',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='reports/figures',
                       help='Output directory for plots')
    parser.add_argument('--target', type=str, default='total',
                       help='Target column name')
    parser.add_argument('--date', type=str, default='date',
                       help='Date column name')
    parser.add_argument('--control-point', type=str, default='control_point',
                       help='Control point column name')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample data')
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo with sample data
        print("🚀 Running demo with sample data...")
        
        # Create sample data
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
        
        # Add seasonality
        sample_data['total'] = sample_data['total'] * (1 + 0.3 * np.sin(2 * np.pi * sample_data['date'].dt.dayofyear / 365))
        
        print(f"📊 Created sample data with {len(sample_data)} records")
        
        # Create visualizer with sample data
        viz = TrafficVisualizer(sample_data, args.output)
        viz.data_path = 'sample_data'
        
        # Generate plots
        viz.generate_all_plots()
        
        print("\n✅ Demo completed successfully!")
        
        # Show sample plots
        print("\n📈 Displaying sample plots...")
        plt.show()
        
    else:
        # Generate report from data file
        viz = create_visualization_report(
            data_path=args.data,
            output_dir=args.output,
            target_col=args.target,
            date_col=args.date,
            control_point_col=args.control_point
        )
        
        print(f"\n✅ Report generated successfully!")
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Data processed: {len(viz.data):,} records")


if __name__ == "__main__":
    main()