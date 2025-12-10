"""
Enhanced Machine Learning models for passenger traffic analysis with comprehensive visualization.
"""
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score, precision_score, recall_score, f1_score,
                             roc_curve, auc)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

class TrafficModels:
    """Enhanced ML models for traffic analysis with visualization capabilities."""
    
    def __init__(self, df: pd.DataFrame, features: list, target: str, problem_type='regression'):
        self.df = df.reset_index(drop=True).copy()
        self.features = features
        self.target = target
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.visualizations = {}
        self.data = {}
        
        # Store feature names for visualization
        self.feature_names = features
        
        # Check if visualization is available
        self.visualization_available = self._check_visualization()
    
    def _check_visualization(self):
        """Check if visualization module is available."""
        try:
            from visualization import TrafficVisualizer
            return True
        except ImportError:
            print("⚠️  Visualization module not available. Plots will be skipped.")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42, scale=True):
        """Prepare X/y, scaling and train-test split with enhanced error handling."""
        print("=" * 60)
        print("Preparing Data for Machine Learning")
        print("=" * 60)
        
        X = self.df[self.features].copy()
        
        # Convert categorical columns to numeric
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].nunique() <= 10:  # For low cardinality categoricals
                X[col] = pd.factorize(X[col])[0]
            else:  # For high cardinality, use one-hot or drop
                print(f"  Dropping high cardinality categorical column: {col}")
                X = X.drop(columns=[col])
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        y = self.df[self.target].copy()
        
        # Prepare target based on problem type
        if self.problem_type == 'classification':
            if y.nunique() > 20:  # Likely continuous, need to bin
                print(f"  Target has {y.nunique()} unique values, converting to binary...")
                median_val = y.median()
                y = (y > median_val).astype(int)
                print(f"  Created binary target: 0=below median, 1=above median")
            elif y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('NA'))
                self.label_encoder = le
                print(f"  Encoded categorical target with {len(le.classes_)} classes")
        else:  # regression
            y = pd.to_numeric(y, errors='coerce').fillna(y.median())
            print(f"  Target range: {y.min():.0f} to {y.max():.0f}, Mean: {y.mean():.0f}")
        
        # Scale features if requested
        if scale:
            X_scaled = self.scaler.fit_transform(X)
            print(f"  Features scaled: Mean={X_scaled.mean():.2f}, Std={X_scaled.std():.2f}")
        else:
            X_scaled = X.values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        self.data.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.features,
            'X_original': X,
            'y_original': y
        })
        
        print(f"\n📊 Data Summary:")
        print(f"  Original shape: X={X.shape}, y={y.shape}")
        print(f"  Training set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
        print(f"  Testing set:  {len(X_test)} samples ({100*test_size:.0f}%)")
        print(f"  Features used: {len(self.features)}")
        print("=" * 60)
        
        return self.data
    
    def linear_regression(self, show_plots=True, save_plots=False):
        """Enhanced Linear Regression with comprehensive analysis."""
        print("\n" + "=" * 60)
        print("Linear Regression Analysis")
        print("=" * 60)
        
        if self.problem_type != 'regression':
            print("⚠️  Switching to regression mode...")
            self.problem_type = 'regression'
        
        data = self.data if self.data else self.prepare_data()
        
        # Train model
        model = LinearRegression()
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred_train = model.predict(data['X_train'])
        y_pred_test = model.predict(data['X_test'])
        
        # Calculate metrics
        metrics_train = {
            'MSE': mean_squared_error(data['y_train'], y_pred_train),
            'RMSE': np.sqrt(mean_squared_error(data['y_train'], y_pred_train)),
            'MAE': mean_absolute_error(data['y_train'], y_pred_train),
            'R2': r2_score(data['y_train'], y_pred_train),
            'MAPE': np.mean(np.abs((data['y_train'] - y_pred_train) / (data['y_train'] + 1e-10))) * 100
        }
        
        metrics_test = {
            'MSE': mean_squared_error(data['y_test'], y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(data['y_test'], y_pred_test)),
            'MAE': mean_absolute_error(data['y_test'], y_pred_test),
            'R2': r2_score(data['y_test'], y_pred_test),
            'MAPE': np.mean(np.abs((data['y_test'] - y_pred_test) / (data['y_test'] + 1e-10))) * 100
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, data['X_train'], data['y_train'], 
                                   cv=5, scoring='r2')
        
        # Store results
        self.models['linear_regression'] = model
        res = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': {
                'train': metrics_train,
                'test': metrics_test
            },
            'coefficients': dict(zip(self.features, model.coef_)),
            'intercept': float(model.intercept_),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        self.results['linear_regression'] = res
        
        # Print results
        print("\n📈 Model Performance:")
        print(f"  Training R²: {metrics_train['R2']:.4f}")
        print(f"  Testing R²:  {metrics_test['R2']:.4f}")
        print(f"  Training RMSE: {metrics_train['RMSE']:,.2f}")
        print(f"  Testing RMSE:  {metrics_test['RMSE']:,.2f}")
        print(f"  Training MAE:  {metrics_train['MAE']:,.2f}")
        print(f"  Testing MAE:   {metrics_test['MAE']:,.2f}")
        print(f"  Training MAPE: {metrics_train['MAPE']:.2f}%")
        print(f"  Testing MAPE:  {metrics_test['MAPE']:.2f}%")
        print(f"  5-Fold CV R²:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        print("\n🔍 Top 5 Feature Coefficients:")
        coeff_df = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': model.coef_
        })
        coeff_df['Abs_Coefficient'] = np.abs(coeff_df['Coefficient'])
        top_features = coeff_df.nlargest(5, 'Abs_Coefficient')
        for _, row in top_features.iterrows():
            print(f"  {row['Feature']:25s}: {row['Coefficient']:.6f}")
        
        # Generate visualizations
        if show_plots and self.visualization_available:
            self._plot_linear_regression_results(res, data, save_plots)
        elif show_plots:
            print("⚠️  Visualization skipped - module not available")
        
        print("\n✅ Linear Regression completed successfully!")
        return res
    
    def _plot_linear_regression_results(self, results, data, save_plots=False):
        """Generate comprehensive plots for linear regression results."""
        try:
            from visualization import TrafficVisualizer
        except ImportError:
            print("❌ Visualization module not found")
            return
        
        # Create DataFrame with predictions for visualization
        viz_df = pd.DataFrame({
            'y_true': data['y_test'],
            'y_pred': results['predictions']['test']
        })
        
        # Create visualizer
        viz = TrafficVisualizer(viz_df)
        
        try:
            # Generate regression analysis plot
            fig_reg = viz.plot_regression_analysis(
                data['y_test'], results['predictions']['test'], 
                "Linear Regression"
            )
            
            # Generate feature importance plot
            fig_feat = viz.plot_feature_importance(
                self.features, results['coefficients'], 
                "Linear Regression Coefficients"
            )
            
            # Store visualizations
            self.visualizations['linear_regression'] = {
                'regression_analysis': fig_reg,
                'feature_importance': fig_feat
            }
            
            # Save plots if requested
            if save_plots:
                output_dir = 'reports/figures'
                os.makedirs(output_dir, exist_ok=True)
                fig_reg.savefig(f'{output_dir}/linear_regression_analysis.png', dpi=300, bbox_inches='tight')
                fig_feat.savefig(f'{output_dir}/linear_regression_feature_importance.png', dpi=300, bbox_inches='tight')
                print(f"  📊 Plots saved to {output_dir}/")
            
            plt.show()
            
        except AttributeError as e:
            print(f"❌ Visualization method not found: {e}")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def logistic_regression(self, show_plots=True, save_plots=False):
        """Enhanced Logistic Regression with comprehensive analysis."""
        print("\n" + "=" * 60)
        print("Logistic Regression Analysis")
        print("=" * 60)
        
        if self.problem_type != 'classification':
            print("⚠️  Switching to classification mode...")
            self.problem_type = 'classification'
        
        data = self.data if self.data else self.prepare_data()
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred_train = model.predict(data['X_train'])
        y_pred_test = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        # Calculate metrics
        accuracy_train = accuracy_score(data['y_train'], y_pred_train)
        accuracy_test = accuracy_score(data['y_test'], y_pred_test)
        precision = precision_score(data['y_test'], y_pred_test, zero_division=0)
        recall = recall_score(data['y_test'], y_pred_test, zero_division=0)
        f1 = f1_score(data['y_test'], y_pred_test, zero_division=0)
        
        # ROC Curve metrics
        fpr, tpr, thresholds = roc_curve(data['y_test'], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Classification report
        report = classification_report(data['y_test'], y_pred_test, output_dict=True)
        cm = confusion_matrix(data['y_test'], y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, data['X_train'], data['y_train'], 
                                   cv=5, scoring='accuracy')
        
        # Store results
        self.models['logistic_regression'] = model
        res = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'probabilities': y_pred_proba
            },
            'metrics': {
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'coefficients': dict(zip(self.features, model.coef_[0])),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        self.results['logistic_regression'] = res
        
        # Print results
        print("\n📈 Model Performance:")
        print(f"  Training Accuracy: {accuracy_train:.4f}")
        print(f"  Testing Accuracy:  {accuracy_test:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        print("\n📋 Classification Report:")
        print(classification_report(data['y_test'], y_pred_test))
        
        print("\n🔍 Top 5 Feature Coefficients:")
        coeff_df = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': model.coef_[0]
        })
        coeff_df['Abs_Coefficient'] = np.abs(coeff_df['Coefficient'])
        top_features = coeff_df.nlargest(5, 'Abs_Coefficient')
        for _, row in top_features.iterrows():
            print(f"  {row['Feature']:25s}: {row['Coefficient']:.6f}")
        
        # Generate visualizations
        if show_plots and self.visualization_available:
            self._plot_logistic_regression_results(res, data, save_plots)
        elif show_plots:
            print("⚠️  Visualization skipped - module not available")
        
        print("\n✅ Logistic Regression completed successfully!")
        return res
    
    def _plot_logistic_regression_results(self, results, data, save_plots=False):
        """Generate comprehensive plots for logistic regression results."""
        try:
            from visualization import TrafficVisualizer
        except ImportError:
            print("❌ Visualization module not found")
            return
        
        # Create DataFrame with predictions for visualization
        viz_df = pd.DataFrame({
            'y_true': data['y_test'],
            'y_pred': results['predictions']['test'],
            'y_proba': results['predictions']['probabilities']
        })
        
        # Create visualizer
        viz = TrafficVisualizer(viz_df)
        
        try:
            # Generate classification analysis plot
            fig_clf = viz.plot_classification_analysis(
                data['y_test'], results['predictions']['test'], 
                results['predictions']['probabilities'], "Logistic Regression"
            )
            
            # Generate feature importance plot
            fig_feat = viz.plot_feature_importance(
                self.features, results['coefficients'], 
                "Logistic Regression Coefficients"
            )
            
            # Store visualizations
            self.visualizations['logistic_regression'] = {
                'classification_analysis': fig_clf,
                'feature_importance': fig_feat
            }
            
            # Save plots if requested
            if save_plots:
                output_dir = 'reports/figures'
                os.makedirs(output_dir, exist_ok=True)
                fig_clf.savefig(f'{output_dir}/logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
                fig_feat.savefig(f'{output_dir}/logistic_regression_feature_importance.png', dpi=300, bbox_inches='tight')
                print(f"  📊 Plots saved to {output_dir}/")
            
            plt.show()
            
        except AttributeError as e:
            print(f"❌ Visualization method not found: {e}")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def svm_classification(self, kernel='rbf', C=1.0, show_plots=True, save_plots=False):
        """Enhanced SVM classification with comprehensive analysis."""
        print("\n" + "=" * 60)
        print(f"SVM Classification ({kernel} kernel, C={C})")
        print("=" * 60)
        
        if self.problem_type != 'classification':
            print("⚠️  Switching to classification mode...")
            self.problem_type = 'classification'
        
        data = self.data if self.data else self.prepare_data()
        
        # Train model
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42, 
                   class_weight='balanced')
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred_train = model.predict(data['X_train'])
        y_pred_test = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test'])[:, 1]
        
        # Calculate metrics
        accuracy_train = accuracy_score(data['y_train'], y_pred_train)
        accuracy_test = accuracy_score(data['y_test'], y_pred_test)
        precision = precision_score(data['y_test'], y_pred_test, zero_division=0)
        recall = recall_score(data['y_test'], y_pred_test, zero_division=0)
        f1 = f1_score(data['y_test'], y_pred_test, zero_division=0)
        
        # ROC Curve metrics
        fpr, tpr, thresholds = roc_curve(data['y_test'], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Classification report
        report = classification_report(data['y_test'], y_pred_test, output_dict=True)
        cm = confusion_matrix(data['y_test'], y_pred_test)
        
        # Store results
        model_name = f'svm_{kernel}'
        self.models[model_name] = model
        res = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'probabilities': y_pred_proba
            },
            'metrics': {
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'support_vectors': len(model.support_vectors_)
            },
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'cv_scores': cross_val_score(model, data['X_train'], data['y_train'], cv=5)
        }
        self.results[model_name] = res
        
        # Print results
        print("\n📈 Model Performance:")
        print(f"  Training Accuracy: {accuracy_train:.4f}")
        print(f"  Testing Accuracy:  {accuracy_test:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  Support Vectors: {len(model.support_vectors_)}")
        
        print("\n📋 Classification Report:")
        print(classification_report(data['y_test'], y_pred_test))
        
        # Generate visualizations
        if show_plots and self.visualization_available:
            self._plot_svm_results(res, data, kernel, save_plots)
        elif show_plots:
            print("⚠️  Visualization skipped - module not available")
        
        print(f"\n✅ SVM ({kernel}) completed successfully!")
        return res
    
    def _plot_svm_results(self, results, data, kernel, save_plots=False):
        """Generate comprehensive plots for SVM results."""
        try:
            from visualization import TrafficVisualizer
        except ImportError:
            print("❌ Visualization module not found")
            return
        
        # Create DataFrame with predictions for visualization
        viz_df = pd.DataFrame({
            'y_true': data['y_test'],
            'y_pred': results['predictions']['test'],
            'y_proba': results['predictions']['probabilities']
        })
        
        # Create visualizer
        viz = TrafficVisualizer(viz_df)
        
        try:
            # Generate classification analysis plot
            fig_clf = viz.plot_classification_analysis(
                data['y_test'], results['predictions']['test'], 
                results['predictions']['probabilities'], f"SVM ({kernel})"
            )
            
            # Store visualizations
            self.visualizations[f'svm_{kernel}'] = {
                'classification_analysis': fig_clf
            }
            
            # Save plots if requested
            if save_plots:
                output_dir = 'reports/figures'
                os.makedirs(output_dir, exist_ok=True)
                fig_clf.savefig(f'{output_dir}/svm_{kernel}_analysis.png', dpi=300, bbox_inches='tight')
                print(f"  📊 Plots saved to {output_dir}/")
            
            plt.show()
            
        except AttributeError as e:
            print(f"❌ Visualization method not found: {e}")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def kmeans_clustering(self, n_clusters=3, feature_subset=None, show_plots=True, save_plots=False):
        """Enhanced K-means clustering with comprehensive analysis."""
        print("\n" + "=" * 60)
        print(f"K-means Clustering Analysis ({n_clusters} clusters)")
        print("=" * 60)
        
        # Determine features for clustering
        if feature_subset is None:
            feature_subset = [f for f in ['total', 'hk_residents', 'mainland_visitors', 
                                         'other_visitors', 'day_of_week', 'month'] 
                            if f in self.df.columns]
        
        if not feature_subset:
            print("❌ No features available for clustering")
            return None
        
        print(f"  Using features: {feature_subset}")
        
        # Prepare data
        clustering_data = self.df[feature_subset].copy().fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clustering_data)
        
        # Apply K-means
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels) if 1 < n_clusters < len(X_scaled) else None
        inertia = model.inertia_
        
        # Add clusters to a copy of the dataframe
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = labels
        
        # Analyze clusters
        cluster_stats = {}
        for cluster in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
            cluster_stats[cluster] = {
                'count': len(cluster_data),
                'percentage': (len(cluster_data) / len(df_with_clusters)) * 100,
                'mean_total': cluster_data['total'].mean() if 'total' in cluster_data.columns else np.nan,
                'std_total': cluster_data['total'].std() if 'total' in cluster_data.columns else np.nan,
                'avg_day_of_week': cluster_data['day_of_week'].mean() if 'day_of_week' in cluster_data.columns else np.nan,
                'avg_month': cluster_data['month'].mean() if 'month' in cluster_data.columns else np.nan
            }
        
        # Store results
        self.models['kmeans'] = model
        res = {
            'model': model,
            'n_clusters': n_clusters,
            'labels': labels,
            'cluster_stats': cluster_stats,
            'metrics': {
                'silhouette_score': silhouette,
                'inertia': inertia
            },
            'cluster_centers': model.cluster_centers_,
            'feature_subset': feature_subset,
            'df_with_clusters': df_with_clusters
        }
        self.results['kmeans'] = res
        
        # Print results
        print("\n📊 Clustering Results:")
        if silhouette is not None:
            print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Inertia: {inertia:.2f}")
        
        print("\n👥 Cluster Distribution:")
        for cluster, stats in cluster_stats.items():
            print(f"  Cluster {cluster}: {stats['count']:4d} samples ({stats['percentage']:5.1f}%)")
            if not np.isnan(stats['mean_total']):
                print(f"          Avg Total: {stats['mean_total']:,.0f} ± {stats['std_total']:,.0f}")
        
        # Generate visualizations
        if show_plots and self.visualization_available:
            self._plot_clustering_results(res, save_plots)
        elif show_plots:
            print("⚠️  Visualization skipped - module not available")
        
        print("\n✅ K-means Clustering completed successfully!")
        return res
    
    def _plot_clustering_results(self, results, save_plots=False):
        """Generate comprehensive plots for clustering results."""
        try:
            from visualization import TrafficVisualizer
        except ImportError:
            print("❌ Visualization module not found")
            return
        
        # Create visualizer with clustered data
        viz = TrafficVisualizer(results['df_with_clusters'])
        
        try:
            # Generate clustering analysis plot
            fig_cluster = viz.plot_clustering_analysis(
                results['df_with_clusters'], 
                cluster_col='cluster',
                feature_cols=results['feature_subset'],
                title=f"K-means Clustering ({results['n_clusters']} clusters)"
            )
            
            # Generate elbow method plot (for optimal cluster selection)
            if len(results['df_with_clusters']) > 10:
                fig_elbow = self._plot_elbow_method(results['df_with_clusters'], results['feature_subset'])
                self.visualizations['kmeans_elbow'] = fig_elbow
            
            # Store visualizations
            self.visualizations['kmeans'] = {
                'clustering_analysis': fig_cluster
            }
            
            # Save plots if requested
            if save_plots:
                output_dir = 'reports/figures'
                os.makedirs(output_dir, exist_ok=True)
                fig_cluster.savefig(f'{output_dir}/kmeans_clustering_analysis.png', dpi=300, bbox_inches='tight')
                if 'kmeans_elbow' in self.visualizations:
                    self.visualizations['kmeans_elbow'].savefig(f'{output_dir}/kmeans_elbow_method.png', dpi=300, bbox_inches='tight')
                print(f"  📊 Plots saved to {output_dir}/")
            
            plt.show()
            
        except AttributeError as e:
            print(f"❌ Visualization method not found: {e}")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def _plot_elbow_method(self, df, feature_subset, max_clusters=10):
        """Plot elbow method for optimal cluster selection."""
        from sklearn.cluster import KMeans
        
        # Prepare data
        X = df[feature_subset].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Calculate inertia for different numbers of clusters
        inertias = []
        k_range = range(1, min(max_clusters + 1, len(X_scaled)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for k, inertia in zip(k_range, inertias):
            ax.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points", 
                       xytext=(0,10), ha='center')
        
        plt.tight_layout()
        return fig

    def compare_control_points(self, control_point_col='control_point', 
                              target_col='total', 
                              time_col='date',
                              show_plots=True,
                              save_plots=False):
        """
        Compare traffic patterns between different control points.
        
        Parameters:
        -----------
        control_point_col : str
            Column name containing control point identifiers
        target_col : str
            Column name containing traffic volume
        time_col : str
            Column name containing timestamps
        show_plots : bool
            Whether to display plots
        save_plots : bool
            Whether to save plots to file
        
        Returns:
        --------
        dict with comparison statistics and visualizations
        """
        print("\n" + "=" * 60)
        print("Control Point Traffic Comparison")
        print("=" * 60)
        
        # Check if control point column exists
        if control_point_col not in self.df.columns:
            print(f"❌ Control point column '{control_point_col}' not found in data")
            available_cols = self.df.columns.tolist()
            print(f"   Available columns: {available_cols[:10]}...")
            return None
        
        # Get unique control points
        control_points = self.df[control_point_col].unique()
        print(f"📊 Found {len(control_points)} control points:")
        for cp in control_points[:10]:  # Show first 10
            print(f"  • {cp}")
        if len(control_points) > 10:
            print(f"  ... and {len(control_points) - 10} more")
        
        # Calculate statistics for each control point
        stats_by_cp = {}
        for cp in control_points:
            cp_data = self.df[self.df[control_point_col] == cp].copy()
            
            if cp_data.empty:
                continue
            
            # Basic statistics
            stats = {
                'count': len(cp_data),
                'mean': cp_data[target_col].mean(),
                'median': cp_data[target_col].median(),
                'std': cp_data[target_col].std(),
                'min': cp_data[target_col].min(),
                'max': cp_data[target_col].max(),
                'q25': cp_data[target_col].quantile(0.25),
                'q75': cp_data[target_col].quantile(0.75),
                'data': cp_data
            }
            
            # Add time-based statistics if time column exists
            if time_col in cp_data.columns and pd.api.types.is_datetime64_any_dtype(cp_data[time_col]):
                cp_data['hour'] = cp_data[time_col].dt.hour
                cp_data['day_of_week'] = cp_data[time_col].dt.dayofweek
                cp_data['month'] = cp_data[time_col].dt.month
                
                stats.update({
                    'peak_hour': cp_data.groupby('hour')[target_col].mean().idxmax(),
                    'peak_hour_volume': cp_data.groupby('hour')[target_col].mean().max(),
                    'busiest_weekday': cp_data.groupby('day_of_week')[target_col].mean().idxmax(),
                    'busiest_month': cp_data.groupby('month')[target_col].mean().idxmax(),
                    'weekday_mean': cp_data[cp_data['day_of_week'] < 5][target_col].mean(),
                    'weekend_mean': cp_data[cp_data['day_of_week'] >= 5][target_col].mean(),
                    'weekend_ratio': (cp_data[cp_data['day_of_week'] >= 5][target_col].mean() / 
                                     cp_data[cp_data['day_of_week'] < 5][target_col].mean() 
                                     if cp_data[cp_data['day_of_week'] < 5][target_col].mean() > 0 else 0)
                })
            
            stats_by_cp[cp] = stats
        
        # Create comparison DataFrame
        comparison_data = {}
        for cp, stats in stats_by_cp.items():
            comparison_data[cp] = {
                'Total Records': stats['count'],
                f'Mean {target_col}': stats['mean'],
                f'Median {target_col}': stats['median'],
                'Std Dev': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Peak Hour': stats.get('peak_hour', 'N/A'),
                'Peak Volume': stats.get('peak_hour_volume', 'N/A'),
                'Weekday Mean': stats.get('weekday_mean', 'N/A'),
                'Weekend Mean': stats.get('weekend_mean', 'N/A'),
                'Weekend Ratio': stats.get('weekend_ratio', 'N/A')
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Sort by mean traffic (descending)
        comparison_df = comparison_df.sort_values(f'Mean {target_col}', ascending=False)
        
        # Calculate rankings
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        comparison_df['Traffic Share (%)'] = (comparison_df[f'Mean {target_col}'] / 
                                             comparison_df[f'Mean {target_col}'].sum()) * 100
        
        print("\n📊 Control Point Comparison (Top 10 by Mean Traffic):")
        print(comparison_df.head(10).to_string())
        
        # Generate visualizations
        visualizations = {}
        if show_plots and self.visualization_available:
            try:
                from visualization import TrafficVisualizer
                
                # Create visualizer with full data
                viz = TrafficVisualizer(self.df)
                
                # Try to generate different types of visualizations
                try:
                    # 1. Bar chart of mean traffic by control point
                    fig1 = viz.plot_control_point_comparison(
                        self.df, 
                        control_point_col=control_point_col,
                        target_col=target_col,
                        title='Mean Traffic by Control Point'
                    )
                    visualizations['mean_traffic_bar'] = fig1
                except AttributeError:
                    print("⚠️  plot_control_point_comparison not available in visualization module")
                
                # 2. Time series comparison for top control points
                if time_col in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df[time_col]):
                    try:
                        top_cps = comparison_df.head(5).index.tolist()
                        fig2 = viz.plot_control_point_time_series(
                            self.df,
                            control_point_col=control_point_col,
                            target_col=target_col,
                            time_col=time_col,
                            control_points=top_cps,
                            title='Traffic Time Series (Top 5 Control Points)'
                        )
                        visualizations['time_series_comparison'] = fig2
                    except AttributeError:
                        print("⚠️  plot_control_point_time_series not available in visualization module")
                
                # Try other visualization methods
                try:
                    # 3. Box plot comparison
                    fig3 = viz.plot_control_point_boxplots(
                        self.df,
                        control_point_col=control_point_col,
                        target_col=target_col,
                        control_points=comparison_df.head(8).index.tolist(),
                        title='Traffic Distribution Comparison'
                    )
                    visualizations['boxplot_comparison'] = fig3
                except AttributeError:
                    print("⚠️  plot_control_point_boxplots not available in visualization module")
                
                # Save plots if requested
                if save_plots and visualizations:
                    output_dir = 'reports/control_point_analysis'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    for plot_name, fig in visualizations.items():
                        fig.savefig(f'{output_dir}/{plot_name}.png', 
                                   dpi=300, bbox_inches='tight')
                    print(f"\n📊 Plots saved to {output_dir}/")
                
                if visualizations:
                    plt.show()
                
            except ImportError:
                print("⚠️  Visualization module not available. Skipping plots.")
            except Exception as e:
                print(f"⚠️  Error generating visualizations: {e}")
        elif show_plots:
            print("⚠️  Visualization skipped - module not available")
        
        # Perform statistical tests for significant differences
        print("\n" + "=" * 60)
        print("Statistical Comparison Tests")
        print("=" * 60)
        
        statistical_tests = {}
        
        try:
            from scipy import stats
            
            # Compare top 3 control points pairwise
            top_control_points = comparison_df.head(3).index.tolist()
            
            for i in range(len(top_control_points)):
                for j in range(i+1, len(top_control_points)):
                    cp1 = top_control_points[i]
                    cp2 = top_control_points[j]
                    
                    data1 = stats_by_cp[cp1]['data'][target_col].values
                    data2 = stats_by_cp[cp2]['data'][target_col].values
                    
                    # t-test for mean difference
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                    
                    # Mann-Whitney U test for distribution difference
                    u_stat, u_p_value = stats.mannwhitneyu(data1, data2)
                    
                    # Calculate effect size (Cohen's d)
                    n1, n2 = len(data1), len(data2)
                    pooled_std = np.sqrt(((n1-1)*np.std(data1)**2 + (n2-1)*np.std(data2)**2) / (n1+n2-2))
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    
                    statistical_tests[f"{cp1}_vs_{cp2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant_at_0.05': p_value < 0.05,
                        'mann_whitney_u': u_stat,
                        'mann_whitney_p': u_p_value,
                        'cohens_d': cohens_d,
                        'mean_difference': np.mean(data1) - np.mean(data2),
                        'mean_difference_percent': (np.mean(data1) - np.mean(data2)) / np.mean(data2) * 100
                    }
                    
                    print(f"\n🔬 {cp1} vs {cp2}:")
                    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
                    print(f"   {'✓ SIGNIFICANT difference' if p_value < 0.05 else '✗ No significant difference'}")
                    print(f"   Cohen's d (effect size): {cohens_d:.3f}")
                    print(f"   Mean difference: {np.mean(data1) - np.mean(data2):,.0f} "
                          f"({(np.mean(data1) - np.mean(data2))/np.mean(data2)*100:+.1f}%)")
        except ImportError:
            print("⚠️  scipy not available for statistical tests")
        
        # Create summary insights
        print("\n" + "=" * 60)
        print("Key Insights")
        print("=" * 60)
        
        if len(comparison_df) > 0:
            top_cp = comparison_df.index[0]
            bottom_cp = comparison_df.index[-1]
            ratio = comparison_df.loc[top_cp, f'Mean {target_col}'] / comparison_df.loc[bottom_cp, f'Mean {target_col}']
            
            print(f"🏆 Busiest Control Point: {top_cp}")
            print(f"   • Average daily traffic: {comparison_df.loc[top_cp, f'Mean {target_col}']:,.0f}")
            print(f"   • Share of total traffic: {comparison_df.loc[top_cp, 'Traffic Share (%)']:.1f}%")
            print(f"   • Peak hour: {comparison_df.loc[top_cp, 'Peak Hour']}:00")
            
            print(f"\n⚡ Quietest Control Point: {bottom_cp}")
            print(f"   • Average daily traffic: {comparison_df.loc[bottom_cp, f'Mean {target_col}']:,.0f}")
            print(f"   • Share of total traffic: {comparison_df.loc[bottom_cp, 'Traffic Share (%)']:.1f}%")
            
            print(f"\n📈 Traffic Ratio (Busiest/Quietest): {ratio:.1f}x")
            
            # Identify control points with different patterns
            weekday_weekend_ratio = comparison_df['Weekend Ratio'].replace('N/A', np.nan).dropna()
            if len(weekday_weekend_ratio) > 0:
                business_oriented = weekday_weekend_ratio.idxmin()
                leisure_oriented = weekday_weekend_ratio.idxmax()
                
                print(f"\n🏢 Business-Oriented (lowest weekend ratio): {business_oriented}")
                print(f"   Weekend/Weekday ratio: {weekday_weekend_ratio[business_oriented]:.2f}")
                
                print(f"\n🎯 Leisure-Oriented (highest weekend ratio): {leisure_oriented}")
                print(f"   Weekend/Weekday ratio: {weekday_weekend_ratio[leisure_oriented]:.2f}")
        
        # Store results
        self.results['control_point_comparison'] = {
            'comparison_df': comparison_df,
            'stats_by_cp': stats_by_cp,
            'statistical_tests': statistical_tests,
            'visualizations': visualizations,
            'top_control_point': top_cp if len(comparison_df) > 0 else None,
            'bottom_control_point': bottom_cp if len(comparison_df) > 0 else None,
            'traffic_ratio': ratio if len(comparison_df) > 0 else None,
            'total_control_points': len(control_points)
        }
        
        print("\n✅ Control Point Comparison completed successfully!")
        return self.results['control_point_comparison']
    
    def compare_models(self, show_plots=True):
        """Compare performance of all trained models."""
        print("\n" + "=" * 60)
        print("Model Comparison Summary")
        print("=" * 60)
        
        if not self.results:
            print("❌ No models have been trained yet.")
            return None
        
        comparison_data = {}
        
        # Collect metrics from all models
        for model_name, results in self.results.items():
            if model_name.startswith('linear'):
                comparison_data[model_name] = {
                    'R²': results['metrics']['test']['R2'],
                    'RMSE': results['metrics']['test']['RMSE'],
                    'MAE': results['metrics']['test']['MAE'],
                    'Type': 'Regression'
                }
            elif model_name.startswith('logistic') or model_name.startswith('svm'):
                comparison_data[model_name] = {
                    'Accuracy': results['metrics']['accuracy_test'],
                    'Precision': results['metrics']['precision'],
                    'Recall': results['metrics']['recall'],
                    'F1-Score': results['metrics']['f1_score'],
                    'ROC AUC': results['metrics'].get('roc_auc', np.nan),
                    'Type': 'Classification'
                }
            elif model_name.startswith('kmeans'):
                comparison_data[model_name] = {
                    'Silhouette Score': results['metrics']['silhouette_score'],
                    'Inertia': results['metrics']['inertia'],
                    'Type': 'Clustering'
                }
            elif model_name == 'control_point_comparison':
                # Skip control point comparison in model comparison
                continue
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        print("\n📊 Model Comparison:")
        print(comparison_df.to_string())
        
        # Generate comparison plot
        if show_plots and self.visualization_available and len(comparison_data) > 1:
            try:
                from visualization import TrafficVisualizer
                viz = TrafficVisualizer(pd.DataFrame())
                fig_comparison = viz.plot_model_comparison(self.results)
                self.visualizations['model_comparison'] = fig_comparison
                plt.show()
            except AttributeError:
                print("⚠️  plot_model_comparison not available in visualization module")
            except Exception as e:
                print(f"⚠️  Error generating model comparison plot: {e}")
        
        return comparison_df
    
    def run_all_models(self, regression=True, classification=True, clustering=True, 
                      control_point_analysis=True,  # New parameter
                      show_plots=True, save_plots=True):
        """Run all models in a comprehensive pipeline."""
        print("=" * 70)
        print("COMPREHENSIVE MACHINE LEARNING PIPELINE")
        print("=" * 70)
        
        # Prepare data once for all models
        print("\n📊 Preparing data for all models...")
        self.prepare_data()
        
        results = {}
        
        # Run regression models
        if regression:
            print("\n" + "=" * 50)
            print("REGRESSION MODELS")
            print("=" * 50)
            
            print("\n🚀 Running Linear Regression...")
            try:
                self.problem_type = 'regression'
                results['linear_regression'] = self.linear_regression(show_plots=show_plots, save_plots=save_plots)
            except Exception as e:
                print(f"❌ Linear Regression failed: {e}")
        
        # Run classification models
        if classification:
            print("\n" + "=" * 50)
            print("CLASSIFICATION MODELS")
            print("=" * 50)
            
            # Switch to classification mode
            original_target = self.target
            if 'total' in self.df.columns:
                # Create binary classification target
                threshold = self.df['total'].quantile(0.75)
                self.df['traffic_level'] = (self.df['total'] > threshold).astype(int)
                self.target = 'traffic_level'
                self.problem_type = 'classification'
            
            print("\n🚀 Running Logistic Regression...")
            try:
                results['logistic_regression'] = self.logistic_regression(show_plots=show_plots, save_plots=save_plots)
            except Exception as e:
                print(f"❌ Logistic Regression failed: {e}")
            
            print("\n🚀 Running SVM (RBF kernel)...")
            try:
                results['svm_rbf'] = self.svm_classification(kernel='rbf', C=1.0, 
                                                           show_plots=show_plots, save_plots=save_plots)
            except Exception as e:
                print(f"❌ SVM failed: {e}")
            
            # Restore original target
            self.target = original_target
        
        # Run clustering
        if clustering:
            print("\n" + "=" * 50)
            print("CLUSTERING MODELS")
            print("=" * 50)
            
            print("\n🚀 Running K-means Clustering...")
            try:
                results['kmeans'] = self.kmeans_clustering(n_clusters=3, 
                                                          show_plots=show_plots, save_plots=save_plots)
            except Exception as e:
                print(f"❌ K-means failed: {e}")
        
        # Run control point analysis
        if control_point_analysis and 'control_point' in self.df.columns:
            print("\n" + "=" * 50)
            print("CONTROL POINT ANALYSIS")
            print("=" * 50)
            
            print("\n🚀 Comparing control point traffic patterns...")
            try:
                results['control_point_comparison'] = self.compare_control_points(
                    control_point_col='control_point',
                    target_col='total' if 'total' in self.df.columns else self.target,
                    time_col='date' if 'date' in self.df.columns else None,
                    show_plots=show_plots,
                    save_plots=save_plots
                )
            except Exception as e:
                print(f"❌ Control point comparison failed: {e}")
        elif control_point_analysis:
            print("\n⚠️  'control_point' column not found. Skipping control point analysis.")
            print("   Available columns:", list(self.df.columns)[:10], "...")
        
        # Compare models
        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        
        comparison = self.compare_models(show_plots=show_plots)
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return {
            'results': results,
            'comparison': comparison,
            'visualizations': self.visualizations
        }


if __name__ == "__main__":
    # Enhanced demo with visualization
    import os
    import sys
    
    print("Enhanced Models Module Demo with Visualization")
    print("=" * 60)
    
    # Check if visualization module exists
    try:
        from visualization import TrafficVisualizer
        print("✅ Visualization module available")
    except ImportError:
        print("⚠️  Visualization module not available, plots will be limited")
    
    # Check if processed data exists
    proc_path = os.path.join("data", "processed", "passenger_data_daily.csv")
    if not os.path.exists(proc_path):
        print(f"Processed data not found: {proc_path}")
        print("Please run data preprocessing first.")
        sys.exit(1)
    
    # Load data
    try:
        df = pd.read_csv(proc_path, parse_dates=['date'])
        print(f"✅ Loaded data: {df.shape}")
        
        # Ensure numeric types
        for col in ['total', 'hk_residents', 'mainland_visitors', 'other_visitors']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create date-derived features
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        
        # Run control point comparison demo if column exists
        if 'control_point' in df.columns:
            print("\n" + "=" * 60)
            print("DEMO: CONTROL POINT COMPARISON")
            print("=" * 60)
            
            # Select features for modeling
            features = [f for f in ['day_of_week', 'month', 'is_weekend',
                                   'hk_residents', 'mainland_visitors', 'other_visitors']
                       if f in df.columns]
            target = 'total'
            
            if target not in df.columns:
                print(f"❌ Target column '{target}' not found")
                sys.exit(1)
            
            print(f"📊 Using {len(features)} features: {features}")
            print(f"🎯 Target: {target}")
            
            # Create model instance
            tm = TrafficModels(df=df, features=features, target=target, problem_type='regression')
            
            # Run control point comparison separately
            cp_results = tm.compare_control_points(
                control_point_col='control_point',
                target_col='total',
                time_col='date',
                show_plots=True,
                save_plots=True
            )
            
            print("\n" + "=" * 60)
            print("DEMO: FULL MODEL PIPELINE")
            print("=" * 60)
        
        # Select features for modeling (if not already done above)
        features = [f for f in ['day_of_week', 'month', 'is_weekend',
                               'hk_residents', 'mainland_visitors', 'other_visitors']
                   if f in df.columns]
        target = 'total'
        
        if target not in df.columns:
            print(f"❌ Target column '{target}' not found")
            sys.exit(1)
        
        print(f"📊 Using {len(features)} features: {features}")
        print(f"🎯 Target: {target}")
        
        # Create model instance
        tm = TrafficModels(df=df, features=features, target=target, problem_type='regression')
        
        # Run comprehensive pipeline
        print("\n🚀 Running comprehensive model pipeline...")
        pipeline_results = tm.run_all_models(
            regression=True,
            classification=True,
            clustering=True,
            control_point_analysis=True,  # Include control point analysis
            show_plots=True,
            save_plots=True
        )
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()