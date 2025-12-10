"""
Machine learning models for passenger traffic analysis.
Updated for the actual dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, classification_report, confusion_matrix,
                            silhouette_score, precision_score, recall_score, f1_score)

class TrafficModels:
    """Implement all required ML models for traffic analysis."""
    
    def __init__(self, df, features, target, problem_type='regression'):
        """
        Initialize models with data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed data
        features : list
            List of feature column names
        target : str
            Target column name
        problem_type : str
            'regression' or 'classification'
        """
        self.df = df
        self.features = features
        self.target = target
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.visualizations = {}
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare train-test split based on problem type."""
        print(f"Preparing data for {self.problem_type}...")
        
        X = self.df[self.features].copy()
        
        if self.problem_type == 'regression':
            y = self.df[self.target].copy()
            
            # For regression, also create classification target for additional analysis
            threshold = y.quantile(0.75)
            y_class = (y > threshold).astype(int)
            self.classification_target = y_class
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.data = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_class': y_class.loc[y_train.index],
                'y_test_class': y_class.loc[y_test.index]
            }
            
        elif self.problem_type == 'classification':
            # For classification, use traffic_level as target
            if 'traffic_level' in self.df.columns:
                y = self.df['traffic_level'].copy()
            else:
                # Create binary target
                y_original = self.df[self.target].copy()
                threshold = y_original.quantile(0.75)
                y = (y_original > threshold).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.data = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test
            }
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        return self.data
    
    def linear_regression(self):
        """Implement Linear Regression for traffic prediction."""
        print("\n" + "=" * 50)
        print("Linear Regression Analysis")
        print("=" * 50)
        
        data = self.data
        model = LinearRegression()
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        y_pred_train = model.predict(data['X_train'])
        y_pred_test = model.predict(data['X_test'])
        
        # Metrics
        train_mse = mean_squared_error(data['y_train'], y_pred_train)
        test_mse = mean_squared_error(data['y_test'], y_pred_test)
        train_r2 = r2_score(data['y_train'], y_pred_train)
        test_r2 = r2_score(data['y_test'], y_pred_test)
        train_mae = mean_absolute_error(data['y_train'], y_pred_train)
        test_mae = mean_absolute_error(data['y_test'], y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, data['X_train'], data['y_train'], 
                                   cv=5, scoring='r2')
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'metrics': {
                'train': {
                    'MSE': train_mse,
                    'MAE': train_mae,
                    'R2': train_r2
                },
                'test': {
                    'MSE': test_mse,
                    'MAE': test_mae,
                    'R2': test_r2
                }
            },
            'coefficients': dict(zip(self.features, model.coef_)),
            'intercept': model.intercept_,
            'cv_scores': cv_scores
        }
        
        print("\n📊 Model Performance:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²:  {test_r2:.4f}")
        print(f"  Training MSE: {train_mse:,.2f}")
        print(f"  Testing MSE:  {test_mse:,.2f}")
        print(f"  5-Fold CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        print("\n🔍 Feature Coefficients:")
        for feature, coef in zip(self.features, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        
        return self.results['linear_regression']
    
    def logistic_regression(self):
        """Implement Logistic Regression for traffic classification."""
        print("\n" + "=" * 50)
        print("Logistic Regression Analysis")
        print("=" * 50)
        
        # Use classification data if available
        if 'y_train_class' in self.data:
            y_train = self.data['y_train_class']
            y_test = self.data['y_test_class']
        else:
            y_train = self.data['y_train']
            y_test = self.data['y_test']
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.data['X_train'], y_train)
        
        # Predictions
        y_pred_train = model.predict(self.data['X_train'])
        y_pred_test = model.predict(self.data['X_test'])
        y_pred_proba = model.predict_proba(self.data['X_test'])[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        report = classification_report(y_test, y_pred_test, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.data['X_train'], y_train, 
                                   cv=5, scoring='accuracy')
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'probabilities': y_pred_proba
            },
            'metrics': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_accuracy': cv_scores.mean()
            },
            'classification_report': report,
            'confusion_matrix': cm,
            'coefficients': dict(zip(self.features, model.coef_[0]))
        }
        
        print("\n📊 Model Performance:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        print("\n🔍 Feature Importance (Coefficients):")
        for feature, coef in zip(self.features, model.coef_[0]):
            print(f"  {feature}: {coef:.4f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        return self.results['logistic_regression']
    
    def svm_classification(self, kernel='rbf', C=1.0):
        """Implement SVM for traffic classification."""
        print("\n" + "=" * 50)
        print(f"SVM Classification ({kernel} kernel)")
        print("=" * 50)
        
        # Use classification data
        if 'y_train_class' in self.data:
            y_train = self.data['y_train_class']
            y_test = self.data['y_test_class']
        else:
            y_train = self.data['y_train']
            y_test = self.data['y_test']
        
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(self.data['X_train'], y_train)
        
        # Predictions
        y_pred_train = model.predict(self.data['X_train'])
        y_pred_test = model.predict(self.data['X_test'])
        y_pred_proba = model.predict_proba(self.data['X_test'])[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        
        report = classification_report(y_test, y_pred_test, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_test)
        
        self.models[f'svm_{kernel}'] = model
        self.results[f'svm_{kernel}'] = {
            'model': model,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'probabilities': y_pred_proba
            },
            'metrics': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'classification_report': report,
            'confusion_matrix': cm,
            'support_vectors': model.support_vectors_
        }
        
        print("\n📊 Model Performance:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Support Vectors: {len(model.support_vectors_)}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test))
        
        return self.results[f'svm_{kernel}']
    
    def kmeans_clustering(self, n_clusters=3, analysis_type='daily_patterns'):
        """Implement K-means clustering for pattern discovery."""
        print("\n" + "=" * 50)
        print(f"K-means Clustering ({n_clusters} clusters)")
        print("=" * 50)
        
        # Prepare data for clustering
        if analysis_type == 'daily_patterns':
            # Use daily aggregated data
            clustering_data = self.df.copy()
        else:
            clustering_data = self.df.copy()
        
        # Select features for clustering
        cluster_features = ['total', 'day_of_week', 'month']
        available_features = [f for f in cluster_features if f in clustering_data.columns]
        
        X_cluster = clustering_data[available_features].copy()
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Apply K-means
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        
        # Metrics
        silhouette = silhouette_score(X_scaled, clusters)
        
        # Analyze clusters
        clustering_data['cluster'] = clusters
        
        # Cluster statistics
        cluster_stats = clustering_data.groupby('cluster').agg({
            'total': ['mean', 'std', 'min', 'max', 'count'],
            'day_of_week': 'mean',
            'month': 'mean'
        }).round(2)
        
        self.models['kmeans'] = model
        self.results['kmeans'] = {
            'model': model,
            'clusters': clusters,
            'cluster_labels': clustering_data['cluster'],
            'metrics': {
                'silhouette_score': silhouette,
                'inertia': model.inertia_
            },
            'cluster_statistics': cluster_stats,
            'cluster_counts': clustering_data['cluster'].value_counts().sort_index().to_dict(),
            'cluster_centers': model.cluster_centers_,
            'feature_names': available_features
        }
        
        print("\n📊 Clustering Results:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Inertia: {model.inertia_:.2f}")
        
        print("\n👥 Cluster Distribution:")
        for cluster, count in self.results['kmeans']['cluster_counts'].items():
            percentage = (count / len(clustering_data)) * 100
            print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
        
        print("\n📈 Cluster Statistics:")
        print(cluster_stats)
        
        print("\n📍 Cluster Centers (scaled features):")
        for i, center in enumerate(model.cluster_centers_):
            print(f"  Cluster {i}: {center}")
        
        return self.results['kmeans']
    
    def compare_models(self):
        """Compare performance of different models."""
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)
        
        comparison = {}
        
        # Linear Regression metrics
        if 'linear_regression' in self.results:
            lr_metrics = self.results['linear_regression']['metrics']['test']
            comparison['Linear Regression'] = {
                'R²': lr_metrics['R2'],
                'MSE': lr_metrics['MSE'],
                'MAE': lr_metrics['MAE']
            }
        
        # Logistic Regression metrics
        if 'logistic_regression' in self.results:
            lr_clf_metrics = self.results['logistic_regression']['metrics']
            comparison['Logistic Regression'] = {
                'Accuracy': lr_clf_metrics['test_accuracy'],
                'Precision': lr_clf_metrics['precision'],
                'Recall': lr_clf_metrics['recall'],
                'F1-Score': lr_clf_metrics['f1_score']
            }
        
        # SVM metrics
        svm_models = [k for k in self.results.keys() if k.startswith('svm_')]
        for svm_model in svm_models:
            svm_metrics = self.results[svm_model]['metrics']
            comparison[svm_model] = {
                'Accuracy': svm_metrics['test_accuracy'],
                'Precision': svm_metrics['precision'],
                'Recall': svm_metrics['recall'],
                'F1-Score': svm_metrics['f1_score']
            }
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison).T
        
        print("\n📊 Model Comparison:")
        print(comparison_df)
        
        return comparison_df
    
    def run_all_models(self, analysis_type='daily_total'):
        """Run all required models."""
        print("=" * 60)
        print("Running All Machine Learning Models")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data()
        
        results = {}
        
        # Run models
        print("\n🚀 Starting model training...")
        
        # Linear Regression
        print("\n1️⃣  Linear Regression:")
        results['linear_regression'] = self.linear_regression()
        
        # Logistic Regression
        print("\n2️⃣  Logistic Regression:")
        results['logistic_regression'] = self.logistic_regression()
        
        # SVM
        print("\n3️⃣  Support Vector Machine:")
        results['svm_rbf'] = self.svm_classification(kernel='rbf', C=1.0)
        
        # K-means Clustering
        print("\n4️⃣  K-means Clustering:")
        results['kmeans'] = self.kmeans_clustering(n_clusters=3, analysis_type=analysis_type)
        
        # Model Comparison
        print("\n5️⃣  Model Comparison:")
        results['comparison'] = self.compare_models()
        
        print("\n" + "=" * 60)
        print("✅ All Models Completed Successfully!")
        print("=" * 60)
        
        return results