"""
Machine learning models for passenger traffic analysis.
Minimal, working implementation to replace incomplete placeholders.
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score, precision_score, recall_score, f1_score)
import numpy as np
import pandas as pd

class TrafficModels:
    """Implement ML models for traffic analysis (minimal working)."""
    
    def __init__(self, df: pd.DataFrame, features: list, target: str, problem_type='regression'):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target
        self.problem_type = problem_type
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.data = {}
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare X/y, scaling and train-test split."""
        X = self.df[self.features].copy()
        # ensure no string columns slip through
        for c in X.select_dtypes(include=['object']).columns:
            X[c] = X[c].astype('category').cat.codes
        
        y = self.df[self.target].copy()
        if self.problem_type == 'classification':
            # ensure binary or integer classes
            if y.dtype == 'object' or y.nunique() > 20:
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('NA'))
                self.data['label_encoder'] = le
        else:
            # regression: numeric
            y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        # scale features
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        self.data.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.features
        })
        print(f"Prepared data: X={X.shape}, y={y.shape}, train={len(X_train)}, test={len(X_test)}")
        return self.data
    
    def linear_regression(self):
        """Linear Regression for regression problems."""
        if self.problem_type != 'regression':
            raise ValueError("linear_regression requires problem_type='regression'")
        data = self.data or self.prepare_data()
        model = LinearRegression()
        model.fit(data['X_train'], data['y_train'])
        
        y_pred_train = model.predict(data['X_train'])
        y_pred_test = model.predict(data['X_test'])
        
        self.models['linear_regression'] = model
        res = {
            'train': {
                'MSE': mean_squared_error(data['y_train'], y_pred_train),
                'MAE': mean_absolute_error(data['y_train'], y_pred_train),
                'R2': r2_score(data['y_train'], y_pred_train)
            },
            'test': {
                'MSE': mean_squared_error(data['y_test'], y_pred_test),
                'MAE': mean_absolute_error(data['y_test'], y_pred_test),
                'R2': r2_score(data['y_test'], y_pred_test)
            },
            'coefficients': dict(zip(self.features, model.coef_)),
            'intercept': float(model.intercept_),
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'cv_r2': cross_val_score(model, data['X_train'], data['y_train'], cv=5, scoring='r2')
        }
        self.results['linear_regression'] = res
        print(f"Linear Regression done. Test R2: {res['test']['R2']:.4f}")
        return res
    
    def logistic_regression(self):
        """Logistic Regression for classification."""
        if self.problem_type != 'classification':
            raise ValueError("logistic_regression requires problem_type='classification'")
        data = self.data or self.prepare_data()
        y_train = data['y_train']
        y_test = data['y_test']
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(data['X_train'], y_train)
        
        y_pred_test = model.predict(data['X_test'])
        y_pred_train = model.predict(data['X_train'])
        
        res = {
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'model': model
        }
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = res
        print(f"Logistic Regression done. Test Acc: {res['accuracy_test']:.4f}")
        return res
    
    def svm_classification(self, kernel='rbf', C=1.0):
        """SVM for classification."""
        if self.problem_type != 'classification':
            raise ValueError("svm_classification requires problem_type='classification'")
        data = self.data or self.prepare_data()
        y_train = data['y_train']
        y_test = data['y_test']
        
        model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        model.fit(data['X_train'], y_train)
        
        y_pred_test = model.predict(data['X_test'])
        res = {
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'model': model
        }
        self.models[f'svm_{kernel}'] = model
        self.results[f'svm_{kernel}'] = res
        print(f"SVM ({kernel}) done. Test Acc: {res['accuracy_test']:.4f}")
        return res
    
    def kmeans_clustering(self, n_clusters=3, feature_subset=None):
        """KMeans clustering and attach cluster labels to dataframe."""
        df = self.df.copy()
        if feature_subset is None:
            feature_subset = [f for f in ['total','hk_residents','mainland_visitors','other_visitors'] if f in df.columns]
        if not feature_subset:
            raise ValueError("No features available for clustering")
        
        X = df[feature_subset].fillna(0).values
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        df['cluster'] = labels
        # store results
        self.models['kmeans'] = km
        sil = None
        try:
            if 1 < n_clusters < len(Xs):
                sil = float(silhouette_score(Xs, labels))
        except Exception:
            sil = None
        self.results['kmeans'] = {
            'n_clusters': n_clusters,
            'cluster_counts': dict(pd.Series(labels).value_counts()),
            'silhouette_score': sil,
            'cluster_centers': km.cluster_centers_,
            'feature_subset': feature_subset
        }
        # write back labels to main df
        try:
            self.df['cluster'] = labels
        except Exception:
            pass
        print(f"KMeans done. clusters: {self.results['kmeans']['cluster_counts']}")
        return self.results['kmeans']
    
    def compare_models(self):
        """Return a summary of results available."""
        summary = {}
        for name, res in self.results.items():
            if isinstance(res, dict):
                summary[name] = {k: v for k, v in res.items() if k.endswith('_test') or k in ('accuracy_test','R2','silhouette_score')}
        return summary
    
    def run_all_models(self):
        """Helper to run basic pipeline according to problem_type."""
        self.prepare_data()
        if self.problem_type == 'regression':
            self.linear_regression()
        else:
            self.logistic_regression()
            self.svm_classification()
        return self.results