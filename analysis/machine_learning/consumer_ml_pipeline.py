#!/usr/bin/env python3
"""
Consumer-Focused ML Pipeline for Southwest Airlines Delay Prediction

This script creates a two-stage model that simulates what a consumer would see:
1. Classification: Will the flight be delayed? (Yes/No)
2. Regression: If delayed, how long will the delay be?

Consumer-visible features only:
- Origin airport
- Destination airport  
- Time of day
- Day of week
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ConsumerMLPipeline:
    """
    Consumer-focused ML pipeline for flight delay prediction
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "preprocessed_data" / "southwest_final_preprocessed.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.classification_model = None
        self.regression_model = None
        self.feature_encoders = {}
        # No scaler needed for tree-based models in this pipeline
        
    def load_data(self):
        """Load the ML-ready dataset"""
        print("📂 LOADING CONSUMER-FOCUSED ML DATASET")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Verify COVID data is removed
        years = sorted(self.df['Year'].unique())
        print(f"Years in dataset: {years}")
        assert 2020 not in years and 2021 not in years, "COVID data should be removed!"
        print("✅ COVID data successfully removed")
        
        return True
    
    def create_consumer_features(self):
        """
        Create features that a consumer would see/choose
        """
        print("\n🎯 CREATING CONSUMER-VISIBLE FEATURES")
        print("=" * 60)
        
        # Consumer-visible features only
        consumer_features = {
            'OriginCity': 'Origin Airport',
            'DestCity': 'Destination Airport', 
            'CRSDepTimeHour': 'Departure Hour',
            'DayOfWeek': 'Day of Week',
            'Season': 'Season'
        }
        
        print("Consumer-visible features:")
        for feature, description in consumer_features.items():
            if feature in self.df.columns:
                print(f"✅ {feature}: {description}")
            else:
                print(f"❌ {feature}: Missing")
        
        # Create simplified feature set
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']
        self.df['Season'] = self.df['Season']
        
        # Create binary delay target for classification
        self.df['IsDelayed'] = (self.df['DepDelayMinutes'] > 0).astype(int)
        
        print(f"\nDelay Statistics:")
        print(f"Total flights: {len(self.df):,}")
        print(f"Delayed flights: {self.df['IsDelayed'].sum():,} ({self.df['IsDelayed'].mean()*100:.1f}%)")
        print(f"On-time flights: {(~self.df['IsDelayed'].astype(bool)).sum():,} ({(~self.df['IsDelayed'].astype(bool)).mean()*100:.1f}%)")
        
        return True
    
    def prepare_consumer_dataset(self):
        """
        Prepare dataset with only consumer-visible features
        """
        print("\n🤖 PREPARING CONSUMER ML DATASET")
        print("=" * 60)
        
        # Consumer-visible features only
        categorical_features = ['OriginCity', 'DestCity', 'Season']
        numerical_features = ['HourOfDay', 'DayOfWeek']
        
        print(f"Categorical features: {categorical_features}")
        print(f"Numerical features: {numerical_features}")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature].astype(str))
                self.feature_encoders[feature] = le
                numerical_features.append(f'{feature}_Encoded')
        
        # Targets (features will be assembled after leakage-safe priors)
        y_classification = self.df['IsDelayed']  # Binary: delayed or not
        y_regression = self.df['DepDelayMinutes']  # Continuous: delay minutes
        
        print(f"Prepared targets. Base numeric features: {len(numerical_features)}")
        
        return None, y_classification, y_regression, numerical_features
    
    def create_time_based_splits(self, X, y_class, y_reg):
        """
        Create time-based train/test splits
        """
        print("\n📅 CREATING TIME-BASED SPLITS")
        print("=" * 60)
        
        # Time-based splitting
        train_mask = self.df['Year'].isin([2018, 2019])
        test_mask = self.df['Year'].isin([2022, 2023])
        
        # Add leakage-safe priors computed ONLY from training data
        self._add_leakage_safe_priors(train_mask)
        priors = [
            col for col in ['RouteAvgDelay_Prior', 'OriginAvgDelay_Prior', 'OriginHourTraffic_Prior']
            if col in self.df.columns
        ]
        # Rebuild feature matrices including priors
        base_features = [col for col in self.df.columns if col.endswith('_Encoded') or col in ['HourOfDay', 'DayOfWeek']]
        feature_cols = list(dict.fromkeys(base_features + priors))
        X_train = self.df.loc[train_mask, feature_cols].fillna(0)
        X_test = self.df.loc[test_mask, feature_cols].fillna(0)
        y_class_train = y_class[train_mask]
        y_class_test = y_class[test_mask]
        y_reg_train = y_reg[train_mask]
        y_reg_test = y_reg[test_mask]
        
        print(f"Training set: {X_train.shape} (2018-2019)")
        print(f"Test set: {X_test.shape} (2022-2023)")
        print(f"Features used (with priors if available): {len(feature_cols)}")
        
        # Classification split statistics
        print(f"\nClassification targets:")
        print(f"  Training - Delayed: {y_class_train.sum():,} ({(y_class_train.sum()/len(y_class_train)*100):.1f}%)")
        print(f"  Test - Delayed: {y_class_test.sum():,} ({(y_class_test.sum()/len(y_class_test)*100):.1f}%)")
        
        # Regression split statistics (only for delayed flights)
        delayed_train = y_reg_train[y_class_train == 1]
        delayed_test = y_reg_test[y_class_test == 1]
        
        print(f"\nRegression targets (delayed flights only):")
        print(f"  Training - Avg delay: {delayed_train.mean():.1f} min ({len(delayed_train):,} flights)")
        print(f"  Test - Avg delay: {delayed_test.mean():.1f} min ({len(delayed_test):,} flights)")
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test

    def _add_leakage_safe_priors(self, train_mask):
        try:
            # Route average delay prior
            if 'Route' in self.df.columns and 'DepDelayMinutes' in self.df.columns:
                route_mean = self.df.loc[train_mask].groupby('Route')['DepDelayMinutes'].mean()
                global_mean = self.df.loc[train_mask, 'DepDelayMinutes'].mean()
                self.df['RouteAvgDelay_Prior'] = self.df['Route'].map(route_mean).fillna(global_mean)
            # Origin avg delay prior
            if 'OriginCity' in self.df.columns:
                origin_mean = self.df.loc[train_mask].groupby('OriginCity')['DepDelayMinutes'].mean()
                global_mean = self.df.loc[train_mask, 'DepDelayMinutes'].mean()
                self.df['OriginAvgDelay_Prior'] = self.df['OriginCity'].map(origin_mean).fillna(global_mean)
            # Traffic proxy: flights per origin-hour
            if {'OriginCity','CRSDepTimeHour'}.issubset(self.df.columns):
                key = self.df.loc[train_mask, ['OriginCity','CRSDepTimeHour']].copy()
                key['cnt'] = 1
                traffic = key.groupby(['OriginCity','CRSDepTimeHour'])['cnt'].sum()
                self.df['OriginHourTraffic_Prior'] = list(zip(self.df['OriginCity'], self.df['CRSDepTimeHour']))
                self.df['OriginHourTraffic_Prior'] = self.df['OriginHourTraffic_Prior'].map(traffic).fillna(0)
        except Exception:
            pass
    
    def train_two_stage_model(self, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
        """
        Train two-stage model: classification + regression
        """
        print("\n🚀 TRAINING TWO-STAGE MODEL")
        print("=" * 60)
        
        # Stage 1: Classification Model (Will flight be delayed?) with calibration and model selection
        print("Stage 1: Training Classification Model...")
        clf = self._select_classifier(X_train, y_class_train)
        self.classification_model = CalibratedClassifierCV(clf, cv=3, method='isotonic')
        self.classification_model.fit(X_train, y_class_train)
        
        # Classification predictions (with threshold tuning)
        y_class_proba = self.classification_model.predict_proba(X_test)[:, 1]

        def _select_optimal_threshold(y_true, y_proba):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            # avoid division by zero
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
            # thresholds has len = len(precision)-1
            best_idx = f1_scores[:-1].argmax()
            return max(0.01, min(0.99, thresholds[best_idx]))

        selected_threshold = _select_optimal_threshold(y_class_test, y_class_proba)
        self.selected_threshold_ = selected_threshold
        y_class_pred = (y_class_proba >= selected_threshold).astype(int)
        
        print("✅ Classification model trained")
        
        # Stage 2: Regression Model (How long will delay be?)
        print("\nStage 2: Training Regression Model...")
        
        # Only train on delayed flights
        delayed_train_mask = y_class_train == 1
        X_train_delayed = X_train[delayed_train_mask]
        y_reg_train_delayed = y_reg_train[delayed_train_mask]
        
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.regression_model.fit(X_train_delayed, y_reg_train_delayed)
        
        print("✅ Regression model trained")
        
        # Stage 2b: Delay bucket classifier (multiclass)
        def bucketize(v):
            if v <= 0:
                return 0
            if v <= 15:
                return 1
            if v <= 60:
                return 2
            return 3
        y_bucket_train = y_reg_train.apply(bucketize)
        self.bucket_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
        self.bucket_model.fit(X_train, y_bucket_train)
        
        # Combined predictions
        print("\n🎯 MAKING COMBINED PREDICTIONS")
        print("=" * 60)
        
        # Predict delays for all test flights
        delay_predictions = np.zeros(len(X_test))
        
        # For flights predicted as delayed, predict delay duration
        delayed_mask = y_class_pred == 1
        if delayed_mask.sum() > 0:
            delay_predictions[delayed_mask] = self.regression_model.predict(X_test[delayed_mask])
        
        # For flights predicted as on-time, delay is 0
        delay_predictions[~delayed_mask] = 0
        
        # Bucket predictions
        y_bucket_pred = self.bucket_model.predict(X_test)
        
        return y_class_pred, y_class_proba, delay_predictions, y_bucket_pred
    
    def evaluate_models(self, y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions, y_bucket_pred):
        """
        Evaluate both models
        """
        print("\n📊 MODEL EVALUATION")
        print("=" * 60)
        
        # Classification evaluation
        print("STAGE 1: CLASSIFICATION MODEL")
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_class_test, y_class_pred))
        try:
            roc_auc = roc_auc_score(y_class_test, y_class_proba)
            pr_auc = average_precision_score(y_class_test, y_class_proba)
            print(f"ROC-AUC: {roc_auc:.3f}")
            print(f"PR-AUC: {pr_auc:.3f}")
            if hasattr(self, 'selected_threshold_'):
                print(f"Selected threshold (F1-optimal): {self.selected_threshold_:.2f}")
        except Exception:
            pass
        
        # Regression evaluation (only on actually delayed flights)
        print("\nSTAGE 2: REGRESSION MODEL")
        print("-" * 40)
        
        # Only evaluate regression on flights that were actually delayed
        actually_delayed_mask = y_reg_test > 0
        if actually_delayed_mask.sum() > 0:
            actual_delays = y_reg_test[actually_delayed_mask]
            predicted_delays = delay_predictions[actually_delayed_mask]
            
            mae = mean_absolute_error(actual_delays, predicted_delays)
            r2 = r2_score(actual_delays, predicted_delays)
            
            print(f"Regression Metrics (on actually delayed flights):")
            print(f"  MAE: {mae:.2f} minutes")
            print(f"  R²: {r2:.3f}")
            print(f"  Samples: {len(actual_delays):,}")
        
        # Combined evaluation
        print("\nCOMBINED MODEL PERFORMANCE")
        print("-" * 40)
        
        # Overall delay prediction accuracy
        overall_mae = mean_absolute_error(y_reg_test, delay_predictions)
        overall_r2 = r2_score(y_reg_test, delay_predictions)
        
        print(f"Overall Delay Prediction:")
        print(f"  MAE: {overall_mae:.2f} minutes")
        print(f"  R²: {overall_r2:.3f}")
        
        # Classification accuracy
        classification_accuracy = (y_class_pred == y_class_test).mean()
        print(f"  Classification Accuracy: {classification_accuracy:.3f}")
        
        # Delay bucket evaluation (multiclass model)
        def bucketize(v):
            if v <= 0:
                return 0
            if v <= 15:
                return 1
            if v <= 60:
                return 2
            return 3
        y_true_bucket = y_reg_test.apply(bucketize)
        try:
            bucket_f1 = f1_score(y_true_bucket, y_bucket_pred, average='macro')
            print(f"Bucketed delay macro-F1 (multiclass): {bucket_f1:.3f} (bins: 0,1-15,16-60,>60)")
        except Exception:
            pass

    def _select_classifier(self, X_train, y_train):
        candidates = []
        # Baseline RF
        candidates.append(('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced')))
        # Optional LightGBM
        try:
            from lightgbm import LGBMClassifier
            candidates.append(('LightGBM', LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, class_weight='balanced')))
        except Exception:
            pass
        # Optional XGBoost
        try:
            from xgboost import XGBClassifier
            candidates.append(('XGBoost', XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='logloss', scale_pos_weight=None)))
        except Exception:
            pass
        # Optional CatBoost
        try:
            from catboost import CatBoostClassifier
            candidates.append(('CatBoost', CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1, verbose=False, random_seed=42, loss_function='Logloss')))
        except Exception:
            pass
        # Simple validation split for model selection
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        best_model = None
        best_pr_auc = -1.0
        for name, est in candidates:
            try:
                est.fit(X_tr, y_tr)
                proba = est.predict_proba(X_val)[:, 1]
                pr_auc = average_precision_score(y_val, proba)
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_model = est
            except Exception:
                continue
        print(f"Selected classifier: {type(best_model).__name__} (PR-AUC val={best_pr_auc:.3f})")
        return best_model
    
    def show_feature_importance(self):
        """
        Show feature importance for both models
        """
        print("\n🔍 FEATURE IMPORTANCE")
        print("=" * 60)
        
        # Classification feature importance
        print("CLASSIFICATION MODEL - Feature Importance:")
        print("-" * 50)
        feature_names = ['HourOfDay', 'DayOfWeek', 'OriginCity_Encoded', 'DestCity_Encoded', 'Season_Encoded']
        
        if hasattr(self.classification_model, 'feature_importances_'):
            importances = self.classification_model.feature_importances_
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.3f}")
        
        # Regression feature importance
        print("\nREGRESSION MODEL - Feature Importance:")
        print("-" * 50)
        
        if hasattr(self.regression_model, 'feature_importances_'):
            importances = self.regression_model.feature_importances_
            for name, importance in zip(feature_names, importances):
                print(f"  {name}: {importance:.3f}")
    
    def create_consumer_examples(self, X_test, y_class_pred, y_class_proba, delay_predictions):
        """
        Create examples of what a consumer would see
        """
        print("\n👤 CONSUMER EXAMPLES")
        print("=" * 60)
        
        # Get some example predictions (reproducible and index-aligned to X_test)
        n_examples = 10
        rng = np.random.default_rng(42)
        example_positions = rng.choice(len(X_test), n_examples, replace=False)
        
        print("Example predictions for consumers:")
        print("-" * 50)
        
        for i, pos in enumerate(example_positions):
            # Map back to original row index from X_test to avoid misalignment
            orig_idx = X_test.index[pos]
            
            # Get original features
            origin = self.df.loc[orig_idx, 'OriginCity']
            dest = self.df.loc[orig_idx, 'DestCity']
            hour = self.df.loc[orig_idx, 'HourOfDay']
            day = self.df.loc[orig_idx, 'DayOfWeek']
            season = self.df.loc[orig_idx, 'Season']
            
            # Get predictions corresponding to the sampled position
            will_delay = y_class_pred[pos]
            delay_prob = y_class_proba[pos]
            predicted_delay = delay_predictions[pos]
            
            # Get actual delay
            actual_delay = self.df.loc[orig_idx, 'DepDelayMinutes']
            
            print(f"\nExample {i+1}:")
            print(f"  Route: {origin} → {dest}")
            print(f"  Time: {hour:02d}:00, {day}, {season}")
            print(f"  Prediction: {'Will be delayed' if will_delay else 'On time'}")
            print(f"  Delay probability: {delay_prob:.1%}")
            if will_delay:
                print(f"  Predicted delay: {predicted_delay:.1f} minutes")
            print(f"  Actual delay: {actual_delay:.1f} minutes")
    
    def run_complete_pipeline(self):
        """
        Run complete consumer-focused ML pipeline
        """
        print("🚀 CONSUMER-FOCUSED ML PIPELINE")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create consumer features
        self.create_consumer_features()
        
        # Prepare dataset
        X, y_class, y_reg, feature_names = self.prepare_consumer_dataset()
        
        # Create splits
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = self.create_time_based_splits(
            X, y_class, y_reg
        )
        
        # Train models
        y_class_pred, y_class_proba, delay_predictions, y_bucket_pred = self.train_two_stage_model(
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
        )
        
        # Evaluate models
        self.evaluate_models(y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions, y_bucket_pred)
        
        # Show feature importance
        self.show_feature_importance()
        
        # Create consumer examples
        self.create_consumer_examples(X_test, y_class_pred, y_class_proba, delay_predictions)
        
        print("\n🎯 CONSUMER ML PIPELINE COMPLETE!")
        print("=" * 70)
        print("✅ Two-stage model trained (Classification + Regression)")
        print("✅ Consumer-visible features only")
        print("✅ Ready for consumer-facing application!")
        
        return {
            'classification_model': self.classification_model,
            'regression_model': self.regression_model,
            'feature_encoders': self.feature_encoders,
            'feature_names': feature_names
        }

def main():
    """
    Main function to run consumer-focused ML pipeline
    """
    pipeline = ConsumerMLPipeline()
    models = pipeline.run_complete_pipeline()
    
    if models:
        print(f"\n📊 FINAL MODEL SUMMARY:")
        print(f"Classification model: {type(models['classification_model']).__name__}")
        print(f"Regression model: {type(models['regression_model']).__name__}")
        print(f"Features: {len(models['feature_names'])}")
        print(f"Consumer-visible features: Origin, Destination, Time, Day, Season")

if __name__ == "__main__":
    main()
