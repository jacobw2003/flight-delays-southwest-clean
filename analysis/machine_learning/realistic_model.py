#!/usr/bin/env python3
"""
Realistic Southwest Airlines Model - No Data Leakage

This script creates a realistic model using only features available BEFORE flight departure:
- No historical delay data (data leakage)
- No operational data from the same flight
- Only features a consumer/Southwest would know in advance
- Focus on what's actually predictable vs what's not
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RealisticSouthwestModel:
    """
    Realistic model with only pre-flight features
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
        self.bucket_classifier = None
        self.feature_encoders = {}
        self.y_bucket = None
        
    def load_data(self):
        """Load data"""
        print("📂 LOADING DATA FOR REALISTIC MODEL")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path, nrows=500000)  # Larger sample to ensure we have test data
        print(f"Dataset loaded: {self.df.shape}")
        
        return True
    
    def create_realistic_features(self):
        """
        Create only realistic pre-flight features
        """
        print("\n🎯 CREATING REALISTIC PRE-FLIGHT FEATURES")
        print("=" * 60)
        
        # ONLY features available BEFORE flight departure
        realistic_features = {
            'CRSDepTimeHour': 'Scheduled departure hour',
            'DayOfWeek': 'Day of week',
            'Season': 'Season',
            'OriginCity': 'Origin airport',
            'DestCity': 'Destination airport',
            'Distance': 'Flight distance'
        }
        
        print("REALISTIC PRE-FLIGHT FEATURES:")
        print("-" * 40)
        for feature, description in realistic_features.items():
            if feature in self.df.columns:
                print(f"✅ {feature}: {description}")
            else:
                print(f"❌ {feature}: Missing")
        
        # Create enhanced time features (still realistic)
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']
        self.df['Season'] = self.df['Season']
        
        # Time-based features (realistic)
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        self.df['IsEvening'] = (self.df['HourOfDay'] >= 18).astype(int)
        self.df['IsMorning'] = (self.df['HourOfDay'] <= 10).astype(int)
        self.df['IsPeakHour'] = ((self.df['HourOfDay'] >= 7) & (self.df['HourOfDay'] <= 9) | 
                                (self.df['HourOfDay'] >= 17) & (self.df['HourOfDay'] <= 19)).astype(int)
        
        # Route features (realistic)
        self.df['Route'] = self.df['OriginCity'] + ' → ' + self.df['DestCity']
        
        # Distance features (realistic)
        self.df['Distance'] = self.df['Distance']
        self.df['DistanceGroup'] = pd.cut(self.df['Distance'], 
                                         bins=[0, 500, 1000, 1500, 2000, float('inf')],
                                         labels=[1, 2, 3, 4, 5])
        
        # Create binary delay target
        self.df['IsDelayed'] = (self.df['DepDelayMinutes'] > 0).astype(int)
        
        print(f"\n✅ Realistic features created")
        print(f"Total realistic features: 11")
        
        return True
    
    def prepare_realistic_dataset(self):
        """
        Prepare dataset with only realistic features
        """
        print("\n🤖 PREPARING REALISTIC DATASET")
        print("=" * 60)
        
        # Realistic features only
        categorical_features = ['OriginCity', 'DestCity', 'Season']
        numerical_features = ['HourOfDay', 'DayOfWeek', 'IsWeekend', 'IsEvening', 'IsMorning', 'IsPeakHour', 'Distance', 'DistanceGroup']
        
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
        
        # Create feature matrix
        X = self.df[numerical_features].copy()
        
        # Handle categorical columns properly
        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].astype(float)
        
        X = X.fillna(0)
        
        # Create targets
        y_classification = self.df['IsDelayed']
        y_regression = self.df['DepDelayMinutes']
        # Create delay buckets for multiclass classification (primary output)
        def bucketize_minutes(v: float) -> int:
            if v <= 0:
                return 0
            if v <= 15:
                return 1
            if v <= 60:
                return 2
            return 3
        self.y_bucket = y_regression.apply(bucketize_minutes)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Classification target shape: {y_classification.shape}")
        print(f"Regression target shape: {y_regression.shape}")
        
        return X, y_classification, y_regression, numerical_features
    
    def create_time_based_splits(self, X, y_class, y_reg):
        """
        Create time-based train/test splits
        """
        print("\n📅 CREATING TIME-BASED SPLITS")
        print("=" * 60)
        
        # Since we only have 2018 data in sample, use random split instead
        # In production, you would use time-based splits
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print("Note: Using random split due to limited year data in sample")
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    
    def _select_classifier(self, X_train, y_train):
        """
        Quickly select a strong classifier using a small validation split.
        Candidates: RandomForest + optional LightGBM/XGBoost/CatBoost.
        Selection metric: PR-AUC on validation.
        """
        candidates = []
        candidates.append((
            'RandomForest',
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        ))
        # Optional boosters
        try:
            from lightgbm import LGBMClassifier
            candidates.append((
                'LightGBM',
                LGBMClassifier(
                    n_estimators=400,
                    max_depth=-1,
                    learning_rate=0.07,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            ))
        except Exception:
            pass
        try:
            from xgboost import XGBClassifier
            candidates.append((
                'XGBoost',
                XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            ))
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier
            candidates.append((
                'CatBoost',
                CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.08,
                    verbose=False,
                    random_seed=42
                )
            ))
        except Exception:
            pass

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
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

    def _select_threshold_for_recall(self, y_true, y_proba, target_recall: float = 0.80) -> float:
        """
        Choose the highest-precision threshold that still achieves at least target_recall.
        Falls back to F1-optimal if none meet the recall target.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # thresholds has len = len(precision) - 1
        candidate_thresholds = []
        for idx, thr in enumerate(thresholds):
            r = recall[idx]
            p = precision[idx]
            if r >= target_recall:
                candidate_thresholds.append((p, thr))
        if candidate_thresholds:
            # Maximize precision among candidates
            _, best_thr = max(candidate_thresholds, key=lambda x: x[0])
            return float(max(0.01, min(0.99, best_thr)))

        # Fallback: F1-optimal
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
        best_idx = f1_scores[:-1].argmax()
        return float(max(0.01, min(0.99, thresholds[best_idx])))

    def train_realistic_models(self, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
        """
        Train realistic models
        """
        print("\n🚀 TRAINING REALISTIC MODELS")
        print("=" * 60)
        
        # Classification model with quick model selection + isotonic calibration
        print("Training Classification Model...")
        base_clf = self._select_classifier(X_train, y_class_train)
        self.classification_model = CalibratedClassifierCV(base_clf, cv=3, method='isotonic')
        self.classification_model.fit(X_train, y_class_train)

        # Choose threshold on a validation split to hit recall target
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_class_train, test_size=0.2, random_state=42, stratify=y_class_train
        )
        self.classification_model.fit(X_tr, y_tr)
        val_proba = self.classification_model.predict_proba(X_val)[:, 1]
        selected_threshold = self._select_threshold_for_recall(y_val, val_proba, target_recall=0.80)
        self.selected_threshold_ = selected_threshold

        # Evaluate on test using the selected threshold
        y_class_proba = self.classification_model.predict_proba(X_test)[:, 1]
        y_class_pred = (y_class_proba >= selected_threshold).astype(int)
        
        print("✅ Classification model trained")
        
        # Regression model (only on delayed flights)
        print("\nTraining Regression Model...")
        delayed_train_mask = y_class_train == 1
        X_train_delayed = X_train[delayed_train_mask]
        y_reg_train_delayed = y_reg_train[delayed_train_mask]
        
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.regression_model.fit(X_train_delayed, y_reg_train_delayed)
        
        print("✅ Regression model trained")

        # Multiclass delay-bucket classifier (primary output)
        print("\nTraining Delay-Bucket Classifier (primary output)...")
        # Align bucket targets with train/test indices
        y_bucket_train = self.y_bucket.loc[X_train.index]
        y_bucket_test = self.y_bucket.loc[X_test.index]
        self.bucket_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.bucket_classifier.fit(X_train, y_bucket_train)
        y_bucket_pred = self.bucket_classifier.predict(X_test)
        self._last_bucket_true_ = y_bucket_test
        self._last_bucket_pred_ = y_bucket_pred
        
        # Combined predictions
        delay_predictions = np.zeros(len(X_test))
        delayed_mask = y_class_pred == 1
        if delayed_mask.sum() > 0:
            delay_predictions[delayed_mask] = self.regression_model.predict(X_test[delayed_mask])
        
        return y_class_pred, y_class_proba, delay_predictions
    
    def evaluate_realistic_models(self, y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions):
        """
        Evaluate realistic models
        """
        print("\n📊 REALISTIC MODEL EVALUATION")
        print("=" * 60)
        
        # Classification evaluation (binary delay>0)
        print("CLASSIFICATION MODEL:")
        print("-" * 30)
        print(classification_report(y_class_test, y_class_pred))
        try:
            roc_auc = roc_auc_score(y_class_test, y_class_proba)
            pr_auc = average_precision_score(y_class_test, y_class_proba)
            print(f"ROC-AUC: {roc_auc:.3f}")
            print(f"PR-AUC: {pr_auc:.3f}")
            if hasattr(self, 'selected_threshold_'):
                # Compute achieved precision/recall on test at the chosen threshold
                thr = float(self.selected_threshold_)
                y_hat = (y_class_proba >= thr).astype(int)
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(y_class_test, y_hat)
                rec = recall_score(y_class_test, y_hat)
                print(f"Selected threshold (recall-target=0.80): {thr:.2f} | Test Precision={prec:.3f}, Recall={rec:.3f}")
        except Exception:
            pass
        
        # Regression evaluation
        print("\nREGRESSION MODEL:")
        print("-" * 30)
        actually_delayed_mask = y_reg_test > 0
        if actually_delayed_mask.sum() > 0:
            actual_delays = y_reg_test[actually_delayed_mask]
            predicted_delays = delay_predictions[actually_delayed_mask]
            
            mae = mean_absolute_error(actual_delays, predicted_delays)
            r2 = r2_score(actual_delays, predicted_delays)
            
            print(f"MAE: {mae:.2f} minutes")
            print(f"R²: {r2:.3f}")
            print(f"Samples: {len(actual_delays):,}")
        
        # Overall performance (note: regression is de-prioritized)
        print("\nOVERALL PERFORMANCE:")
        print("-" * 30)
        overall_mae = mean_absolute_error(y_reg_test, delay_predictions)
        overall_r2 = r2_score(y_reg_test, delay_predictions)
        classification_accuracy = (y_class_pred == y_class_test).mean()
        
        print(f"Overall MAE: {overall_mae:.2f} minutes")
        print(f"Overall R²: {overall_r2:.3f}")
        print(f"Classification Accuracy: {classification_accuracy:.3f}")
        
        # Delay-bucket classifier evaluation (PRIMARY)
        try:
            y_bucket_true = getattr(self, '_last_bucket_true_', None)
            y_bucket_pred = getattr(self, '_last_bucket_pred_', None)
            if y_bucket_true is not None and y_bucket_pred is not None:
                bucket_macro_f1 = f1_score(y_bucket_true, y_bucket_pred, average='macro')
                print("\nDELAY-BUCKET CLASSIFIER (PRIMARY):")
                print("-" * 30)
                print(classification_report(y_bucket_true, y_bucket_pred, digits=3))
                print(f"Bucketed delay macro-F1: {bucket_macro_f1:.3f} (bins: 0,1-15,16-60,>60)")
        except Exception:
            pass
        
        return overall_r2, overall_mae, classification_accuracy
    
    def analyze_realistic_insights(self, X_test, y_class_pred, y_class_proba, delay_predictions):
        """
        Analyze realistic insights
        """
        print("\n🎯 REALISTIC OPERATIONAL INSIGHTS")
        print("=" * 60)
        
        # Feature importance
        print("FEATURE IMPORTANCE:")
        print("-" * 30)
        
        feature_names = ['HourOfDay', 'DayOfWeek', 'IsWeekend', 'IsEvening', 'IsMorning', 'IsPeakHour', 
                        'Distance', 'DistanceGroup', 'OriginCity_Encoded', 'DestCity_Encoded', 'Season_Encoded']
        
        if hasattr(self.classification_model, 'feature_importances_'):
            importances = self.classification_model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("Classification Model - Most Important Features:")
            for name, importance in feature_importance[:5]:
                print(f"  {name}: {importance:.3f}")
        
        # Operational insights
        print(f"\nOPERATIONAL INSIGHTS:")
        print("-" * 30)
        
        # Analyze worst performing routes
        route_delays = self.df.groupby('Route')['DepDelayMinutes'].agg(['count', 'mean']).round(2)
        route_delays.columns = ['Flight_Count', 'Avg_Delay']
        worst_routes = route_delays[route_delays['Flight_Count'] >= 100].nlargest(5, 'Avg_Delay')
        
        print("Top 5 Problematic Routes:")
        for route, stats in worst_routes.iterrows():
            print(f"  {route}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Analyze worst performing times
        hourly_delays = self.df.groupby('CRSDepTimeHour')['DepDelayMinutes'].mean().round(2)
        worst_hours = hourly_delays.nlargest(3)
        
        print(f"\nWorst Departure Hours:")
        for hour, avg_delay in worst_hours.items():
            print(f"  {hour:02d}:00: {avg_delay:.1f} min avg")
        
        # What's actually predictable
        print(f"\nWHAT'S ACTUALLY PREDICTABLE:")
        print("-" * 30)
        print("✅ Time-based patterns (evening flights)")
        print("✅ Route-specific trends")
        print("✅ Seasonal variations")
        print("✅ Airport-specific delays")
        print("❌ Exact delay duration")
        print("❌ Random external events")
        print("❌ Weather-related delays")
        print("❌ Air traffic control delays")
    
    def run_realistic_analysis(self):
        """
        Run realistic model analysis
        """
        print("🚀 REALISTIC SOUTHWEST MODEL ANALYSIS")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create realistic features
        self.create_realistic_features()
        
        # Prepare dataset
        X, y_class, y_reg, feature_names = self.prepare_realistic_dataset()
        
        # Create splits
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = self.create_time_based_splits(
            X, y_class, y_reg
        )
        
        # Train models
        y_class_pred, y_class_proba, delay_predictions = self.train_realistic_models(
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
        )
        
        # Evaluate models
        overall_r2, overall_mae, classification_accuracy = self.evaluate_realistic_models(
            y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions
        )
        
        # Analyze insights
        self.analyze_realistic_insights(X_test, y_class_pred, y_class_proba, delay_predictions)
        
        print("\n🎯 REALISTIC ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"✅ Realistic R²: {overall_r2:.3f}")
        print(f"✅ Realistic MAE: {overall_mae:.2f} minutes")
        print(f"✅ Classification Accuracy: {classification_accuracy:.3f}")
        print(f"✅ No data leakage - only pre-flight features")
        print(f"✅ Ready for realistic operational use!")
        
        return {
            'r2': overall_r2,
            'mae': overall_mae,
            'classification_accuracy': classification_accuracy,
            'feature_names': feature_names
        }

def main():
    """Main function"""
    analyzer = RealisticSouthwestModel()
    results = analyzer.run_realistic_analysis()

if __name__ == "__main__":
    main()
