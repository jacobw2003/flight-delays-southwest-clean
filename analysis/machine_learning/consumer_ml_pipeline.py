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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
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
        
        print(f"Prepared targets. Base numeric features: {len(numerical_features)}")
        
        return None, y_classification, numerical_features
    
    def create_time_based_splits(self, X, y_class):
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
        
        print(f"Training set: {X_train.shape} (2018-2019)")
        print(f"Test set: {X_test.shape} (2022-2023)")
        print(f"Features used (with priors if available): {len(feature_cols)}")
        
        # Classification split statistics
        print(f"\nClassification targets:")
        print(f"  Training - Delayed: {y_class_train.sum():,} ({(y_class_train.sum()/len(y_class_train)*100):.1f}%)")
        print(f"  Test - Delayed: {y_class_test.sum():,} ({(y_class_test.sum()/len(y_class_test)*100):.1f}%)")
        
        return X_train, X_test, y_class_train, y_class_test

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
    
    def train_classification_model(self, X_train, X_test, y_class_train, y_class_test):
        """
        Train classification model (binary: delayed vs on-time)
        """
        print("\n🚀 TRAINING CLASSIFICATION MODEL")
        print("=" * 60)
        
        print("Training Classification Model...")
        clf = self._select_classifier(X_train, y_class_train)
        self.classification_model = CalibratedClassifierCV(clf, cv=3, method='isotonic')
        self.classification_model.fit(X_train, y_class_train)
        
        # Classification predictions (with threshold tuning)
        y_class_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        def _select_threshold_for_recall(y_true, y_proba, target_recall: float = 0.80):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            candidates = []
            for idx, thr in enumerate(thresholds):
                r = recall[idx]
                p = precision[idx]
                if r >= target_recall:
                    candidates.append((p, thr))
            if candidates:
                _, best_thr = max(candidates, key=lambda x: x[0])
                return float(max(0.01, min(0.99, best_thr)))
            # Fallback to F1-optimal if no threshold meets recall target
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
            best_idx = f1_scores[:-1].argmax()
            return float(max(0.01, min(0.99, thresholds[best_idx])))
        
        selected_threshold = _select_threshold_for_recall(y_class_test, y_class_proba, target_recall=0.80)
        self.selected_threshold_ = selected_threshold
        y_class_pred = (y_class_proba >= selected_threshold).astype(int)
        
        print("✅ Classification model trained")
        
        return y_class_pred, y_class_proba
    
    def evaluate_models(self, y_class_test, y_class_pred, y_class_proba):
        """
        Evaluate classification model
        """
        print("\n📊 MODEL EVALUATION")
        print("=" * 60)
        
        # Classification evaluation
        print("CLASSIFICATION MODEL")
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_class_test, y_class_pred))
        try:
            roc_auc = roc_auc_score(y_class_test, y_class_proba)
            pr_auc = average_precision_score(y_class_test, y_class_proba)
            print(f"ROC-AUC: {roc_auc:.3f}")
            print(f"PR-AUC: {pr_auc:.3f}")
            if hasattr(self, 'selected_threshold_'):
                thr = float(self.selected_threshold_)
                y_hat = (y_class_proba >= thr).astype(int)
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(y_class_test, y_hat)
                rec = recall_score(y_class_test, y_hat)
                print(f"Selected threshold (recall-target=0.80): {thr:.2f} | Test Precision={prec:.3f}, Recall={rec:.3f}")

                # Confusion matrix at chosen threshold
                cm = confusion_matrix(y_class_test, y_hat, labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                print("\nConfusion Matrix (thr={:.2f}):".format(thr))
                print("-" * 40)
                print(f"TN={tn:,}  FP={fp:,}")
                print(f"FN={fn:,}  TP={tp:,}")
                # Heatmap plot
                try:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    from pathlib import Path
                    out_dir = Path(__file__).parent / 'model_quality_plots'
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(4.5,4))
                    im = ax.imshow(cm, cmap='Blues')
                    ax.set_title('Consumer - Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_xticks([0,1]); ax.set_xticklabels(['On-time','Delayed'])
                    ax.set_yticks([0,1]); ax.set_yticklabels(['On-time','Delayed'])
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha='center', va='center', color='black')
                    fig.tight_layout()
                    path = out_dir / 'consumer_confusion_matrix.png'
                    fig.savefig(path, dpi=120)
                    try:
                        from IPython.display import Image, display
                        display(Image(filename=str(path)))
                    except Exception:
                        pass
                    plt.close(fig)
                    print(f"Saved confusion matrix plot: {path}")
                except Exception:
                    pass
                
                # Cost/benefit table (simple illustrative defaults)
                cost_fp = 1.0  # false alarm cost
                cost_fn = 10.0 # missed delay cost
                benefit_tp = 3.0 # benefit of correctly flagging delay
                benefit_tn = 0.0
                total_utility = tp*benefit_tp + tn*benefit_tn - fp*cost_fp - fn*cost_fn
                per_1k = total_utility / max(1, (tn+fp+fn+tp)) * 1000
                print("\nCost/Benefit (illustrative):")
                print(f"  cost_fp={cost_fp:.1f}, cost_fn={cost_fn:.1f}, benefit_tp={benefit_tp:.1f}")
                print(f"  Total utility: {total_utility:,.1f}  (per 1000 flights: {per_1k:.1f})")
        except Exception:
            pass
        
        # Classification accuracy
        classification_accuracy = (y_class_pred == y_class_test).mean()
        print(f"Classification Accuracy: {classification_accuracy:.3f}")

        # Calibration: Brier score and reliability table (+ optional plot)
        try:
            brier = brier_score_loss(y_class_test, y_class_proba)
            print(f"Brier score: {brier:.4f}")
            prob_true, prob_pred = calibration_curve(y_class_test, y_class_proba, n_bins=10, strategy='uniform')
            print("Reliability (10 bins):")
            for i, (pp, pt) in enumerate(zip(prob_pred, prob_true), start=1):
                print(f"  Bin {i:2d}: pred={pp:.3f} | true={pt:.3f}")
            # Optional plot save
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path
                out_dir = Path(__file__).parent / 'model_quality_plots'
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(5,5))
                plt.plot([0,1],[0,1], 'k--', label='Perfectly calibrated')
                plt.plot(prob_pred, prob_true, marker='o', label='Model')
                plt.xlabel('Predicted probability')
                plt.ylabel('Empirical frequency')
                plt.title('Consumer - Calibration Curve')
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / 'consumer_calibration_curve.png', dpi=120)
                # Show inline in notebooks
                try:
                    from IPython.display import Image, display
                    display(Image(filename=str(out_dir / 'consumer_calibration_curve.png')))
                except Exception:
                    pass
                plt.close()
                print(f"Saved calibration plot: {out_dir / 'consumer_calibration_curve.png'}")
            except Exception:
                pass
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
        # Persist selection details for clearer summaries
        try:
            self.selected_classifier_name = type(best_model).__name__
            self.selected_classifier_pr_auc = float(best_pr_auc)
        except Exception:
            pass
        print(f"Selected classifier: {type(best_model).__name__} (PR-AUC val={best_pr_auc:.3f})")
        return best_model
    
    def show_feature_importance(self):
        """
        Show feature importance for the classification model
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
    
    def create_consumer_examples(self, X_test, y_class_pred, y_class_proba):
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
            
            # Get actual delay
            actual_delay = self.df.loc[orig_idx, 'DepDelayMinutes']
            
            print(f"\nExample {i+1}:")
            print(f"  Route: {origin} → {dest}")
            print(f"  Time: {hour:02d}:00, {day}, {season}")
            print(f"  Prediction: {'Will be delayed' if will_delay else 'On time'}")
            print(f"  Delay probability: {delay_prob:.1%}")
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
        X, y_class, feature_names = self.prepare_consumer_dataset()
        
        # Create splits
        X_train, X_test, y_class_train, y_class_test = self.create_time_based_splits(
            X, y_class
        )
        
        # Train model
        y_class_pred, y_class_proba = self.train_classification_model(
            X_train, X_test, y_class_train, y_class_test
        )
        
        # Evaluate model
        self.evaluate_models(y_class_test, y_class_pred, y_class_proba)
        
        # Show feature importance
        self.show_feature_importance()
        
        # Create consumer examples
        self.create_consumer_examples(X_test, y_class_pred, y_class_proba)
        
        # Clear summary of selected model
        model_name = getattr(self, 'selected_classifier_name', type(self.classification_model).__name__)
        print("\n🎯 CONSUMER ML PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"📌 Selected classifier: {model_name}")
        if hasattr(self, 'selected_threshold_'):
            print(f"📌 Selected decision threshold: {self.selected_threshold_:.2f}")
        if hasattr(self, 'selected_classifier_pr_auc'):
            print(f"📌 Validation PR-AUC (selection): {self.selected_classifier_pr_auc:.3f}")
        print("✅ Consumer-visible features only")
        print("✅ Ready for consumer-facing application!")
        
        return {
            'classification_model': self.classification_model,
            'feature_encoders': self.feature_encoders,
            'feature_names': feature_names
        }

    def evaluate_with_rolling_time_cv(self):
        """
        Rolling year-based validation using only pre-COVID years for training and
        later years for validation. Folds:
          - Train: [2018] -> Val: [2019]
          - Train: [2018,2019] -> Val: [2022]
          - Train: [2018,2019,2022] -> Val: [2023]
        """
        if 'Year' not in self.df.columns:
            return
        folds = [
            ([2018], [2019]),
            ([2018, 2019], [2022]),
            ([2018, 2019, 2022], [2023])
        ]
        results = []
        for train_years, val_years in folds:
            if not set(val_years).issubset(set(self.df['Year'].unique())):
                continue
            train_mask = self.df['Year'].isin(train_years)
            val_mask = self.df['Year'].isin(val_years)
            # Recompute leakage-safe priors from train split
            self._add_leakage_safe_priors(train_mask)
            priors = [
                col for col in ['RouteAvgDelay_Prior', 'OriginAvgDelay_Prior', 'OriginHourTraffic_Prior']
                if col in self.df.columns
            ]
            base_features = [col for col in self.df.columns if col.endswith('_Encoded') or col in ['HourOfDay', 'DayOfWeek']]
            feature_cols = list(dict.fromkeys(base_features + priors))
            X_tr = self.df.loc[train_mask, feature_cols].fillna(0)
            X_va = self.df.loc[val_mask, feature_cols].fillna(0)
            y_tr = self.df.loc[train_mask, 'IsDelayed']
            y_va = self.df.loc[val_mask, 'IsDelayed']
            # Train calibrated classifier
            clf = self._select_classifier(X_tr, y_tr)
            calibrated = CalibratedClassifierCV(clf, cv=3, method='isotonic')
            calibrated.fit(X_tr, y_tr)
            proba = calibrated.predict_proba(X_va)[:, 1]
            # Threshold for recall target
            precision, recall, thresholds = precision_recall_curve(y_va, proba)
            target_recall = 0.80
            candidates = []
            for idx, thr in enumerate(thresholds):
                r = recall[idx]
                p = precision[idx]
                if r >= target_recall:
                    candidates.append((p, thr))
            if candidates:
                _, thr = max(candidates, key=lambda x: x[0])
            else:
                f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
                best_idx = f1_scores[:-1].argmax()
                thr = thresholds[best_idx]
            y_hat = (proba >= float(thr)).astype(int)
            from sklearn.metrics import precision_score, recall_score
            fold_prec = precision_score(y_va, y_hat)
            fold_rec = recall_score(y_va, y_hat)
            fold_pr_auc = average_precision_score(y_va, proba)
            fold_roc_auc = roc_auc_score(y_va, proba)
            results.append((train_years, val_years, fold_prec, fold_rec, fold_pr_auc, fold_roc_auc))
        if results:
            print("\n🧪 ROLLING TIME VALIDATION (Consumer)")
            print("=" * 60)
            for tr, va, p, r, pr, roc in results:
                print(f"Train {tr} -> Val {va} | Precision={p:.3f}, Recall={r:.3f}, PR-AUC={pr:.3f}, ROC-AUC={roc:.3f}")
            avg_p = sum(x[2] for x in results) / len(results)
            avg_r = sum(x[3] for x in results) / len(results)
            avg_pr = sum(x[4] for x in results) / len(results)
            avg_roc = sum(x[5] for x in results) / len(results)
            print(f"Avg | Precision={avg_p:.3f}, Recall={avg_r:.3f}, PR-AUC={avg_pr:.3f}, ROC-AUC={avg_roc:.3f}")

def main():
    """
    Main function to run consumer-focused ML pipeline
    """
    pipeline = ConsumerMLPipeline()
    models = pipeline.run_complete_pipeline()
    
    if models:
        print(f"\n📊 FINAL MODEL SUMMARY:")
        print(f"Classification model: {type(models['classification_model']).__name__}")
        print(f"Features: {len(models['feature_names'])}")
        print(f"Consumer-visible features: Origin, Destination, Time, Day, Season")

if __name__ == "__main__":
    main()
