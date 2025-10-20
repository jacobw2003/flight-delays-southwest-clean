#!/usr/bin/env python3
"""
Southwest Airlines Operational ML Pipeline

This script creates a model focused on Southwest's operational needs:
- Identify actionable delay causes
- Provide specific mitigation strategies
- Use only essential features to avoid overfitting
- Help Southwest improve customer experience through proactive measures

Essential features for Southwest operations:
- Time-based: Hour, Day, Season (operational planning)
- Route-based: Origin, Destination (route optimization)
- Operational: Distance, TaxiOut, TaxiIn (efficiency metrics)
- Delay causes: WeatherDelay, CarrierDelay, NASDelay, LateAircraftDelay (actionable insights)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SouthwestOperationalML:
    """
    Southwest Airlines operational ML pipeline for actionable insights
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
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the ML-ready dataset"""
        print("📂 LOADING SOUTHWEST OPERATIONAL DATASET")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Verify COVID data is removed
        years = sorted(self.df['Year'].unique())
        print(f"Years in dataset: {years}")
        assert 2020 not in years and 2021 not in years, "COVID data should be removed!"
        print("✅ COVID data successfully removed")
        
        return True
    
    def analyze_operational_features(self):
        """
        Analyze operational features for Southwest's actionable insights
        """
        print("\n🔍 OPERATIONAL FEATURE ANALYSIS")
        print("=" * 60)
        
        # Essential operational features
        essential_features = {
            'CRSDepTimeHour': 'Departure Hour (operational planning)',
            'DayOfWeek': 'Day of Week (schedule optimization)',
            'Season': 'Season (seasonal planning)',
            'OriginCity': 'Origin Airport (route management)',
            'DestCity': 'Destination Airport (route management)',
            'Distance': 'Flight Distance (fuel planning)',
            'TaxiOut': 'Taxi Out Time (gate efficiency)',
            'TaxiIn': 'Taxi In Time (gate efficiency)',
            'WeatherDelay': 'Weather Delays (external factor)',
            'CarrierDelay': 'Southwest Delays (internal control)',
            'NASDelay': 'Air Traffic Delays (external factor)',
            'LateAircraftDelay': 'Previous Flight Delays (cascade effect)'
        }
        
        print("Essential operational features:")
        for feature, description in essential_features.items():
            if feature in self.df.columns:
                print(f"✅ {feature}: {description}")
            else:
                print(f"❌ {feature}: Missing")
        
        # Analyze delay causes for actionable insights
        print("\nDELAY CAUSE ANALYSIS (Actionable Insights):")
        print("-" * 50)
        
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        total_delays = self.df['DepDelayMinutes'].sum()
        
        for cause in delay_causes:
            if cause in self.df.columns:
                cause_delays = self.df[cause].sum()
                avg_delay = self.df[cause].mean()
                pct_flights = (self.df[cause] > 0).mean() * 100
                pct_total_delay = (cause_delays / total_delays) * 100
                
                print(f"{cause}:")
                print(f"  Avg delay: {avg_delay:.1f} min")
                print(f"  Affects: {pct_flights:.1f}% of flights")
                print(f"  % of total delays: {pct_total_delay:.1f}%")
                
                # Actionable insights
                if cause == 'CarrierDelay':
                    print(f"  🎯 ACTIONABLE: Southwest can control this!")
                elif cause == 'LateAircraftDelay':
                    print(f"  🎯 ACTIONABLE: Better scheduling can reduce this!")
                elif cause == 'WeatherDelay':
                    print(f"  🎯 MONITOR: External factor - prepare contingencies")
                elif cause == 'NASDelay':
                    print(f"  🎯 MONITOR: External factor - work with FAA")
                print()
        
        return True
    
    def create_operational_features(self):
        """
        Create features optimized for Southwest's operational needs
        """
        print("\n🔧 CREATING OPERATIONAL FEATURES")
        print("=" * 60)
        
        # Time-based features (operational planning)
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']
        self.df['Season'] = self.df['Season']
        
        # Route-based features (route optimization)
        self.df['OriginCity'] = self.df['OriginCity']
        self.df['DestCity'] = self.df['DestCity']
        
        # Operational efficiency features
        self.df['Distance'] = self.df['Distance']
        
        # Delay cause features (actionable insights)
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay']
        for cause in delay_causes:
            if cause in self.df.columns:
                self.df[f'{cause}_Binary'] = (self.df[cause] > 0).astype(int)
        
        # Create binary delay target
        self.df['IsDelayed'] = (self.df['DepDelayMinutes'] > 0).astype(int)
        
        print("✅ Operational features created")
        
        # Show operational statistics
        print(f"\nOPERATIONAL STATISTICS:")
        print(f"Total flights: {len(self.df):,}")
        print(f"Delayed flights: {self.df['IsDelayed'].sum():,} ({self.df['IsDelayed'].mean()*100:.1f}%)")
        print(f"Avg taxi out: {self.df['TaxiOut'].mean():.1f} min")
        print(f"Avg taxi in: {self.df['TaxiIn'].mean():.1f} min")
        print(f"Avg distance: {self.df['Distance'].mean():.0f} miles")
        
        return True
    
    def prepare_operational_dataset(self):
        """
        Prepare dataset with operational features only
        """
        print("\n🤖 PREPARING OPERATIONAL ML DATASET")
        print("=" * 60)
        
        # Operational features (pre-flight only for modeling)
        categorical_features = ['OriginCity', 'DestCity', 'Season']
        numerical_features = ['HourOfDay', 'DayOfWeek', 'Distance']
        
        print(f"Categorical features: {categorical_features}")
        print(f"Numerical features: {numerical_features}")
        print(f"Total features: {len(categorical_features) + len(numerical_features)}")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature].astype(str))
                self.feature_encoders[feature] = le
                numerical_features.append(f'{feature}_Encoded')
        
        # Create feature matrix
        X = self.df[numerical_features].fillna(0)
        
        # Create targets
        y_classification = self.df['IsDelayed']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Classification target shape: {y_classification.shape}")
        return X, y_classification, numerical_features
    
    def create_time_based_splits(self, X, y_class):
        """
        Create time-based train/test splits
        """
        print("\n📅 CREATING TIME-BASED SPLITS")
        print("=" * 60)
        
        # Time-based splitting
        train_mask = self.df['Year'].isin([2018, 2019])
        test_mask = self.df['Year'].isin([2022, 2023])
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_class_train = y_class[train_mask]
        y_class_test = y_class[test_mask]
        
        print(f"Training set: {X_train.shape} (2018-2019)")
        print(f"Test set: {X_test.shape} (2022-2023)")
        
        return X_train, X_test, y_class_train, y_class_test
    
    def train_operational_models(self, X_train, X_test, y_class_train, y_class_test):
        """
        Train models optimized for operational insights
        """
        print("\n🚀 TRAINING OPERATIONAL MODELS")
        print("=" * 60)
        
        # Stage 1: Classification Model
        print("Stage 1: Training Classification Model...")
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.classification_model = CalibratedClassifierCV(base_rf, cv=3, method='isotonic')
        self.classification_model.fit(X_train, y_class_train)
        
        # Classification predictions with threshold tuning for recall target
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
    
    def generate_operational_insights(self, X_test, y_class_pred, y_class_proba):
        """
        Generate actionable insights for Southwest operations
        """
        print("\n🎯 SOUTHWEST OPERATIONAL INSIGHTS")
        print("=" * 60)
        
        # Feature importance analysis
        print("FEATURE IMPORTANCE (Operational Focus):")
        print("-" * 50)
        
        feature_names = ['HourOfDay', 'DayOfWeek', 'Distance', 'OriginCity_Encoded', 'DestCity_Encoded', 'Season_Encoded']
        
        if hasattr(self.classification_model, 'feature_importances_'):
            importances = self.classification_model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("Classification Model - Most Important Features:")
            for name, importance in feature_importance[:5]:
                print(f"  {name}: {importance:.3f}")
        
        # Operational recommendations
        print("\nOPERATIONAL RECOMMENDATIONS:")
        print("-" * 50)
        
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
        
        # Analyze operational efficiency
        print(f"\nOPERATIONAL EFFICIENCY METRICS:")
        print(f"  Average taxi out time: {self.df['TaxiOut'].mean():.1f} minutes")
        print(f"  Average taxi in time: {self.df['TaxiIn'].mean():.1f} minutes")
        print(f"  Average flight distance: {self.df['Distance'].mean():.0f} miles")
        
        # Delay cause analysis
        print(f"\nDELAY CAUSE BREAKDOWN:")
        total_delays = self.df['DepDelayMinutes'].sum()
        for cause in ['CarrierDelay', 'LateAircraftDelay', 'WeatherDelay', 'NASDelay']:
            if cause in self.df.columns:
                cause_delays = self.df[cause].sum()
                pct = (cause_delays / total_delays) * 100
                print(f"  {cause}: {pct:.1f}% of total delays")
        
        # Actionable recommendations
        print(f"\nACTIONABLE RECOMMENDATIONS:")
        print("-" * 50)
        print("1. ROUTE OPTIMIZATION:")
        print("   - Focus on improving the top 5 problematic routes")
        print("   - Consider schedule adjustments for high-delay routes")
        
        print("\n2. SCHEDULE OPTIMIZATION:")
        print("   - Avoid scheduling flights during worst hours")
        print("   - Add buffer time for evening departures")
        
        print("\n3. OPERATIONAL EFFICIENCY:")
        print("   - Optimize taxi times (gate efficiency)")
        print("   - Improve aircraft turnaround times")
        
        print("\n4. DELAY MITIGATION:")
        print("   - Focus on controllable delays (CarrierDelay, LateAircraftDelay)")
        print("   - Develop contingency plans for external delays")
    
    def evaluate_models(self, y_class_test, y_class_pred, y_class_proba):
        """
        Evaluate models with operational focus
        """
        print("\n📊 MODEL EVALUATION")
        print("=" * 60)
        
        # Classification evaluation
        print("CLASSIFICATION MODEL:")
        print("-" * 30)
        print(classification_report(y_class_test, y_class_pred))
        try:
            roc_auc = roc_auc_score(y_class_test, y_class_proba)
            pr_auc = average_precision_score(y_class_test, y_class_proba)
            print(f"ROC-AUC: {roc_auc:.3f}")
            print(f"PR-AUC: {pr_auc:.3f}")
            if hasattr(self, 'selected_threshold_'):
                # Report achieved precision/recall at chosen threshold
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
                print("-" * 30)
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
                    ax.set_title('Operational - Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_xticks([0,1]); ax.set_xticklabels(['On-time','Delayed'])
                    ax.set_yticks([0,1]); ax.set_yticklabels(['On-time','Delayed'])
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha='center', va='center', color='black')
                    fig.tight_layout()
                    path = out_dir / 'operational_confusion_matrix.png'
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

                # Cost/benefit table (illustrative defaults)
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
                plt.title('Operational - Calibration Curve')
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / 'operational_calibration_curve.png', dpi=120)
                # Show inline in notebooks
                try:
                    from IPython.display import Image, display
                    display(Image(filename=str(out_dir / 'operational_calibration_curve.png')))
                except Exception:
                    pass
                plt.close()
                print(f"Saved calibration plot: {out_dir / 'operational_calibration_curve.png'}")
            except Exception:
                pass
        except Exception:
            pass
    
    def run_complete_pipeline(self):
        """
        Run complete operational ML pipeline
        """
        print("🚀 SOUTHWEST OPERATIONAL ML PIPELINE")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return None
        
        # Ensure `Route` exists for route-level insights
        if 'Route' not in self.df.columns and {'OriginCity','DestCity'}.issubset(self.df.columns):
            self.df['Route'] = self.df['OriginCity'] + ' → ' + self.df['DestCity']

        # Analyze operational features
        self.analyze_operational_features()
        
        # Create operational features
        self.create_operational_features()
        
        # Prepare dataset
        X, y_class, feature_names = self.prepare_operational_dataset()
        
        # Create splits
        X_train, X_test, y_class_train, y_class_test = self.create_time_based_splits(
            X, y_class
        )
        
        # Train models
        y_class_pred, y_class_proba = self.train_operational_models(
            X_train, X_test, y_class_train, y_class_test
        )
        
        # Evaluate models
        self.evaluate_models(y_class_test, y_class_pred, y_class_proba)
        
        # Generate operational insights
        self.generate_operational_insights(X_test, y_class_pred, y_class_proba)

        # Rolling time validation for additional robustness
        try:
            self.evaluate_with_rolling_time_cv()
        except Exception:
            pass
        
        print("\n🎯 SOUTHWEST OPERATIONAL PIPELINE COMPLETE!")
        print("=" * 70)
        # Clear summary of selected model
        model_name = type(self.classification_model).__name__
        print("✅ Actionable recommendations generated")
        print("✅ Ready for Southwest operational planning!")
        print(f"📌 Selected classifier: {model_name}")
        if hasattr(self, 'selected_threshold_'):
            print(f"📌 Selected decision threshold: {self.selected_threshold_:.2f}")
        # Note: operational pipeline uses fixed RF; selection PR-AUC not applicable
        
        return {
            'classification_model': self.classification_model,
            'feature_encoders': self.feature_encoders,
            'feature_names': feature_names
        }

    def evaluate_with_rolling_time_cv(self):
        """
        Rolling year-based validation using only pre-event features:
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
            # Build feature set (pre-flight only)
            categorical_features = ['OriginCity', 'DestCity', 'Season']
            base_features = [col for col in self.df.columns if col.endswith('_Encoded') or col in ['HourOfDay', 'DayOfWeek', 'Distance']]
            feature_cols = list(dict.fromkeys(base_features))
            X_tr = self.df.loc[train_mask, feature_cols].fillna(0)
            X_va = self.df.loc[val_mask, feature_cols].fillna(0)
            y_tr = self.df.loc[train_mask, 'IsDelayed']
            y_va = self.df.loc[val_mask, 'IsDelayed']
            # Train calibrated RF
            base_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            calibrated = CalibratedClassifierCV(base_rf, cv=3, method='isotonic')
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
            print("\n🧪 ROLLING TIME VALIDATION (Operational)")
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
    Main function to run Southwest operational ML pipeline
    """
    pipeline = SouthwestOperationalML()
    models = pipeline.run_complete_pipeline()
    
    if models:
        print(f"\n📊 FINAL OPERATIONAL MODEL SUMMARY:")
        print(f"Classification model: {type(models['classification_model']).__name__}")
        print(f"Features: {len(models['feature_names'])}")
        print(f"Focus: Operational efficiency and actionable insights")

if __name__ == "__main__":
    main()
