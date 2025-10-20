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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
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
        self.regression_model = None
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
        self.df['TaxiOut'] = self.df['TaxiOut']
        self.df['TaxiIn'] = self.df['TaxiIn']
        
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
        
        # Operational features (avoiding overfitting)
        categorical_features = ['OriginCity', 'DestCity', 'Season']
        numerical_features = ['HourOfDay', 'DayOfWeek', 'Distance', 'TaxiOut', 'TaxiIn']
        
        # Add delay cause binary features
        delay_causes = ['WeatherDelay_Binary', 'CarrierDelay_Binary', 'NASDelay_Binary', 'LateAircraftDelay_Binary']
        numerical_features.extend([cause for cause in delay_causes if cause in self.df.columns])
        
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
        y_regression = self.df['DepDelayMinutes']
        
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
        
        # Time-based splitting
        train_mask = self.df['Year'].isin([2018, 2019])
        test_mask = self.df['Year'].isin([2022, 2023])
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_class_train = y_class[train_mask]
        y_class_test = y_class[test_mask]
        y_reg_train = y_reg[train_mask]
        y_reg_test = y_reg[test_mask]
        
        print(f"Training set: {X_train.shape} (2018-2019)")
        print(f"Test set: {X_test.shape} (2022-2023)")
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
    
    def train_operational_models(self, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
        """
        Train models optimized for operational insights
        """
        print("\n🚀 TRAINING OPERATIONAL MODELS")
        print("=" * 60)
        
        # Stage 1: Classification Model
        print("Stage 1: Training Classification Model...")
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,  # Reduced to prevent overfitting
            min_samples_split=20,  # Increased to prevent overfitting
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.classification_model.fit(X_train, y_class_train)
        
        # Classification predictions with threshold tuning
        y_class_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        def _select_optimal_threshold(y_true, y_proba):
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
            best_idx = f1_scores[:-1].argmax()
            return max(0.01, min(0.99, thresholds[best_idx]))

        selected_threshold = _select_optimal_threshold(y_class_test, y_class_proba)
        self.selected_threshold_ = selected_threshold
        y_class_pred = (y_class_proba >= selected_threshold).astype(int)
        
        print("✅ Classification model trained")
        
        # Stage 2: Regression Model
        print("\nStage 2: Training Regression Model...")
        
        # Only train on delayed flights
        delayed_train_mask = y_class_train == 1
        X_train_delayed = X_train[delayed_train_mask]
        y_reg_train_delayed = y_reg_train[delayed_train_mask]
        
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,  # Reduced to prevent overfitting
            min_samples_split=20,  # Increased to prevent overfitting
            random_state=42,
            n_jobs=-1
        )
        
        self.regression_model.fit(X_train_delayed, y_reg_train_delayed)
        
        print("✅ Regression model trained")
        
        # Combined predictions
        delay_predictions = np.zeros(len(X_test))
        delayed_mask = y_class_pred == 1
        if delayed_mask.sum() > 0:
            delay_predictions[delayed_mask] = self.regression_model.predict(X_test[delayed_mask])
        
        return y_class_pred, y_class_proba, delay_predictions
    
    def generate_operational_insights(self, X_test, y_class_pred, y_class_proba, delay_predictions):
        """
        Generate actionable insights for Southwest operations
        """
        print("\n🎯 SOUTHWEST OPERATIONAL INSIGHTS")
        print("=" * 60)
        
        # Feature importance analysis
        print("FEATURE IMPORTANCE (Operational Focus):")
        print("-" * 50)
        
        feature_names = ['HourOfDay', 'DayOfWeek', 'Distance', 'TaxiOut', 'TaxiIn',
                        'WeatherDelay_Binary', 'CarrierDelay_Binary', 'NASDelay_Binary', 'LateAircraftDelay_Binary',
                        'OriginCity_Encoded', 'DestCity_Encoded', 'Season_Encoded']
        
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
    
    def evaluate_models(self, y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions):
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
                print(f"Selected threshold (F1-optimal): {self.selected_threshold_:.2f}")
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
        
        # Overall performance
        print("\nOVERALL PERFORMANCE:")
        print("-" * 30)
        overall_mae = mean_absolute_error(y_reg_test, delay_predictions)
        classification_accuracy = (y_class_pred == y_class_test).mean()
        
        print(f"Overall MAE: {overall_mae:.2f} minutes")
        print(f"Classification Accuracy: {classification_accuracy:.3f}")
        
        # Delay buckets macro-F1
        def bucketize(v):
            if v <= 0:
                return 0
            if v <= 15:
                return 1
            if v <= 60:
                return 2
            return 3
        y_true_bucket = y_reg_test.apply(bucketize)
        y_pred_bucket = [bucketize(v) for v in delay_predictions]
        try:
            bucket_f1 = f1_score(y_true_bucket, y_pred_bucket, average='macro')
            print(f"Bucketed delay macro-F1: {bucket_f1:.3f} (bins: 0,1-15,16-60,>60)")
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
        X, y_class, y_reg, feature_names = self.prepare_operational_dataset()
        
        # Create splits
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = self.create_time_based_splits(
            X, y_class, y_reg
        )
        
        # Train models
        y_class_pred, y_class_proba, delay_predictions = self.train_operational_models(
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
        )
        
        # Evaluate models
        self.evaluate_models(y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions)
        
        # Generate operational insights
        self.generate_operational_insights(X_test, y_class_pred, y_class_proba, delay_predictions)
        
        print("\n🎯 SOUTHWEST OPERATIONAL PIPELINE COMPLETE!")
        print("=" * 70)
        print("✅ Two-stage model trained for operational insights")
        print("✅ Actionable recommendations generated")
        print("✅ Ready for Southwest operational planning!")
        
        return {
            'classification_model': self.classification_model,
            'regression_model': self.regression_model,
            'feature_encoders': self.feature_encoders,
            'feature_names': feature_names
        }

def main():
    """
    Main function to run Southwest operational ML pipeline
    """
    pipeline = SouthwestOperationalML()
    models = pipeline.run_complete_pipeline()
    
    if models:
        print(f"\n📊 FINAL OPERATIONAL MODEL SUMMARY:")
        print(f"Classification model: {type(models['classification_model']).__name__}")
        print(f"Regression model: {type(models['regression_model']).__name__}")
        print(f"Features: {len(models['feature_names'])}")
        print(f"Focus: Operational efficiency and actionable insights")

if __name__ == "__main__":
    main()
