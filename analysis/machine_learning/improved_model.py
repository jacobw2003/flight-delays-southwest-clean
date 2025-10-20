#!/usr/bin/env python3
"""
Improved Southwest Airlines Model with Better R² Score

This script addresses the low R² score by:
1. Adding more predictive features
2. Using better model architecture
3. Implementing proper feature engineering
4. Using ensemble methods
5. Focusing on what's actually predictable
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ImprovedSouthwestModel:
    """
    Improved model with better R² score
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "preprocessed_data" / "southwest_final_preprocessed.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load data"""
        print("📂 LOADING DATA FOR IMPROVED MODEL")
        print("=" * 60)
        
        # Load sample for testing
        self.df = pd.read_csv(self.data_path, nrows=200000)
        print(f"Dataset loaded: {self.df.shape}")
        
        return True
    
    def create_enhanced_features(self):
        """
        Create enhanced features for better prediction
        """
        print("\n🔧 CREATING ENHANCED FEATURES")
        print("=" * 60)
        
        # Basic features
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']
        self.df['Season'] = self.df['Season']
        
        # Enhanced time features
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)
        self.df['IsEvening'] = (self.df['HourOfDay'] >= 18).astype(int)
        self.df['IsMorning'] = (self.df['HourOfDay'] <= 10).astype(int)
        self.df['IsPeakHour'] = ((self.df['HourOfDay'] >= 7) & (self.df['HourOfDay'] <= 9) | 
                                (self.df['HourOfDay'] >= 17) & (self.df['HourOfDay'] <= 19)).astype(int)
        
        # Route-based features
        self.df['OriginCity'] = self.df['OriginCity']
        self.df['DestCity'] = self.df['DestCity']
        
        # Create route delay history (this is key!)
        route_delays = self.df.groupby('Route')['DepDelayMinutes'].agg(['mean', 'std', 'count']).round(2)
        route_delays.columns = ['RouteAvgDelay', 'RouteDelayStd', 'RouteCount']
        
        self.df['RouteAvgDelay'] = self.df['Route'].map(route_delays['RouteAvgDelay'])
        self.df['RouteDelayStd'] = self.df['Route'].map(route_delays['RouteDelayStd'])
        self.df['RouteCount'] = self.df['Route'].map(route_delays['RouteCount'])
        
        # Airport-based features
        origin_delays = self.df.groupby('OriginCity')['DepDelayMinutes'].agg(['mean', 'std']).round(2)
        origin_delays.columns = ['OriginAvgDelay', 'OriginDelayStd']
        
        self.df['OriginAvgDelay'] = self.df['OriginCity'].map(origin_delays['OriginAvgDelay'])
        self.df['OriginDelayStd'] = self.df['OriginCity'].map(origin_delays['OriginDelayStd'])
        
        dest_delays = self.df.groupby('DestCity')['DepDelayMinutes'].agg(['mean', 'std']).round(2)
        dest_delays.columns = ['DestAvgDelay', 'DestDelayStd']
        
        self.df['DestAvgDelay'] = self.df['DestCity'].map(dest_delays['DestAvgDelay'])
        self.df['DestDelayStd'] = self.df['DestCity'].map(dest_delays['DestDelayStd'])
        
        # Operational features
        self.df['Distance'] = self.df['Distance']
        self.df['TaxiOut'] = self.df['TaxiOut']
        self.df['TaxiIn'] = self.df['TaxiIn']
        
        # Delay cause features
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay']
        for cause in delay_causes:
            if cause in self.df.columns:
                self.df[f'{cause}_Binary'] = (self.df[cause] > 0).astype(int)
                self.df[f'{cause}_Minutes'] = self.df[cause]
        
        # Historical features (rolling averages)
        self.df['DistanceGroup'] = pd.cut(self.df['Distance'], 
                                         bins=[0, 500, 1000, 1500, 2000, float('inf')],
                                         labels=[1, 2, 3, 4, 5])
        
        # Fill missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        print("✅ Enhanced features created")
        print(f"Total features: {len(numeric_cols)}")
        
        return True
    
    def prepare_enhanced_dataset(self):
        """
        Prepare enhanced dataset
        """
        print("\n🤖 PREPARING ENHANCED DATASET")
        print("=" * 60)
        
        # Select features
        feature_cols = [
            'HourOfDay', 'DayOfWeek', 'IsWeekend', 'IsEvening', 'IsMorning', 'IsPeakHour',
            'Distance', 'TaxiOut', 'TaxiIn', 'DistanceGroup',
            'RouteAvgDelay', 'RouteDelayStd', 'RouteCount',
            'OriginAvgDelay', 'OriginDelayStd', 'DestAvgDelay', 'DestDelayStd',
            'WeatherDelay_Binary', 'CarrierDelay_Binary', 'NASDelay_Binary', 'LateAircraftDelay_Binary',
            'WeatherDelay_Minutes', 'CarrierDelay_Minutes', 'NASDelay_Minutes', 'LateAircraftDelay_Minutes'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in self.df.columns]
        print(f"Using {len(available_features)} features:")
        for feature in available_features:
            print(f"  ✅ {feature}")
        
        X = self.df[available_features]
        y = self.df['DepDelayMinutes']
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Target statistics:")
        print(f"  Mean: {y.mean():.2f} minutes")
        print(f"  Std: {y.std():.2f} minutes")
        print(f"  Min: {y.min():.1f} minutes")
        print(f"  Max: {y.max():.1f} minutes")
        
        return X, y, available_features
    
    def test_improved_models(self, X, y):
        """
        Test improved models
        """
        print("\n🚀 TESTING IMPROVED MODELS")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test different models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Ensemble': VotingRegressor([
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('ridge', Ridge(alpha=1.0))
            ])
        }
        
        results = {}
        
        print("MODEL COMPARISON:")
        print("-" * 50)
        
        for name, model in models.items():
            # Train model
            if name in ['Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {'r2': r2, 'mae': mae}
            
            print(f"{name}:")
            print(f"  R²: {r2:.3f}")
            print(f"  MAE: {mae:.2f} minutes")
            print()
        
        # Test on delayed flights only
        print("DELAYED FLIGHTS ONLY:")
        print("-" * 50)
        
        delayed_mask = y_test > 0
        if delayed_mask.sum() > 0:
            y_test_delayed = y_test[delayed_mask]
            
            for name, model in models.items():
                if name in ['Ridge Regression', 'Lasso Regression']:
                    y_pred_delayed = model.predict(X_test_scaled[delayed_mask])
                else:
                    y_pred_delayed = model.predict(X_test[delayed_mask])
                
                r2_delayed = r2_score(y_test_delayed, y_pred_delayed)
                mae_delayed = mean_absolute_error(y_test_delayed, y_pred_delayed)
                
                print(f"{name}:")
                print(f"  R²: {r2_delayed:.3f}")
                print(f"  MAE: {mae_delayed:.2f} minutes")
                print()
        
        return results
    
    def analyze_prediction_difficulty(self, X, y):
        """
        Analyze why prediction is difficult
        """
        print("\n🔍 ANALYZING PREDICTION DIFFICULTY")
        print("=" * 60)
        
        # Target variable analysis
        print("TARGET VARIABLE ANALYSIS:")
        print("-" * 30)
        print(f"On-time flights (0 min): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
        print(f"Minor delays (1-15 min): {((y > 0) & (y <= 15)).sum():,} ({((y > 0) & (y <= 15)).mean()*100:.1f}%)")
        print(f"Moderate delays (16-60 min): {((y > 15) & (y <= 60)).sum():,} ({((y > 15) & (y <= 60)).mean()*100:.1f}%)")
        print(f"Major delays (>60 min): {(y > 60).sum():,} ({(y > 60).mean()*100:.1f}%)")
        
        # Feature correlation analysis
        print(f"\nFEATURE CORRELATION ANALYSIS:")
        print("-" * 30)
        
        # Calculate correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        print("Top 10 features correlated with delays:")
        for feature, corr in correlations.head(10).items():
            print(f"  {feature}: {corr:.3f}")
        
        # Why prediction is difficult
        print(f"\nWHY PREDICTION IS DIFFICULT:")
        print("-" * 30)
        print("1. HIGH VARIANCE: Flight delays have high variability")
        print("2. EXTERNAL FACTORS: Weather, air traffic, mechanical issues")
        print("3. CASCADE EFFECTS: One delay causes others")
        print("4. RANDOM EVENTS: Unpredictable circumstances")
        print("5. LIMITED FEATURES: Missing weather, traffic, maintenance data")
        
        # What's actually predictable
        print(f"\nWHAT'S ACTUALLY PREDICTABLE:")
        print("-" * 30)
        print("✅ Route-specific delay patterns")
        print("✅ Time-based delay trends")
        print("✅ Airport-specific delays")
        print("✅ Operational efficiency metrics")
        print("❌ Exact delay duration")
        print("❌ Random external events")
    
    def run_improved_analysis(self):
        """
        Run improved model analysis
        """
        print("🚀 IMPROVED SOUTHWEST MODEL ANALYSIS")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create enhanced features
        self.create_enhanced_features()
        
        # Prepare dataset
        X, y, features = self.prepare_enhanced_dataset()
        
        # Test improved models
        results = self.test_improved_models(X, y)
        
        # Analyze prediction difficulty
        self.analyze_prediction_difficulty(X, y)
        
        print("\n🎯 IMPROVED ANALYSIS COMPLETE!")
        print("=" * 70)
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"Best model: {best_model[0]}")
        print(f"Best R²: {best_model[1]['r2']:.3f}")
        print(f"Best MAE: {best_model[1]['mae']:.2f} minutes")
        
        return results

def main():
    """Main function"""
    analyzer = ImprovedSouthwestModel()
    results = analyzer.run_improved_analysis()

if __name__ == "__main__":
    main()
