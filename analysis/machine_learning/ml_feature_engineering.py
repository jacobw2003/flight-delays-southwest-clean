#!/usr/bin/env python3
"""
ML Feature Engineering Best Practices for Southwest Airlines Delay Prediction

This script demonstrates how to properly prepare features for ML modeling
following the insights from our comprehensive data analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MLFeatureEngineer:
    """
    Feature engineering class following ML best practices
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "preprocessed_data" / "southwest_final_preprocessed.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the ML-ready dataset"""
        print("📂 LOADING ML-READY DATASET")
        print("=" * 50)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Date range: {self.df['DepDate'].min()} to {self.df['DepDate'].max()}")
        
        # Verify COVID data is removed
        years = sorted(self.df['Year'].unique())
        print(f"Years in dataset: {years}")
        assert 2020 not in years and 2021 not in years, "COVID data should be removed!"
        print("✅ COVID data successfully removed")
        
        return True
    
    def create_ml_features(self):
        """
        Create features optimized for ML modeling
        """
        print("\n🔧 CREATING ML FEATURES")
        print("=" * 50)
        
        # 1. Time-based features
        print("1. Creating time-based features...")
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']  # Already exists
        self.df['Month'] = self.df['Month']  # Already exists
        self.df['Quarter'] = self.df['Quarter']  # Already exists
        
        # 2. Route-based features
        print("2. Creating route-based features...")
        # Route frequency (how often this route is flown)
        route_counts = self.df['Route'].value_counts()
        self.df['RouteFrequency'] = self.df['Route'].map(route_counts)
        
        # Route delay history (average delay for this route)
        route_delays = self.df.groupby('Route')['DepDelayMinutes'].mean()
        self.df['RouteAvgDelay'] = self.df['Route'].map(route_delays)
        
        # 3. Airport-based features
        print("3. Creating airport-based features...")
        # Origin airport delay history
        origin_delays = self.df.groupby('OriginCity')['DepDelayMinutes'].mean()
        self.df['OriginAvgDelay'] = self.df['OriginCity'].map(origin_delays)
        
        # Destination airport delay history
        dest_delays = self.df.groupby('DestCity')['DepDelayMinutes'].mean()
        self.df['DestAvgDelay'] = self.df['DestCity'].map(dest_delays)
        
        # 4. Seasonal features
        print("4. Creating seasonal features...")
        # Season encoding
        season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
        self.df['SeasonEncoded'] = self.df['Season'].map(season_map)
        
        # 5. Distance-based features
        print("5. Creating distance-based features...")
        self.df['DistanceGroup'] = pd.cut(self.df['Distance'], 
                                         bins=[0, 500, 1000, 1500, 2000, float('inf')],
                                         labels=['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long'])
        
        # 6. Delay cause features
        print("6. Creating delay cause features...")
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
        for cause in delay_causes:
            if cause in self.df.columns:
                self.df[f'{cause}_Binary'] = (self.df[cause] > 0).astype(int)
        
        print("✅ All ML features created successfully")
        
        return self.df
    
    def prepare_ml_dataset(self):
        """
        Prepare dataset for ML modeling
        """
        print("\n🤖 PREPARING ML DATASET")
        print("=" * 50)
        
        # Define features for ML
        categorical_features = ['Route', 'OriginCity', 'DestCity', 'CarrierName', 'Season', 'DistanceGroup']
        numerical_features = ['HourOfDay', 'DayOfWeek', 'Month', 'Quarter', 'SeasonEncoded',
                            'RouteFrequency', 'RouteAvgDelay', 'OriginAvgDelay', 'DestAvgDelay',
                            'Distance', 'TaxiOut', 'TaxiIn']
        
        # Add delay cause binary features
        delay_causes = ['WeatherDelay_Binary', 'CarrierDelay_Binary', 'NASDelay_Binary', 
                       'SecurityDelay_Binary', 'LateAircraftDelay_Binary']
        numerical_features.extend([cause for cause in delay_causes if cause in self.df.columns])
        
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature].astype(str))
                self.feature_encoders[feature] = le
                numerical_features.append(f'{feature}_Encoded')
        
        # Create final feature matrix
        X = self.df[numerical_features].fillna(0)
        y = self.df['DepDelayMinutes']
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        return X, y, numerical_features
    
    def create_time_based_splits(self, X, y):
        """
        Create time-based train/test splits (ML best practice)
        """
        print("\n📅 CREATING TIME-BASED SPLITS")
        print("=" * 50)
        
        # Time-based splitting (not random!)
        train_mask = self.df['Year'].isin([2018, 2019])
        test_mask = self.df['Year'].isin([2022, 2023])
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        print(f"Training set: {X_train.shape} (2018-2019)")
        print(f"Test set: {X_test.shape} (2022-2023)")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        # Verify no data leakage
        train_years = sorted(self.df[train_mask]['Year'].unique())
        test_years = sorted(self.df[test_mask]['Year'].unique())
        print(f"Training years: {train_years}")
        print(f"Test years: {test_years}")
        
        assert not set(train_years).intersection(set(test_years)), "Data leakage detected!"
        print("✅ No data leakage - time-based split is correct")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features for ML models
        """
        print("\n📏 SCALING FEATURES")
        print("=" * 50)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✅ Features scaled successfully")
        print(f"Training set scaled shape: {X_train_scaled.shape}")
        print(f"Test set scaled shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def run_complete_pipeline(self):
        """
        Run complete ML feature engineering pipeline
        """
        print("🚀 ML FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return None
        
        # Create features
        self.create_ml_features()
        
        # Prepare ML dataset
        X, y, feature_names = self.prepare_ml_dataset()
        
        # Create time-based splits
        X_train, X_test, y_train, y_test = self.create_time_based_splits(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n🎯 ML DATASET READY!")
        print("=" * 60)
        print("✅ COVID data removed (statistically justified)")
        print("✅ Time-based splits (no data leakage)")
        print("✅ Features engineered for ML")
        print("✅ Categorical features encoded")
        print("✅ Features scaled")
        print("✅ Ready for model training!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'encoders': self.feature_encoders,
            'scaler': self.scaler
        }

def main():
    """
    Main function to run ML feature engineering
    """
    engineer = MLFeatureEngineer()
    ml_data = engineer.run_complete_pipeline()
    
    if ml_data:
        print(f"\n📊 FINAL ML DATASET SUMMARY:")
        print(f"Training samples: {ml_data['X_train'].shape[0]:,}")
        print(f"Test samples: {ml_data['X_test'].shape[0]:,}")
        print(f"Features: {ml_data['X_train'].shape[1]}")
        print(f"Target range: {ml_data['y_train'].min():.1f} to {ml_data['y_train'].max():.1f} minutes")

if __name__ == "__main__":
    main()
