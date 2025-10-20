#!/usr/bin/env python3
"""
Data Preprocessing Script for Southwest Airlines Flight Delay Analysis

This script performs comprehensive data cleaning and preprocessing:
1. Analyzes data quality issues (nulls, duplicates, outliers)
2. Removes redundant columns
3. Converts data types appropriately
4. Creates new features for better analysis
5. Saves cleaned data for ML modeling

Problem Statement Focus:
- Which routes experience the most frequent and severe delays?
- What are the primary causes of delays for Southwest Airlines?
- How can Southwest Airlines predict future delays?
- Are there seasonal patterns?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality(df, dataset_name):
    """
    Comprehensive data quality analysis
    """
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ANALYSIS: {dataset_name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    print(f"\n📊 MISSING VALUES ANALYSIS:")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percentage': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("✅ No missing values found!")
    
    # Duplicate analysis
    print(f"\n🔄 DUPLICATE ANALYSIS:")
    print("-" * 40)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # Data types analysis
    print(f"\n📋 DATA TYPES ANALYSIS:")
    print("-" * 40)
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n🔢 NUMERIC COLUMNS ANALYSIS:")
    print("-" * 40)
    print(f"Number of numeric columns: {len(numeric_cols)}")
    
    # Check for potential outliers in delay columns
    delay_cols = [col for col in numeric_cols if 'Delay' in col or 'delay' in col]
    if delay_cols:
        print(f"\nDelay columns found: {delay_cols}")
        for col in delay_cols:
            if col in df.columns:
                print(f"\n{col} statistics:")
                print(f"  Min: {df[col].min()}")
                print(f"  Max: {df[col].max()}")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Median: {df[col].median():.2f}")
                print(f"  Std: {df[col].std():.2f}")
                
                # Check for extreme outliers
                q99 = df[col].quantile(0.99)
                extreme_outliers = (df[col] > q99).sum()
                print(f"  Extreme outliers (>99th percentile): {extreme_outliers:,}")
    
    return missing_df

def preprocess_flight_data():
    """
    Main preprocessing function
    """
    print("🚀 STARTING SOUTHWEST AIRLINES DATA PREPROCESSING")
    print("=" * 60)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_file = project_root / "data" / "csv_data" / "features_added_southwest.csv"
    output_dir = project_root / "data" / "preprocessed_data"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Initial data quality analysis
    missing_df = analyze_data_quality(df, "ORIGINAL DATASET")
    
    print(f"\n🔧 STARTING PREPROCESSING STEPS...")
    print("=" * 60)
    
    # Step 1: Remove redundant columns
    print(f"\n1️⃣ REMOVING REDUNDANT COLUMNS")
    print("-" * 40)
    redundant_cols = ['Year', 'Month', 'DayofMonth']
    cols_to_remove = [col for col in redundant_cols if col in df.columns]
    
    if cols_to_remove:
        print(f"Removing columns: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)
        print(f"✅ Removed {len(cols_to_remove)} redundant columns")
    else:
        print("✅ No redundant columns found to remove")
    
    # Step 2: Convert FlightDate to DateTime
    print(f"\n2️⃣ CONVERTING FLIGHT DATE")
    print("-" * 40)
    if 'FlightDate' in df.columns:
        print(f"Original FlightDate dtype: {df['FlightDate'].dtype}")
        print(f"Sample FlightDate values: {df['FlightDate'].head(3).tolist()}")
        
        df['DepDate'] = pd.to_datetime(df['FlightDate'])
        df = df.drop(columns=['FlightDate'])
        print(f"✅ Converted FlightDate to DepDate (DateTime)")
        print(f"New DepDate dtype: {df['DepDate'].dtype}")
        print(f"Date range: {df['DepDate'].min()} to {df['DepDate'].max()}")
    else:
        print("❌ FlightDate column not found!")
    
    # Step 3: Split Origin and Destination columns
    print(f"\n3️⃣ SPLITTING ORIGIN AND DESTINATION")
    print("-" * 40)
    
    # Split OriginCityName
    if 'OriginCityName' in df.columns:
        print("Splitting OriginCityName...")
        df[['OriginCity', 'OriginState']] = df['OriginCityName'].str.split(', ', expand=True)
        df = df.drop(columns=['OriginCityName'])
        print(f"✅ Created OriginCity and OriginState columns")
        print(f"Unique origin cities: {df['OriginCity'].nunique()}")
        print(f"Unique origin states: {df['OriginState'].nunique()}")
    
    # Split DestCityName
    if 'DestCityName' in df.columns:
        print("Splitting DestCityName...")
        df[['DestCity', 'DestState']] = df['DestCityName'].str.split(', ', expand=True)
        df = df.drop(columns=['DestCityName'])
        print(f"✅ Created DestCity and DestState columns")
        print(f"Unique destination cities: {df['DestCity'].nunique()}")
        print(f"Unique destination states: {df['DestState'].nunique()}")
    
    # Step 4: Map Marketing_Airline_Network to Carrier Name
    print(f"\n4️⃣ MAPPING AIRLINE CODES")
    print("-" * 40)
    
    # Airline code mapping
    airline_mapping = {
        'WN': 'Southwest Airlines',
        'AA': 'American Airlines',
        'DL': 'Delta Air Lines',
        'UA': 'United Airlines',
        'B6': 'JetBlue Airways',
        'NK': 'Spirit Airlines',
        'F9': 'Frontier Airlines',
        'AS': 'Alaska Airlines',
        'HA': 'Hawaiian Airlines',
        'VX': 'Virgin America',
        'G4': 'Allegiant Air'
    }
    
    if 'Marketing_Airline_Network' in df.columns:
        print(f"Unique airline codes: {df['Marketing_Airline_Network'].unique()}")
        df['CarrierName'] = df['Marketing_Airline_Network'].map(airline_mapping)
        
        # Fill any unmapped values
        unmapped = df['CarrierName'].isnull().sum()
        if unmapped > 0:
            print(f"⚠️  {unmapped} unmapped airline codes found")
            df['CarrierName'] = df['CarrierName'].fillna('Unknown Airline')
        
        print(f"✅ Created CarrierName column")
        print(f"Carrier distribution:")
        print(df['CarrierName'].value_counts())
    
    # Step 5: Additional data cleaning
    print(f"\n5️⃣ ADDITIONAL DATA CLEANING")
    print("-" * 40)
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"✅ Removed {duplicates_removed:,} duplicate rows")
    
    # Handle any remaining missing values in critical columns
    critical_cols = ['DepDelayMinutes', 'ArrDelayMinutes', 'DepDate']
    for col in critical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            missing_count = df[col].isnull().sum()
            print(f"⚠️  {missing_count:,} missing values in {col}")
            
            if col in ['DepDelayMinutes', 'ArrDelayMinutes']:
                # Fill delay minutes with 0 (no delay)
                df[col] = df[col].fillna(0)
                print(f"✅ Filled {col} missing values with 0")
    
    # Step 6: Create additional useful features
    print(f"\n6️⃣ CREATING ADDITIONAL FEATURES")
    print("-" * 40)
    
    if 'DepDate' in df.columns:
        # Extract date components
        df['Year'] = df['DepDate'].dt.year
        df['Month'] = df['DepDate'].dt.month
        df['Day'] = df['DepDate'].dt.day
        df['DayOfWeek'] = df['DepDate'].dt.dayofweek
        df['Quarter'] = df['DepDate'].dt.quarter
        
        # Create season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['Season'] = df['Month'].apply(get_season)
        
        print(f"✅ Created additional date features: Year, Month, Day, DayOfWeek, Quarter, Season")
        print(f"Season distribution:")
        print(df['Season'].value_counts())
    
    # Create route feature
    if 'OriginCity' in df.columns and 'DestCity' in df.columns:
        df['Route'] = df['OriginCity'] + ' → ' + df['DestCity']
        print(f"✅ Created Route feature")
        print(f"Unique routes: {df['Route'].nunique()}")
    
    # Create delay severity categories
    if 'DepDelayMinutes' in df.columns:
        def categorize_delay(delay_minutes):
            if pd.isna(delay_minutes) or delay_minutes <= 0:
                return 'On Time'
            elif delay_minutes <= 15:
                return 'Minor Delay'
            elif delay_minutes <= 60:
                return 'Moderate Delay'
            else:
                return 'Major Delay'
        
        df['DelaySeverity'] = df['DepDelayMinutes'].apply(categorize_delay)
        print(f"✅ Created DelaySeverity feature")
        print(f"Delay severity distribution:")
        print(df['DelaySeverity'].value_counts())
    
    # Final data quality check
    print(f"\n7️⃣ FINAL DATA QUALITY CHECK")
    print("-" * 40)
    analyze_data_quality(df, "PREPROCESSED DATASET")
    
    # Save preprocessed data
    output_file = output_dir / "southwest_preprocessed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 SAVING PREPROCESSED DATA")
    print("-" * 40)
    print(f"✅ Saved preprocessed data to: {output_file}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Final dataset size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Summary statistics
    print(f"\n📈 PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Removed redundant columns: {len(cols_to_remove) if cols_to_remove else 0}")
    print(f"✅ Converted FlightDate to DepDate (DateTime)")
    print(f"✅ Split city/state columns: OriginCity, OriginState, DestCity, DestState")
    print(f"✅ Mapped airline codes to carrier names")
    print(f"✅ Removed duplicates: {duplicates_removed:,}")
    print(f"✅ Created additional features: Season, Route, DelaySeverity, date components")
    print(f"✅ Final dataset ready for ML modeling!")
    
    return df

if __name__ == "__main__":
    preprocessed_df = preprocess_flight_data()
