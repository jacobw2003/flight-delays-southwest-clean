#!/usr/bin/env python3
"""
Complete Southwest Airlines Data Preprocessing Pipeline

This script performs the entire preprocessing pipeline:
1. Load features_added_southwest.csv
2. Analyze data quality
3. Remove redundant columns
4. Convert data types and create new features
5. Filter out COVID data (2020-2021)
6. Save final preprocessed dataset

Result: A clean, ML-ready dataset optimized for flight delay prediction.
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
    
    return missing_df

def remove_redundant_columns(df):
    """
    Remove redundant columns that provide the same information
    """
    print("\n🔧 REMOVING REDUNDANT COLUMNS")
    print("-" * 50)
    
    original_cols = len(df.columns)
    columns_to_remove = []
    
    # 1. Remove DepDelay (keep DepDelayMinutes - non-negative values)
    if 'DepDelay' in df.columns and 'DepDelayMinutes' in df.columns:
        columns_to_remove.append('DepDelay')
        print("✅ Removing DepDelay (keeping DepDelayMinutes)")
    
    # 2. Remove ArrDelay (keep ArrDelayMinutes - non-negative values)
    if 'ArrDelay' in df.columns and 'ArrDelayMinutes' in df.columns:
        columns_to_remove.append('ArrDelay')
        print("✅ Removing ArrDelay (keeping ArrDelayMinutes)")
    
    # 3. Remove DayofWeek (keep DayOfWeek - better naming)
    if 'DayofWeek' in df.columns and 'DayOfWeek' in df.columns:
        columns_to_remove.append('DayofWeek')
        print("✅ Removing DayofWeek (keeping DayOfWeek)")
    
    # 4. Remove Marketing_Airline_Network (keep CarrierName - more descriptive)
    if 'Marketing_Airline_Network' in df.columns and 'CarrierName' in df.columns:
        columns_to_remove.append('Marketing_Airline_Network')
        print("✅ Removing Marketing_Airline_Network (keeping CarrierName)")
    
    # 5. Remove redundant time columns - keep the most useful ones
    time_cols_to_remove = []
    
    # Keep CRSDepTimeHour/Minute, remove CRSDepTimeHourDis (categorical version)
    if 'CRSDepTimeHour' in df.columns and 'CRSDepTimeHourDis' in df.columns:
        time_cols_to_remove.append('CRSDepTimeHourDis')
        print("✅ Removing CRSDepTimeHourDis (keeping CRSDepTimeHour)")
    
    # Keep WheelsOffHour/Minute, remove WheelsOffHourDis
    if 'WheelsOffHour' in df.columns and 'WheelsOffHourDis' in df.columns:
        time_cols_to_remove.append('WheelsOffHourDis')
        print("✅ Removing WheelsOffHourDis (keeping WheelsOffHour)")
    
    # Keep CRSArrTimeHour/Minute, remove CRSArrTimeHourDis
    if 'CRSArrTimeHour' in df.columns and 'CRSArrTimeHourDis' in df.columns:
        time_cols_to_remove.append('CRSArrTimeHourDis')
        print("✅ Removing CRSArrTimeHourDis (keeping CRSArrTimeHour)")
    
    # Keep WheelsOnHour/Minute, remove WheelsOnHourDis
    if 'WheelsOnHour' in df.columns and 'WheelsOnHourDis' in df.columns:
        time_cols_to_remove.append('WheelsOnHourDis')
        print("✅ Removing WheelsOnHourDis (keeping WheelsOnHour)")
    
    columns_to_remove.extend(time_cols_to_remove)
    
    # Remove the identified columns
    df_cleaned = df.drop(columns=columns_to_remove)
    
    removed_count = original_cols - len(df_cleaned.columns)
    print(f"\n📊 REDUNDANCY REMOVAL SUMMARY:")
    print(f"   Original columns: {original_cols}")
    print(f"   Removed columns: {removed_count}")
    print(f"   Final columns: {len(df_cleaned.columns)}")
    print(f"   Removed columns: {columns_to_remove}")
    
    return df_cleaned

def preprocess_data_types_and_features(df):
    """
    Convert data types and create new features
    """
    print(f"\n🔧 DATA TYPE CONVERSION AND FEATURE CREATION")
    print("-" * 50)
    
    # Step 1: Remove redundant columns
    redundant_cols = ['Year', 'Month', 'DayofMonth']
    cols_to_remove = [col for col in redundant_cols if col in df.columns]
    
    if cols_to_remove:
        print(f"Removing redundant columns: {cols_to_remove}")
        df = df.drop(columns=cols_to_remove)
        print(f"✅ Removed {len(cols_to_remove)} redundant columns")
    
    # Step 2: Convert FlightDate to DateTime
    if 'FlightDate' in df.columns:
        print(f"Converting FlightDate to DepDate (DateTime)...")
        df['DepDate'] = pd.to_datetime(df['FlightDate'])
        df = df.drop(columns=['FlightDate'])
        print(f"✅ Converted FlightDate to DepDate (DateTime)")
        print(f"Date range: {df['DepDate'].min()} to {df['DepDate'].max()}")
    
    # Step 3: Split Origin and Destination columns
    if 'OriginCityName' in df.columns:
        print("Splitting OriginCityName...")
        df[['OriginCity', 'OriginState']] = df['OriginCityName'].str.split(', ', expand=True)
        df = df.drop(columns=['OriginCityName'])
        print(f"✅ Created OriginCity and OriginState columns")
    
    if 'DestCityName' in df.columns:
        print("Splitting DestCityName...")
        df[['DestCity', 'DestState']] = df['DestCityName'].str.split(', ', expand=True)
        df = df.drop(columns=['DestCityName'])
        print(f"✅ Created DestCity and DestState columns")
    
    # Step 4: Map Marketing_Airline_Network to Carrier Name
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
        df['CarrierName'] = df['Marketing_Airline_Network'].map(airline_mapping)
        df['CarrierName'] = df['CarrierName'].fillna('Unknown Airline')
        print(f"✅ Created CarrierName column")
    
    # Step 5: Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"✅ Removed {duplicates_removed:,} duplicate rows")
    
    # Step 6: Create additional useful features
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
    
    # Create route feature
    if 'OriginCity' in df.columns and 'DestCity' in df.columns:
        df['Route'] = df['OriginCity'] + ' → ' + df['DestCity']
        print(f"✅ Created Route feature")
    
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
    
    return df

def filter_covid_data(df):
    """
    Remove COVID years (2020-2021) from the dataset
    """
    print(f"\n🦠 FILTERING OUT COVID DATA (2020-2021)")
    print("-" * 50)
    
    # Show year distribution before filtering
    print(f"Original year distribution:")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} flights")
    
    # Filter out COVID years
    covid_years = [2020, 2021]
    df_filtered = df[~df['Year'].isin(covid_years)]
    
    print(f"\nFiltered dataset: {df_filtered.shape}")
    print(f"Removed {len(df) - len(df_filtered):,} flights ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")
    
    # Show filtered year distribution
    print(f"\nFiltered year distribution:")
    filtered_year_counts = df_filtered['Year'].value_counts().sort_index()
    for year, count in filtered_year_counts.items():
        print(f"  {year}: {count:,} flights")
    
    # Analyze delay patterns in filtered data
    print(f"\nDelay patterns in filtered dataset:")
    avg_delay = df_filtered['DepDelayMinutes'].mean()
    on_time_rate = (df_filtered['DepDelayMinutes'] == 0).mean() * 100
    major_delay_rate = (df_filtered['DepDelayMinutes'] > 60).mean() * 100
    
    print(f"  Average departure delay: {avg_delay:.1f} minutes")
    print(f"  On-time rate: {on_time_rate:.1f}%")
    print(f"  Major delay rate: {major_delay_rate:.1f}%")
    
    return df_filtered

def complete_preprocessing_pipeline():
    """
    Complete preprocessing pipeline from raw data to ML-ready dataset
    """
    print("🚀 SOUTHWEST AIRLINES COMPLETE PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_file = project_root / "data" / "csv_data" / "features_added_southwest.csv"
    output_dir = project_root / "data" / "preprocessed_data"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Load data
    print(f"📂 STEP 1: LOADING DATA")
    print("-" * 30)
    print(f"Loading from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original dataset: {df.shape}")
    
    # Step 2: Initial data quality analysis
    print(f"\n📊 STEP 2: INITIAL DATA QUALITY ANALYSIS")
    print("-" * 30)
    analyze_data_quality(df, "ORIGINAL DATASET")
    
    # Step 3: Remove redundant columns
    print(f"\n🔧 STEP 3: REMOVING REDUNDANT COLUMNS")
    print("-" * 30)
    df = remove_redundant_columns(df)
    
    # Step 4: Data type conversion and feature creation
    print(f"\n🔧 STEP 4: DATA TYPE CONVERSION AND FEATURE CREATION")
    print("-" * 30)
    df = preprocess_data_types_and_features(df)
    
    # Step 5: Filter COVID data
    print(f"\n🦠 STEP 5: FILTERING COVID DATA")
    print("-" * 30)
    df = filter_covid_data(df)
    
    # Step 6: Final data quality check
    print(f"\n📊 STEP 6: FINAL DATA QUALITY CHECK")
    print("-" * 30)
    analyze_data_quality(df, "FINAL PREPROCESSED DATASET")
    
    # Step 7: Save final dataset
    output_file = output_dir / "southwest_final_preprocessed.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 STEP 7: SAVING FINAL DATASET")
    print("-" * 30)
    print(f"✅ Saved to: {output_file}")
    print(f"Final dataset: {df.shape}")
    print(f"Final file size: {output_file.stat().st_size / (1024**2):.1f} MB")
    
    # Final summary
    print(f"\n🎯 PREPROCESSING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"✅ Input: features_added_southwest.csv ({pd.read_csv(input_file).shape})")
    print(f"✅ Output: southwest_final_preprocessed.csv ({df.shape})")
    print(f"✅ Removed redundant columns")
    print(f"✅ Converted data types")
    print(f"✅ Created new features")
    print(f"✅ Filtered COVID data (2020-2021)")
    print(f"✅ Dataset ready for ML modeling!")
    
    # Show final column list
    print(f"\n📋 FINAL DATASET COLUMNS ({len(df.columns)}):")
    print("-" * 50)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    return df

if __name__ == "__main__":
    final_df = complete_preprocessing_pipeline()
