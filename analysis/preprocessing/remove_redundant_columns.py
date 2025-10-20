#!/usr/bin/env python3
"""
Updated Data Preprocessing Script - Removing Redundant Columns

This script removes redundant columns that provide the same information:
1. DepDelay vs DepDelayMinutes (keep DepDelayMinutes - non-negative)
2. ArrDelay vs ArrDelayMinutes (keep ArrDelayMinutes - non-negative)  
3. DayofWeek vs DayOfWeek (keep DayOfWeek - better naming)
4. Marketing_Airline_Network vs CarrierName (keep CarrierName - more descriptive)
5. Redundant time columns (keep most useful ones)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def remove_redundant_columns(df):
    """
    Remove redundant columns that provide the same information
    """
    print("🔧 REMOVING REDUNDANT COLUMNS")
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
    
    # 6. Remove redundant date columns - DepDate contains all date info
    date_cols_to_remove = []
    if 'DepDate' in df.columns:
        # Keep Year, Month, Day, Quarter, Season for analysis
        # But we could remove them if DepDate is sufficient
        print("ℹ️  Keeping date components (Year, Month, Day, Quarter, Season) for analysis")
    
    # Remove the identified columns
    df_cleaned = df.drop(columns=columns_to_remove)
    
    removed_count = original_cols - len(df_cleaned.columns)
    print(f"\n📊 REDUNDANCY REMOVAL SUMMARY:")
    print(f"   Original columns: {original_cols}")
    print(f"   Removed columns: {removed_count}")
    print(f"   Final columns: {len(df_cleaned.columns)}")
    print(f"   Removed columns: {columns_to_remove}")
    
    return df_cleaned

def analyze_final_dataset(df):
    """
    Analyze the final cleaned dataset
    """
    print(f"\n📈 FINAL DATASET ANALYSIS")
    print("=" * 50)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📋 FINAL COLUMNS ({len(df.columns)}):")
    print("-" * 30)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Key statistics
    print(f"\n🔑 KEY STATISTICS:")
    print("-" * 30)
    
    if 'DepDelayMinutes' in df.columns:
        delays = df['DepDelayMinutes']
        print(f"Departure Delays:")
        print(f"  On time (0 min): {(delays == 0).sum():,} ({(delays == 0).mean()*100:.1f}%)")
        print(f"  Minor delays (1-15 min): {((delays > 0) & (delays <= 15)).sum():,}")
        print(f"  Moderate delays (16-60 min): {((delays > 15) & (delays <= 60)).sum():,}")
        print(f"  Major delays (>60 min): {(delays > 60).sum():,}")
    
    if 'Season' in df.columns:
        print(f"\nSeasonal Distribution:")
        print(df['Season'].value_counts())
    
    if 'Route' in df.columns:
        print(f"\nTop 10 Busiest Routes:")
        print(df['Route'].value_counts().head(10))

def clean_preprocessed_data():
    """
    Clean the preprocessed data by removing redundant columns
    """
    print("🧹 CLEANING PREPROCESSED DATA - REMOVING REDUNDANCIES")
    print("=" * 60)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_file = project_root / "data" / "preprocessed_data" / "southwest_preprocessed.csv"
    output_file = project_root / "data" / "preprocessed_data" / "southwest_cleaned.csv"
    
    # Load preprocessed data
    print(f"📂 Loading preprocessed data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    
    # Remove redundant columns
    df_cleaned = remove_redundant_columns(df)
    
    # Analyze final dataset
    analyze_final_dataset(df_cleaned)
    
    # Save cleaned data
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n💾 SAVED CLEANED DATA")
    print("-" * 30)
    print(f"✅ Saved to: {output_file}")
    print(f"Final dataset: {df_cleaned.shape}")
    
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = clean_preprocessed_data()
