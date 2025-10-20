#!/usr/bin/env python3
"""
COVID Data Filtering Script

Removes 2020-2021 data from the dataset to create a more consistent
dataset for ML modeling, excluding the anomalous COVID period.
"""

import pandas as pd
from pathlib import Path

def filter_covid_data():
    """
    Remove COVID years (2020-2021) from the dataset
    """
    print("🦠 FILTERING OUT COVID DATA (2020-2021)")
    print("=" * 50)
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_file = project_root / "data" / "preprocessed_data" / "southwest_cleaned.csv"
    output_file = project_root / "data" / "preprocessed_data" / "southwest_no_covid.csv"
    
    # Load data
    print(f"📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original dataset: {df.shape}")
    
    # Show year distribution
    print(f"\nOriginal year distribution:")
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
    
    # Save filtered dataset
    df_filtered.to_csv(output_file, index=False)
    print(f"\n💾 Saved filtered dataset to: {output_file}")
    
    # Show file sizes
    original_size = input_file.stat().st_size / (1024**2)
    filtered_size = output_file.stat().st_size / (1024**2)
    print(f"  Original file size: {original_size:.1f} MB")
    print(f"  Filtered file size: {filtered_size:.1f} MB")
    print(f"  Size reduction: {original_size - filtered_size:.1f} MB ({((original_size - filtered_size)/original_size)*100:.1f}%)")
    
    print(f"\n✅ COVID filtering complete!")
    print(f"   Ready for ML modeling with consistent operational data")
    
    return df_filtered

if __name__ == "__main__":
    filtered_df = filter_covid_data()
