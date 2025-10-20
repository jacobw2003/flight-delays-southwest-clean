#!/usr/bin/env python3
"""
Script to convert parquet files to CSV format and filter for Southwest Airlines flights.
"""

import pandas as pd
import os
from pathlib import Path

def convert_parquet_to_csv():
    """
    Convert parquet files from raw_data directory to CSV format,
    filtering for Southwest Airlines (WN) flights only.
    """
    
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    raw_data_dir = project_root / "data" / "raw_data"
    output_dir = project_root / "data"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # List of parquet files to process
    parquet_files = [
        "Flight_Delay.parquet",
        "features_added.parquet"
    ]
    
    print("Starting parquet to CSV conversion...")
    print("=" * 50)
    
    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file}...")
        
        # Read parquet file
        input_path = raw_data_dir / parquet_file
        df = pd.read_parquet(input_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Total flights: {len(df):,}")
        
        # Filter for Southwest Airlines (WN)
        southwest_mask = df['Marketing_Airline_Network'] == 'WN'
        df_southwest = df[southwest_mask].copy()
        
        print(f"Southwest Airlines flights: {len(df_southwest):,}")
        print(f"Percentage of total: {len(df_southwest)/len(df)*100:.2f}%")
        
        # Create output filename
        csv_filename = parquet_file.replace('.parquet', '_southwest.csv')
        output_path = output_dir / csv_filename
        
        # Save to CSV
        df_southwest.to_csv(output_path, index=False)
        print(f"Saved Southwest Airlines data to: {output_path}")
        
        # Show first 100 rows for overview
        print(f"\nFirst 100 rows overview:")
        print("-" * 30)
        print(df_southwest.head(100))
        
        # Show basic statistics
        print(f"\nBasic statistics for Southwest Airlines flights:")
        print("-" * 50)
        print(f"Date range: {df_southwest['FlightDate'].min()} to {df_southwest['FlightDate'].max()}")
        print(f"Unique origin cities: {df_southwest['OriginCityName'].nunique()}")
        print(f"Unique destination cities: {df_southwest['DestCityName'].nunique()}")
        
        # Show delay statistics
        if 'DepDelayMinutes' in df_southwest.columns:
            print(f"\nDeparture delay statistics:")
            print(f"Average departure delay: {df_southwest['DepDelayMinutes'].mean():.2f} minutes")
            print(f"Max departure delay: {df_southwest['DepDelayMinutes'].max()} minutes")
            print(f"Flights with delays: {(df_southwest['DepDelayMinutes'] > 0).sum():,}")
        
        if 'ArrDelayMinutes' in df_southwest.columns:
            print(f"\nArrival delay statistics:")
            print(f"Average arrival delay: {df_southwest['ArrDelayMinutes'].mean():.2f} minutes")
            print(f"Max arrival delay: {df_southwest['ArrDelayMinutes'].max()} minutes")
            print(f"Flights with arrival delays: {(df_southwest['ArrDelayMinutes'] > 0).sum():,}")
        
        print("\n" + "=" * 50)
    
    print("\nConversion completed successfully!")
    print("All Southwest Airlines data has been saved as CSV files in the data directory.")

if __name__ == "__main__":
    convert_parquet_to_csv()
