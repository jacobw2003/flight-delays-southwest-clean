#!/usr/bin/env python3
"""
Comprehensive Data Analysis for Southwest Airlines Flight Delay Prediction

This script performs a thorough analysis of Southwest Airlines flight data to address
the core problem statement:

Problem Statement: Develop a predictive model that can forecast the likelihood
and expected duration of flight delays for Southwest Airlines based on historical 
operational and environmental data. The goal is to accurately estimate delay 
probability and average delay time for future flights, enabling Southwest Airlines 
to proactively mitigate disruptions, optimize scheduling decisions, and improve 
overall customer experience.

Key Questions Addressed:
1. Which routes experience the most frequent and severe delays?
2. What are the primary causes of delays for Southwest Airlines?
3. How can Southwest Airlines predict future delays?
4. Are there seasonal patterns?

Analysis includes:
- Route delay analysis (frequency and severity)
- Delay cause identification
- Seasonal pattern analysis
- STL decomposition for time series analysis
- Predictive insights and recommendations
- Statistical analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
import plotly.offline as pyo

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SouthwestDataAnalyzer:
    """
    Comprehensive analyzer for Southwest Airlines flight delay data
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the analyzer with data path
        """
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "preprocessed_data" / "southwest_final_preprocessed.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.results = {}
        self.plots_dir = Path(__file__).parent / "analysis_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """
        Load and prepare the dataset
        """
        print("📂 LOADING SOUTHWEST AIRLINES DATA")
        print("=" * 60)
        
        try:
            # Load data in chunks to handle large files
            chunk_size = 100000
            chunks = []
            
            print(f"Loading data from: {self.data_path}")
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunks.append(chunk)
            
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"✅ Data loaded successfully: {self.df.shape}")
            
            # Convert DepDate to datetime if needed
            if 'DepDate' in self.df.columns:
                if self.df['DepDate'].dtype == 'object':
                    self.df['DepDate'] = pd.to_datetime(self.df['DepDate'])
                print(f"Date range: {self.df['DepDate'].min()} to {self.df['DepDate'].max()}")
            
            # Basic data info
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"Columns: {list(self.df.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_route_delays(self):
        """
        Analyze which routes experience the most frequent and severe delays
        """
        print("\n🛫 ROUTE DELAY ANALYSIS")
        print("=" * 60)
        
        if 'Route' not in self.df.columns or 'DepDelayMinutes' not in self.df.columns:
            print("❌ Required columns (Route, DepDelayMinutes) not found")
            return None
        
        # Calculate route statistics
        route_stats = self.df.groupby('Route').agg({
            'DepDelayMinutes': ['count', 'mean', 'median', 'std', 'max'],
            'DepDate': 'nunique'  # Number of unique days
        }).round(2)
        
        # Flatten column names
        route_stats.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay', 'Delay_Std', 'Max_Delay', 'Days_Operated']
        
        # Calculate additional metrics
        route_stats['Delay_Frequency'] = (self.df.groupby('Route')['DepDelayMinutes'].apply(lambda x: (x > 0).mean() * 100)).round(2)
        route_stats['Major_Delay_Rate'] = (self.df.groupby('Route')['DepDelayMinutes'].apply(lambda x: (x > 60).mean() * 100)).round(2)
        route_stats['On_Time_Rate'] = (self.df.groupby('Route')['DepDelayMinutes'].apply(lambda x: (x == 0).mean() * 100)).round(2)
        
        # Filter routes with sufficient data (at least 100 flights)
        route_stats = route_stats[route_stats['Flight_Count'] >= 100].copy()
        
        # Sort by different criteria
        most_delayed_routes = route_stats.sort_values('Avg_Delay', ascending=False).head(20)
        most_frequent_delays = route_stats.sort_values('Delay_Frequency', ascending=False).head(20)
        worst_performing_routes = route_stats.sort_values('Major_Delay_Rate', ascending=False).head(20)
        
        print("📊 TOP 20 ROUTES BY AVERAGE DELAY:")
        print("-" * 50)
        for route, stats in most_delayed_routes.iterrows():
            print(f"{route}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        print("\n📊 TOP 20 ROUTES BY DELAY FREQUENCY:")
        print("-" * 50)
        for route, stats in most_frequent_delays.iterrows():
            print(f"{route}: {stats['Delay_Frequency']:.1f}% delayed ({stats['Flight_Count']} flights)")
        
        print("\n📊 TOP 20 ROUTES BY MAJOR DELAY RATE:")
        print("-" * 50)
        for route, stats in worst_performing_routes.iterrows():
            print(f"{route}: {stats['Major_Delay_Rate']:.1f}% major delays ({stats['Flight_Count']} flights)")
        
        # Store results
        self.results['route_analysis'] = {
            'route_stats': route_stats,
            'most_delayed_routes': most_delayed_routes,
            'most_frequent_delays': most_frequent_delays,
            'worst_performing_routes': worst_performing_routes
        }
        
        return route_stats
    
    def analyze_delay_causes(self):
        """
        Identify primary causes of delays for Southwest Airlines
        """
        print("\n🔍 DELAY CAUSE ANALYSIS")
        print("=" * 60)
        
        # Check for delay cause columns
        delay_cause_cols = [col for col in self.df.columns if 'delay' in col.lower() and 'cause' in col.lower()]
        weather_cols = [col for col in self.df.columns if 'weather' in col.lower()]
        carrier_cols = [col for col in self.df.columns if 'carrier' in col.lower()]
        nas_cols = [col for col in self.df.columns if 'nas' in col.lower()]
        security_cols = [col for col in self.df.columns if 'security' in col.lower()]
        late_aircraft_cols = [col for col in self.df.columns if 'late' in col.lower() and 'aircraft' in col.lower()]
        
        print(f"Available delay cause columns: {delay_cause_cols}")
        print(f"Weather-related columns: {weather_cols}")
        print(f"Carrier-related columns: {carrier_cols}")
        print(f"NAS-related columns: {nas_cols}")
        print(f"Security-related columns: {security_cols}")
        print(f"Late aircraft columns: {late_aircraft_cols}")
        
        # Analyze delay patterns by time of day
        if 'CRSDepTimeHour' in self.df.columns:
            print("\n📊 DELAY PATTERNS BY TIME OF DAY:")
            print("-" * 40)
            hourly_delays = self.df.groupby('CRSDepTimeHour')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
            hourly_delays.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
            
            # Show worst hours
            worst_hours = hourly_delays.sort_values('Avg_Delay', ascending=False).head(10)
            print("Worst departure hours by average delay:")
            for hour, stats in worst_hours.iterrows():
                print(f"  {hour:02d}:00 - {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Analyze delay patterns by day of week
        if 'DayOfWeek' in self.df.columns:
            print("\n📊 DELAY PATTERNS BY DAY OF WEEK:")
            print("-" * 40)
            daily_delays = self.df.groupby('DayOfWeek')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
            daily_delays.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_delays.index = [day_names[i] for i in daily_delays.index]
            
            print("Average delays by day of week:")
            for day, stats in daily_delays.iterrows():
                print(f"  {day}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Analyze delay patterns by season
        if 'Season' in self.df.columns:
            print("\n📊 DELAY PATTERNS BY SEASON:")
            print("-" * 40)
            seasonal_delays = self.df.groupby('Season')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
            seasonal_delays.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
            
            print("Average delays by season:")
            for season, stats in seasonal_delays.iterrows():
                print(f"  {season}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Analyze delay patterns by airport
        if 'OriginCity' in self.df.columns:
            print("\n📊 DELAY PATTERNS BY ORIGIN AIRPORT:")
            print("-" * 40)
            airport_delays = self.df.groupby('OriginCity')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
            airport_delays.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
            
            # Filter airports with sufficient data
            airport_delays = airport_delays[airport_delays['Flight_Count'] >= 1000]
            worst_airports = airport_delays.sort_values('Avg_Delay', ascending=False).head(10)
            
            print("Worst origin airports by average delay:")
            for airport, stats in worst_airports.iterrows():
                print(f"  {airport}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Store results
        self.results['delay_causes'] = {
            'hourly_delays': hourly_delays if 'CRSDepTimeHour' in self.df.columns else None,
            'daily_delays': daily_delays if 'DayOfWeek' in self.df.columns else None,
            'seasonal_delays': seasonal_delays if 'Season' in self.df.columns else None,
            'airport_delays': airport_delays if 'OriginCity' in self.df.columns else None
        }
        
        return self.results['delay_causes']
    
    def analyze_seasonal_patterns(self):
        """
        Analyze seasonal patterns in flight delays
        """
        print("\n📅 SEASONAL PATTERN ANALYSIS")
        print("=" * 60)
        
        if 'DepDate' not in self.df.columns or 'DepDelayMinutes' not in self.df.columns:
            print("❌ Required columns (DepDate, DepDelayMinutes) not found")
            return None
        
        # Create time-based features
        self.df['Year'] = self.df['DepDate'].dt.year
        self.df['Month'] = self.df['DepDate'].dt.month
        self.df['DayOfWeek'] = self.df['DepDate'].dt.dayofweek
        self.df['Quarter'] = self.df['DepDate'].dt.quarter
        
        # Monthly analysis
        print("📊 MONTHLY DELAY PATTERNS:")
        print("-" * 40)
        monthly_stats = self.df.groupby('Month')['DepDelayMinutes'].agg(['count', 'mean', 'median', 'std']).round(2)
        monthly_stats.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay', 'Delay_Std']
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_stats.index = [month_names[i-1] for i in monthly_stats.index]
        
        print("Average delays by month:")
        for month, stats in monthly_stats.iterrows():
            print(f"  {month}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Quarterly analysis
        print("\n📊 QUARTERLY DELAY PATTERNS:")
        print("-" * 40)
        quarterly_stats = self.df.groupby('Quarter')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
        quarterly_stats.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
        
        quarter_names = ['Q1 (Winter)', 'Q2 (Spring)', 'Q3 (Summer)', 'Q4 (Fall)']
        quarterly_stats.index = [quarter_names[i-1] for i in quarterly_stats.index]
        
        print("Average delays by quarter:")
        for quarter, stats in quarterly_stats.iterrows():
            print(f"  {quarter}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Year-over-year analysis
        print("\n📊 YEAR-OVER-YEAR DELAY PATTERNS:")
        print("-" * 40)
        yearly_stats = self.df.groupby('Year')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
        yearly_stats.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
        
        print("Average delays by year:")
        for year, stats in yearly_stats.iterrows():
            print(f"  {year}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Day of week analysis
        print("\n📊 DAY OF WEEK DELAY PATTERNS:")
        print("-" * 40)
        daily_stats = self.df.groupby('DayOfWeek')['DepDelayMinutes'].agg(['count', 'mean', 'median']).round(2)
        daily_stats.columns = ['Flight_Count', 'Avg_Delay', 'Median_Delay']
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats.index = [day_names[i] for i in daily_stats.index]
        
        print("Average delays by day of week:")
        for day, stats in daily_stats.iterrows():
            print(f"  {day}: {stats['Avg_Delay']:.1f} min avg ({stats['Flight_Count']} flights)")
        
        # Store results
        self.results['seasonal_patterns'] = {
            'monthly_stats': monthly_stats,
            'quarterly_stats': quarterly_stats,
            'yearly_stats': yearly_stats,
            'daily_stats': daily_stats
        }
        
        return self.results['seasonal_patterns']
    
    def perform_stl_decomposition(self):
        """
        Perform STL decomposition for time series analysis
        """
        print("\n📈 STL DECOMPOSITION ANALYSIS")
        print("=" * 60)
        
        if 'DepDate' not in self.df.columns or 'DepDelayMinutes' not in self.df.columns:
            print("❌ Required columns (DepDate, DepDelayMinutes) not found")
            return None
        
        try:
            # Prepare time series data
            # Aggregate delays by day
            daily_delays = self.df.groupby('DepDate')['DepDelayMinutes'].mean().reset_index()
            daily_delays = daily_delays.set_index('DepDate').sort_index()
            
            # Fill missing dates with NaN
            date_range = pd.date_range(start=daily_delays.index.min(), 
                                     end=daily_delays.index.max(), 
                                     freq='D')
            daily_delays = daily_delays.reindex(date_range)
            
            # Check for COVID gap and handle it properly
            covid_start = pd.Timestamp('2020-01-01')
            covid_end = pd.Timestamp('2021-12-31')
            
            # Check if we have any data in COVID period
            covid_data = daily_delays[(daily_delays.index >= covid_start) & (daily_delays.index <= covid_end)]
            covid_gap_exists = covid_data.isnull().all().iloc[0] if len(covid_data) > 0 else True
            
            if covid_gap_exists:
                print(f"⚠️  COVID gap detected (2020-2021) - using interpolation instead of forward-fill")
                # Use interpolation to handle the gap more gracefully
                daily_delays = daily_delays.interpolate(method='linear')
            else:
                # Forward fill missing values for small gaps
                daily_delays = daily_delays.fillna(method='ffill')
            
            print(f"Time series data prepared: {len(daily_delays)} days")
            print(f"Date range: {daily_delays.index.min()} to {daily_delays.index.max()}")
            
            # Perform STL decomposition
            # Use seasonal period of 365 days (annual seasonality)
            stl = STL(daily_delays['DepDelayMinutes'], seasonal=365, robust=True)
            result = stl.fit()
            
            print("✅ STL decomposition completed successfully")
            
            # Extract components
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            # Calculate statistics
            trend_stats = {
                'mean': trend.mean(),
                'std': trend.std(),
                'min': trend.min(),
                'max': trend.max()
            }
            
            seasonal_stats = {
                'mean': seasonal.mean(),
                'std': seasonal.std(),
                'min': seasonal.min(),
                'max': seasonal.max()
            }
            
            residual_stats = {
                'mean': residual.mean(),
                'std': residual.std(),
                'min': residual.min(),
                'max': residual.max()
            }
            
            print("\n📊 STL COMPONENT STATISTICS:")
            print("-" * 40)
            print(f"Trend component:")
            print(f"  Mean: {trend_stats['mean']:.2f}")
            print(f"  Std: {trend_stats['std']:.2f}")
            print(f"  Range: {trend_stats['min']:.2f} to {trend_stats['max']:.2f}")
            
            print(f"\nSeasonal component:")
            print(f"  Mean: {seasonal_stats['mean']:.2f}")
            print(f"  Std: {seasonal_stats['std']:.2f}")
            print(f"  Range: {seasonal_stats['min']:.2f} to {seasonal_stats['max']:.2f}")
            
            print(f"\nResidual component:")
            print(f"  Mean: {residual_stats['mean']:.2f}")
            print(f"  Std: {residual_stats['std']:.2f}")
            print(f"  Range: {residual_stats['min']:.2f} to {residual_stats['max']:.2f}")
            
            # Store results
            self.results['stl_decomposition'] = {
                'daily_delays': daily_delays,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'trend_stats': trend_stats,
                'seasonal_stats': seasonal_stats,
                'residual_stats': residual_stats
            }
            
            return self.results['stl_decomposition']
            
        except Exception as e:
            print(f"❌ Error in STL decomposition: {e}")
            return None
    
    def create_visualizations(self):
        """
        Create comprehensive visualizations for the analysis
        """
        print("\n📊 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Route Analysis Visualization
        if 'route_analysis' in self.results:
            self._create_route_visualizations()
        
        # 2. Delay Cause Analysis Visualization
        if 'delay_causes' in self.results:
            self._create_delay_cause_visualizations()
        
        # 3. Seasonal Pattern Visualization
        if 'seasonal_patterns' in self.results:
            self._create_seasonal_visualizations()
        
        # 4. STL Decomposition Visualization
        if 'stl_decomposition' in self.results:
            self._create_stl_visualizations()
        
        print(f"✅ All visualizations saved to: {self.plots_dir}")
    
    def _create_route_visualizations(self):
        """Create route analysis visualizations"""
        route_stats = self.results['route_analysis']['route_stats']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Southwest Airlines Route Delay Analysis', fontsize=16, fontweight='bold')
        
        # Top 20 routes by average delay
        top_delayed = route_stats.nlargest(20, 'Avg_Delay')
        axes[0, 0].barh(range(len(top_delayed)), top_delayed['Avg_Delay'])
        axes[0, 0].set_yticks(range(len(top_delayed)))
        axes[0, 0].set_yticklabels([route[:20] + '...' if len(route) > 20 else route for route in top_delayed.index])
        axes[0, 0].set_xlabel('Average Delay (minutes)')
        axes[0, 0].set_title('Top 20 Routes by Average Delay')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top 20 routes by delay frequency
        top_frequent = route_stats.nlargest(20, 'Delay_Frequency')
        axes[0, 1].barh(range(len(top_frequent)), top_frequent['Delay_Frequency'])
        axes[0, 1].set_yticks(range(len(top_frequent)))
        axes[0, 1].set_yticklabels([route[:20] + '...' if len(route) > 20 else route for route in top_frequent.index])
        axes[0, 1].set_xlabel('Delay Frequency (%)')
        axes[0, 1].set_title('Top 20 Routes by Delay Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Flight count vs average delay scatter
        axes[1, 0].scatter(route_stats['Flight_Count'], route_stats['Avg_Delay'], alpha=0.6)
        axes[1, 0].set_xlabel('Number of Flights')
        axes[1, 0].set_ylabel('Average Delay (minutes)')
        axes[1, 0].set_title('Flight Volume vs Average Delay')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Major delay rate distribution
        axes[1, 1].hist(route_stats['Major_Delay_Rate'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Major Delay Rate (%)')
        axes[1, 1].set_ylabel('Number of Routes')
        axes[1, 1].set_title('Distribution of Major Delay Rates')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'route_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_delay_cause_visualizations(self):
        """Create delay cause analysis visualizations"""
        delay_causes = self.results['delay_causes']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Southwest Airlines Delay Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Hourly delay patterns
        if delay_causes['hourly_delays'] is not None:
            hourly = delay_causes['hourly_delays']
            axes[0, 0].plot(hourly.index, hourly['Avg_Delay'], marker='o', linewidth=2)
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Average Delay (minutes)')
            axes[0, 0].set_title('Average Delay by Hour of Day')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Daily delay patterns
        if delay_causes['daily_delays'] is not None:
            daily = delay_causes['daily_delays']
            axes[0, 1].bar(range(len(daily)), daily['Avg_Delay'])
            axes[0, 1].set_xticks(range(len(daily)))
            axes[0, 1].set_xticklabels(daily.index, rotation=45)
            axes[0, 1].set_ylabel('Average Delay (minutes)')
            axes[0, 1].set_title('Average Delay by Day of Week')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Seasonal delay patterns
        if delay_causes['seasonal_delays'] is not None:
            seasonal = delay_causes['seasonal_delays']
            axes[1, 0].bar(range(len(seasonal)), seasonal['Avg_Delay'])
            axes[1, 0].set_xticks(range(len(seasonal)))
            axes[1, 0].set_xticklabels(seasonal.index)
            axes[1, 0].set_ylabel('Average Delay (minutes)')
            axes[1, 0].set_title('Average Delay by Season')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Airport delay patterns (top 15)
        if delay_causes['airport_delays'] is not None:
            airport = delay_causes['airport_delays']
            top_airports = airport.nlargest(15, 'Avg_Delay')
            axes[1, 1].barh(range(len(top_airports)), top_airports['Avg_Delay'])
            axes[1, 1].set_yticks(range(len(top_airports)))
            axes[1, 1].set_yticklabels(top_airports.index)
            axes[1, 1].set_xlabel('Average Delay (minutes)')
            axes[1, 1].set_title('Top 15 Airports by Average Delay')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'delay_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_seasonal_visualizations(self):
        """Create seasonal pattern visualizations"""
        seasonal = self.results['seasonal_patterns']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Southwest Airlines Seasonal Delay Patterns', fontsize=16, fontweight='bold')
        
        # Monthly patterns
        monthly = seasonal['monthly_stats']
        axes[0, 0].plot(range(1, 13), monthly['Avg_Delay'], marker='o', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Delay (minutes)')
        axes[0, 0].set_title('Average Delay by Month')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].set_xticklabels(monthly.index)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Quarterly patterns
        quarterly = seasonal['quarterly_stats']
        axes[0, 1].bar(range(1, 5), quarterly['Avg_Delay'])
        axes[0, 1].set_xlabel('Quarter')
        axes[0, 1].set_ylabel('Average Delay (minutes)')
        axes[0, 1].set_title('Average Delay by Quarter')
        axes[0, 1].set_xticks(range(1, 5))
        axes[0, 1].set_xticklabels(quarterly.index)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Yearly patterns
        yearly = seasonal['yearly_stats']
        axes[1, 0].plot(yearly.index, yearly['Avg_Delay'], marker='o', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Delay (minutes)')
        axes[1, 0].set_title('Average Delay by Year')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Day of week patterns
        daily = seasonal['daily_stats']
        axes[1, 1].bar(range(7), daily['Avg_Delay'])
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Delay (minutes)')
        axes[1, 1].set_title('Average Delay by Day of Week')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(daily.index, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_stl_visualizations(self):
        """Create STL decomposition visualizations"""
        stl_results = self.results['stl_decomposition']
        
        fig, axes = plt.subplots(4, 1, figsize=(20, 16))
        fig.suptitle('STL Decomposition of Southwest Airlines Delay Time Series\n(COVID Gap: 2020-2021)', fontsize=16, fontweight='bold')
        
        # Original time series
        axes[0].plot(stl_results['daily_delays'].index, stl_results['daily_delays']['DepDelayMinutes'], alpha=0.7)
        
        # Highlight COVID gap
        covid_start = pd.Timestamp('2020-01-01')
        covid_end = pd.Timestamp('2021-12-31')
        axes[0].axvspan(covid_start, covid_end, alpha=0.3, color='red', label='COVID Gap (2020-2021)')
        axes[0].set_title('Original Time Series (Daily Average Delay)')
        axes[0].set_ylabel('Delay (minutes)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Trend component
        axes[1].plot(stl_results['trend'].index, stl_results['trend'], color='red', linewidth=2)
        axes[1].axvspan(covid_start, covid_end, alpha=0.3, color='red')
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Delay (minutes)')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal component
        axes[2].plot(stl_results['seasonal'].index, stl_results['seasonal'], color='green', linewidth=2)
        axes[2].axvspan(covid_start, covid_end, alpha=0.3, color='red')
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Delay (minutes)')
        axes[2].grid(True, alpha=0.3)
        
        # Residual component
        axes[3].plot(stl_results['residual'].index, stl_results['residual'], color='purple', linewidth=2)
        axes[3].axvspan(covid_start, covid_end, alpha=0.3, color='red')
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Delay (minutes)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'stl_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_predictive_insights(self):
        """
        Generate insights for predicting future delays
        """
        print("\n🎯 PREDICTIVE INSIGHTS AND RECOMMENDATIONS")
        print("=" * 60)
        
        insights = []
        
        # Route-based insights
        if 'route_analysis' in self.results:
            route_stats = self.results['route_analysis']['route_stats']
            
            # Identify high-risk routes
            high_risk_routes = route_stats[
                (route_stats['Avg_Delay'] > route_stats['Avg_Delay'].quantile(0.8)) &
                (route_stats['Major_Delay_Rate'] > route_stats['Major_Delay_Rate'].quantile(0.8))
            ]
            
            insights.append({
                'category': 'Route Risk',
                'finding': f"{len(high_risk_routes)} routes identified as high-risk",
                'recommendation': 'Implement proactive monitoring and contingency planning for these routes'
            })
        
        # Time-based insights
        if 'delay_causes' in self.results and self.results['delay_causes']['hourly_delays'] is not None:
            hourly = self.results['delay_causes']['hourly_delays']
            worst_hours = hourly.nlargest(3, 'Avg_Delay')
            
            insights.append({
                'category': 'Time Patterns',
                'finding': f"Worst departure hours: {', '.join([f'{h:02d}:00' for h in worst_hours.index])}",
                'recommendation': 'Consider adjusting schedules or adding buffer time during peak delay hours'
            })
        
        # Seasonal insights
        if 'seasonal_patterns' in self.results:
            monthly = self.results['seasonal_patterns']['monthly_stats']
            worst_month = monthly.loc[monthly['Avg_Delay'].idxmax()]
            best_month = monthly.loc[monthly['Avg_Delay'].idxmin()]
            
            insights.append({
                'category': 'Seasonal Patterns',
                'finding': f"Highest delays in {worst_month.name} ({worst_month['Avg_Delay']:.1f} min), lowest in {best_month.name} ({best_month['Avg_Delay']:.1f} min)",
                'recommendation': 'Adjust operational capacity and staffing during high-risk months'
            })
        
        # STL insights
        if 'stl_decomposition' in self.results:
            stl_results = self.results['stl_decomposition']
            trend_slope = np.polyfit(range(len(stl_results['trend'])), stl_results['trend'].values, 1)[0]
            
            if trend_slope > 0.01:
                trend_direction = "increasing"
            elif trend_slope < -0.01:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            insights.append({
                'category': 'Long-term Trends',
                'finding': f"Overall delay trend is {trend_direction} (slope: {trend_slope:.4f})",
                'recommendation': 'Monitor trend changes and implement interventions if delays are increasing'
            })
        
        # Print insights
        print("KEY INSIGHTS FOR DELAY PREDICTION:")
        print("-" * 50)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight['category']}:")
            print(f"   Finding: {insight['finding']}")
            print(f"   Recommendation: {insight['recommendation']}")
            print()
        
        # Model recommendations
        print("MACHINE LEARNING MODEL RECOMMENDATIONS:")
        print("-" * 50)
        print("1. Feature Engineering:")
        print("   - Include route-specific delay history")
        print("   - Add time-based features (hour, day, month, season)")
        print("   - Consider weather data integration")
        print("   - Include airport congestion metrics")
        
        print("\n2. Model Selection:")
        print("   - Use ensemble methods (Random Forest, XGBoost)")
        print("   - Consider time series models for temporal patterns")
        print("   - Implement separate models for different route types")
        
        print("\n3. Validation Strategy:")
        print("   - Use time-based splits to avoid data leakage")
        print("   - Cross-validate across different seasons")
        print("   - Test on recent data to ensure model relevance")
        
        self.results['predictive_insights'] = insights
        return insights
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🚀 SOUTHWEST AIRLINES COMPREHENSIVE DATA ANALYSIS")
        print("=" * 70)
        
        # Load data
        if not self.load_data():
            return False
        
        # Run all analyses
        print("\n" + "="*70)
        print("ANALYSIS PIPELINE EXECUTION")
        print("="*70)
        
        # 1. Route delay analysis
        self.analyze_route_delays()
        
        # 2. Delay cause analysis
        self.analyze_delay_causes()
        
        # 3. Seasonal pattern analysis
        self.analyze_seasonal_patterns()
        
        # 4. STL decomposition
        self.perform_stl_decomposition()
        
        # 5. Create visualizations
        self.create_visualizations()
        
        # 6. Generate predictive insights
        self.generate_predictive_insights()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"✅ All analyses completed successfully")
        print(f"✅ Visualizations saved to: {self.plots_dir}")
        print(f"✅ Results stored for further analysis")
        
        return True

def main():
    """
    Main function to run the comprehensive analysis
    """
    analyzer = SouthwestDataAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎯 ANALYSIS SUMMARY")
        print("=" * 50)
        print("This analysis provides comprehensive insights for:")
        print("1. Identifying high-risk routes and time periods")
        print("2. Understanding delay patterns and causes")
        print("3. Developing predictive models for flight delays")
        print("4. Optimizing operational decisions")
        print("\nNext steps: Use these insights to build ML models for delay prediction!")

if __name__ == "__main__":
    main()
