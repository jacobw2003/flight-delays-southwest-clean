#!/usr/bin/env python3
"""
COVID-19 Impact Analysis for Southwest Airlines Flight Data

This script analyzes the impact of COVID-19 on flight patterns and delays
to determine if 2020-2021 data should be excluded from ML modeling.

Analysis includes:
1. Flight volume trends by year/month
2. Delay patterns during COVID vs normal periods
3. Route changes during COVID
4. Statistical comparison of delay distributions
5. Recommendations for data inclusion/exclusion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load the cleaned dataset and prepare for analysis
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_file = project_root / "data" / "preprocessed_data" / "southwest_cleaned.csv"
    
    print("📂 Loading cleaned dataset...")
    df = pd.read_csv(input_file)
    
    # Convert DepDate to datetime if it's not already
    if df['DepDate'].dtype == 'object':
        df['DepDate'] = pd.to_datetime(df['DepDate'])
    
    print(f"Dataset loaded: {df.shape}")
    print(f"Date range: {df['DepDate'].min()} to {df['DepDate'].max()}")
    
    return df

def analyze_flight_volume_trends(df):
    """
    Analyze flight volume trends by year and month
    """
    print("\n📊 FLIGHT VOLUME ANALYSIS")
    print("=" * 50)
    
    # Yearly flight counts
    yearly_counts = df.groupby('Year').size()
    print("Yearly Flight Counts:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:,} flights")
    
    # Monthly flight counts
    monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='FlightCount')
    
    # Calculate percentage change from 2019 baseline
    baseline_2019 = monthly_counts[monthly_counts['Year'] == 2019]['FlightCount'].mean()
    print(f"\n2019 Monthly Average: {baseline_2019:,.0f} flights")
    
    covid_impact = {}
    for year in [2020, 2021]:
        year_avg = monthly_counts[monthly_counts['Year'] == year]['FlightCount'].mean()
        pct_change = ((year_avg - baseline_2019) / baseline_2019) * 100
        covid_impact[year] = pct_change
        print(f"{year} Monthly Average: {year_avg:,.0f} flights ({pct_change:+.1f}% vs 2019)")
    
    return yearly_counts, monthly_counts, covid_impact

def analyze_delay_patterns(df):
    """
    Analyze delay patterns during COVID vs normal periods
    """
    print("\n⏰ DELAY PATTERN ANALYSIS")
    print("=" * 50)
    
    # Define periods
    pre_covid = df[df['Year'].isin([2018, 2019])]
    covid_period = df[df['Year'].isin([2020, 2021])]
    post_covid = df[df['Year'].isin([2022, 2023])]
    
    periods = {
        'Pre-COVID (2018-2019)': pre_covid,
        'COVID Period (2020-2021)': covid_period,
        'Post-COVID (2022-2023)': post_covid
    }
    
    delay_stats = {}
    
    for period_name, period_df in periods.items():
        if len(period_df) > 0:
            dep_delays = period_df['DepDelayMinutes']
            arr_delays = period_df['ArrDelayMinutes']
            
            stats = {
                'flight_count': len(period_df),
                'dep_delay_mean': dep_delays.mean(),
                'dep_delay_median': dep_delays.median(),
                'dep_delay_std': dep_delays.std(),
                'arr_delay_mean': arr_delays.mean(),
                'arr_delay_median': arr_delays.median(),
                'arr_delay_std': arr_delays.std(),
                'on_time_pct': (dep_delays == 0).mean() * 100,
                'major_delay_pct': (dep_delays > 60).mean() * 100
            }
            
            delay_stats[period_name] = stats
            
            print(f"\n{period_name}:")
            print(f"  Flights: {stats['flight_count']:,}")
            print(f"  Avg Departure Delay: {stats['dep_delay_mean']:.1f} min")
            print(f"  Avg Arrival Delay: {stats['arr_delay_mean']:.1f} min")
            print(f"  On-Time Rate: {stats['on_time_pct']:.1f}%")
            print(f"  Major Delay Rate: {stats['major_delay_pct']:.1f}%")
    
    return delay_stats

def analyze_route_changes(df):
    """
    Analyze route changes during COVID period
    """
    print("\n🛫 ROUTE ANALYSIS")
    print("=" * 50)
    
    # Top routes by period
    pre_covid_routes = df[df['Year'].isin([2018, 2019])]['Route'].value_counts().head(10)
    covid_routes = df[df['Year'].isin([2020, 2021])]['Route'].value_counts().head(10)
    
    print("Top 10 Routes Pre-COVID (2018-2019):")
    for route, count in pre_covid_routes.items():
        print(f"  {route}: {count:,}")
    
    print("\nTop 10 Routes During COVID (2020-2021):")
    for route, count in covid_routes.items():
        print(f"  {route}: {count:,}")
    
    # Route stability analysis
    all_routes = set(df['Route'].unique())
    pre_covid_route_set = set(df[df['Year'].isin([2018, 2019])]['Route'].unique())
    covid_route_set = set(df[df['Year'].isin([2020, 2021])]['Route'].unique())
    
    routes_dropped = pre_covid_route_set - covid_route_set
    routes_added = covid_route_set - pre_covid_route_set
    
    print(f"\nRoute Changes:")
    print(f"  Total unique routes: {len(all_routes)}")
    print(f"  Routes dropped during COVID: {len(routes_dropped)}")
    print(f"  Routes added during COVID: {len(routes_added)}")
    
    return pre_covid_routes, covid_routes, routes_dropped, routes_added

def create_visualizations(df, yearly_counts, monthly_counts, delay_stats):
    """
    Create visualizations for COVID impact analysis
    """
    print("\n📈 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Create output directory for plots
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    plots_dir = project_root / "analysis" / "preprocessing" / "covid_analysis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Flight Volume Trends
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('COVID-19 Impact on Southwest Airlines Flight Operations', fontsize=16, fontweight='bold')
    
    # Yearly flight counts
    axes[0, 0].bar(yearly_counts.index, yearly_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[0, 0].set_title('Flight Volume by Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Flights')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(yearly_counts.values):
        axes[0, 0].text(yearly_counts.index[i], v + 50000, f'{v:,}', ha='center', va='bottom')
    
    # Monthly trends by year
    monthly_pivot = monthly_counts.pivot(index='Month', columns='Year', values='FlightCount')
    monthly_pivot.plot(ax=axes[0, 1], marker='o', linewidth=2)
    axes[0, 1].set_title('Monthly Flight Trends by Year')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Number of Flights')
    axes[0, 1].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Delay distribution comparison
    delay_data = []
    period_names = []
    for period_name, stats in delay_stats.items():
        delay_data.append(stats['dep_delay_mean'])
        period_names.append(period_name.replace(' (2018-2019)', '').replace(' (2020-2021)', '').replace(' (2022-2023)', ''))
    
    bars = axes[1, 0].bar(period_names, delay_data, color=['#2ca02c', '#d62728', '#9467bd'])
    axes[1, 0].set_title('Average Departure Delay by Period')
    axes[1, 0].set_ylabel('Average Delay (minutes)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, delay_data):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom')
    
    # On-time performance comparison
    on_time_data = [stats['on_time_pct'] for stats in delay_stats.values()]
    bars = axes[1, 1].bar(period_names, on_time_data, color=['#2ca02c', '#d62728', '#9467bd'])
    axes[1, 1].set_title('On-Time Performance by Period')
    axes[1, 1].set_ylabel('On-Time Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, on_time_data):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'covid_impact_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed delay distribution analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Delay Distribution Analysis by Period', fontsize=16, fontweight='bold')
    
    periods_data = {
        'Pre-COVID': df[df['Year'].isin([2018, 2019])]['DepDelayMinutes'],
        'COVID': df[df['Year'].isin([2020, 2021])]['DepDelayMinutes'],
        'Post-COVID': df[df['Year'].isin([2022, 2023])]['DepDelayMinutes']
    }
    
    for i, (period_name, delay_data) in enumerate(periods_data.items()):
        if len(delay_data) > 0:
            # Create histogram for delays 0-120 minutes (most common range)
            delay_subset = delay_data[(delay_data >= 0) & (delay_data <= 120)]
            axes[i].hist(delay_subset, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{period_name} Period\nDelay Distribution (0-120 min)')
            axes[i].set_xlabel('Delay (minutes)')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics
            mean_delay = delay_data.mean()
            median_delay = delay_data.median()
            axes[i].axvline(mean_delay, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_delay:.1f}')
            axes[i].axvline(median_delay, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_delay:.1f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'delay_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Plots saved to: {plots_dir}")
    
    return plots_dir

def statistical_analysis(df):
    """
    Perform statistical tests to compare delay patterns
    """
    print("\n📊 STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # Prepare data for statistical tests
    pre_covid_delays = df[df['Year'].isin([2018, 2019])]['DepDelayMinutes']
    covid_delays = df[df['Year'].isin([2020, 2021])]['DepDelayMinutes']
    post_covid_delays = df[df['Year'].isin([2022, 2023])]['DepDelayMinutes']
    
    from scipy import stats
    
    # Kolmogorov-Smirnov test for distribution differences
    ks_stat_pre_covid, ks_p_pre_covid = stats.ks_2samp(pre_covid_delays, covid_delays)
    ks_stat_post_covid, ks_p_post_covid = stats.ks_2samp(covid_delays, post_covid_delays)
    
    print("Kolmogorov-Smirnov Tests (distribution differences):")
    print(f"  Pre-COVID vs COVID: KS-statistic={ks_stat_pre_covid:.4f}, p-value={ks_p_pre_covid:.2e}")
    print(f"  COVID vs Post-COVID: KS-statistic={ks_stat_post_covid:.4f}, p-value={ks_p_post_covid:.2e}")
    
    # Mann-Whitney U test for median differences
    mw_stat_pre_covid, mw_p_pre_covid = stats.mannwhitneyu(pre_covid_delays, covid_delays, alternative='two-sided')
    mw_stat_post_covid, mw_p_post_covid = stats.mannwhitneyu(covid_delays, post_covid_delays, alternative='two-sided')
    
    print("\nMann-Whitney U Tests (median differences):")
    print(f"  Pre-COVID vs COVID: U-statistic={mw_stat_pre_covid:.0f}, p-value={mw_p_pre_covid:.2e}")
    print(f"  COVID vs Post-COVID: U-statistic={mw_stat_post_covid:.0f}, p-value={mw_p_post_covid:.2e}")
    
    # Effect size (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(), group2.std()
        s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / s_pooled
    
    effect_size_pre_covid = cohens_d(pre_covid_delays, covid_delays)
    effect_size_post_covid = cohens_d(covid_delays, post_covid_delays)
    
    print(f"\nEffect Sizes (Cohen's d):")
    print(f"  Pre-COVID vs COVID: {effect_size_pre_covid:.3f}")
    print(f"  COVID vs Post-COVID: {effect_size_post_covid:.3f}")
    
    return {
        'ks_tests': [(ks_stat_pre_covid, ks_p_pre_covid), (ks_stat_post_covid, ks_p_post_covid)],
        'mw_tests': [(mw_stat_pre_covid, mw_p_pre_covid), (mw_stat_post_covid, mw_p_post_covid)],
        'effect_sizes': [effect_size_pre_covid, effect_size_post_covid]
    }

def provide_recommendations(df, yearly_counts, delay_stats, statistical_results):
    """
    Provide recommendations based on the analysis
    """
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    
    # Calculate key metrics
    covid_years = [2020, 2021]
    normal_years = [2018, 2019, 2022, 2023]
    
    covid_flights = df[df['Year'].isin(covid_years)]
    normal_flights = df[df['Year'].isin(normal_years)]
    
    covid_avg_delay = covid_flights['DepDelayMinutes'].mean()
    normal_avg_delay = normal_flights['DepDelayMinutes'].mean()
    
    covid_on_time = (covid_flights['DepDelayMinutes'] == 0).mean() * 100
    normal_on_time = (normal_flights['DepDelayMinutes'] == 0).mean() * 100
    
    # Statistical significance
    ks_p_value = statistical_results['ks_tests'][0][1]  # Pre-COVID vs COVID
    effect_size = statistical_results['effect_sizes'][0]
    
    print("ANALYSIS SUMMARY:")
    print(f"  COVID Period (2020-2021): {len(covid_flights):,} flights")
    print(f"  Normal Period (2018-2019, 2022-2023): {len(normal_flights):,} flights")
    print(f"  COVID avg delay: {covid_avg_delay:.1f} min vs Normal: {normal_avg_delay:.1f} min")
    print(f"  COVID on-time rate: {covid_on_time:.1f}% vs Normal: {normal_on_time:.1f}%")
    print(f"  Statistical significance: p = {ks_p_value:.2e}")
    print(f"  Effect size: {effect_size:.3f}")
    
    print("\nRECOMMENDATIONS:")
    
    if ks_p_value < 0.001 and abs(effect_size) > 0.2:
        print("✅ STRONG RECOMMENDATION: EXCLUDE COVID DATA")
        print("   Reasons:")
        print("   - Statistically significant difference in delay patterns")
        print("   - Large effect size indicates substantial difference")
        print("   - COVID period represents abnormal operational conditions")
        print("   - Including COVID data may bias ML models")
    elif ks_p_value < 0.05:
        print("⚠️  MODERATE RECOMMENDATION: CONSIDER EXCLUDING COVID DATA")
        print("   Reasons:")
        print("   - Statistically significant difference detected")
        print("   - COVID period may not represent normal operations")
        print("   - Consider excluding for more robust ML models")
    else:
        print("ℹ️  WEAK RECOMMENDATION: COVID DATA MAY BE INCLUDED")
        print("   Reasons:")
        print("   - No strong statistical evidence of different patterns")
        print("   - COVID data provides additional training examples")
        print("   - Consider including but monitor model performance")
    
    print(f"\nSUGGESTED DATASET:")
    if ks_p_value < 0.001 and abs(effect_size) > 0.2:
        recommended_years = normal_years
        print(f"   Include years: {recommended_years}")
        print(f"   Exclude years: {covid_years}")
        recommended_df = df[df['Year'].isin(recommended_years)]
        print(f"   Recommended dataset size: {len(recommended_df):,} flights")
    else:
        print(f"   Include all years: {sorted(df['Year'].unique())}")
        print(f"   Full dataset size: {len(df):,} flights")
        recommended_df = df
    
    return recommended_df

def main():
    """
    Main analysis function
    """
    print("🦠 COVID-19 IMPACT ANALYSIS FOR SOUTHWEST AIRLINES")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Perform analyses
    yearly_counts, monthly_counts, covid_impact = analyze_flight_volume_trends(df)
    delay_stats = analyze_delay_patterns(df)
    pre_covid_routes, covid_routes, routes_dropped, routes_added = analyze_route_changes(df)
    
    # Create visualizations
    plots_dir = create_visualizations(df, yearly_counts, monthly_counts, delay_stats)
    
    # Statistical analysis
    statistical_results = statistical_analysis(df)
    
    # Provide recommendations
    recommended_df = provide_recommendations(df, yearly_counts, delay_stats, statistical_results)
    
    # Save recommended dataset if different from original
    if len(recommended_df) != len(df):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_file = project_root / "data" / "preprocessed_data" / "southwest_covid_filtered.csv"
        recommended_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved recommended dataset to: {output_file}")
        print(f"   Original: {len(df):,} flights")
        print(f"   Recommended: {len(recommended_df):,} flights")
        print(f"   Reduction: {len(df) - len(recommended_df):,} flights ({(len(df) - len(recommended_df))/len(df)*100:.1f}%)")
    
    print(f"\n✅ Analysis complete! Check plots in: {plots_dir}")

if __name__ == "__main__":
    main()
