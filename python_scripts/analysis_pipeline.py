#!/usr/bin/env python3

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(n_days, random_seed, regions_list, figure_dir, data_dir):
    '''
    Main pipeline for weather data simulation and analysis.
    '''
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Simulate regions and variables
    regions = np.random.choice(regions_list, size=n_days)
    
    temperature = np.random.normal(loc=15, scale=5, size=n_days)
    temperature += (regions == 'Coastal') * 2
    temperature += (regions == 'Mountain') * -5
    
    precipitation = np.random.exponential(scale=3, size=n_days)
    precipitation += (regions == 'Mountain') * 1.5
    
    wind_speed = np.random.normal(loc=10, scale=2, size=n_days)
    wind_speed += (regions == 'Coastal') * 3
    
    # Build dataframe
    df = pd.DataFrame({
        'Region': regions,
        'Temperature_C': temperature,
        'Precipitation_mm': precipitation,
        'WindSpeed_kmh': wind_speed,
        'DayOfYear': np.arange(1, n_days + 1)
    })
    
    
    # Create and save figure
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Region', y='Temperature_C', data=df, 
                hue='Region', palette='coolwarm', legend=False)
    plt.title('Temperature Distribution by Region')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig(figure_dir + '/temperature_dists.png')
    
    # Calculate and save summary stats
    summary = df.describe()
    summary.to_csv(data_dir + '/weather_stats.csv')


if __name__ == "__main__":
    
    # Define hyperparameters
    N_DAYS = 365
    RANDOM_SEED = 42
    REGIONS_LIST = ['Coastal', 'Inland', 'Mountain']
    FIGURE_DIR = '/oak/stanford/groups/cyaolai/ElliannaAbrahams/sdss_cc/postdoc_onboarding/figs'
    DATA_DIR = '/oak/stanford/groups/cyaolai/ElliannaAbrahams/sdss_cc/postdoc_onboarding/data'
    
    # Run the pipeline
    main(n_days=N_DAYS, random_seed=RANDOM_SEED, regions_list=REGIONS_LIST,
         figure_dir=FIGURE_DIR, data_dir=DATA_DIR)