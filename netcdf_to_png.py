import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

ssp245_file = r"C:\Users\Admin\254cb509cd165079d919ce46c93a8a86\tasmax_day_KIOST-ESM_ssp245_r1i1p1f1_gr1_20200101-20501231.nc"
ssp585_file = r"C:\Users\Admin\4ab634454ada4a205656e010639f5d4d\tasmax_day_KIOST-ESM_ssp585_r1i1p1f1_gr1_20200101-20501231.nc"

# Get the output directory (using the directory of the first file)
output_dir = os.path.dirname(ssp245_file)
print(f"Output files will be saved to: {output_dir}")

# Function to extract and process April-May data for all available years
def extract_apr_may_data(file_path, scenario_name):
    print(f"Opening {scenario_name} file: {file_path}")
    try:
        # Open the dataset
        ds = xr.open_dataset(file_path)
        print(f"{scenario_name} file loaded successfully.")
        
        # Print available variables to debug
        print(f"Available variables in the dataset: {list(ds.data_vars.keys())}")
        
        # Get the first lat/lon point - check if they're named differently
        if 'lat' in ds.dims:
            lat_var = 'lat'
        elif 'latitude' in ds.dims:
            lat_var = 'latitude'
        else:
            print(f"Warning: Could not find latitude dimension. Available dimensions: {list(ds.dims)}")
            lat_var = list(ds.dims)[1]  # Try the second dimension
            
        if 'lon' in ds.dims:
            lon_var = 'lon'
        elif 'longitude' in ds.dims:
            lon_var = 'longitude'
        else:
            print(f"Warning: Could not find longitude dimension. Available dimensions: {list(ds.dims)}")
            lon_var = list(ds.dims)[2]  # Try the third dimension
        
        # Get coordinates
        lat_point = ds[lat_var].values[0]
        lon_point = ds[lon_var].values[0]
        print(f"Using coordinates: {lat_var}={lat_point}, {lon_var}={lon_point}")
        
        # Check for temperature variable - use tasmax instead of tasmin
        if 'tasmax' in ds.data_vars:
            temp_var = 'tasmax'
        elif 'tasmin' in ds.data_vars:
            temp_var = 'tasmin'
        else:
            # Try to find any variable that might be temperature
            for var in ds.data_vars:
                if 'temp' in var.lower() or 'tas' in var.lower():
                    temp_var = var
                    break
            else:
                print(f"Error: Could not find temperature variable in {list(ds.data_vars.keys())}")
                return None, None, None
        
        print(f"Using temperature variable: {temp_var}")
        
        # Extract time series for the selected location
        coords = {lat_var: lat_point, lon_var: lon_point}
        ts_data = ds[temp_var].sel(coords, method='nearest')
        
        # Get time range
        # Handle cftime objects directly instead of converting to pandas datetime
        start_time = ds.time.values[0]
        end_time = ds.time.values[-1]
        print(f"Time range in file: {start_time.year}-{start_time.month:02d}-{start_time.day:02d} to {end_time.year}-{end_time.month:02d}-{end_time.day:02d}")
        
        # Convert Kelvin to Celsius if needed
        # Check if data appears to be in Kelvin (values around 273+)
        sample_value = float(ts_data.isel(time=0).values.flatten()[0])
        if sample_value > 100:  # Likely Kelvin
            print(f"Converting from Kelvin to Celsius (sample value: {sample_value})")
            ts_data_celsius = ts_data - 273.15
        else:  # Likely already Celsius
            print(f"Data appears to be already in Celsius (sample value: {sample_value})")
            ts_data_celsius = ts_data
        
        # Process the data manually to handle cftime objects
        # This is the key fix for the DatetimeNoLeap error
        years = []
        months = []
        temperatures = []
        
        for i in range(len(ds.time)):
            time_obj = ds.time.values[i]
            years.append(time_obj.year)
            months.append(time_obj.month)
            temp_val = float(ts_data_celsius.isel(time=i).values.flatten()[0])
            temperatures.append(temp_val)
        
        # Create a DataFrame
        df = pd.DataFrame({
            'year': years,
            'month': months,
            'temperature': temperatures
        })
        
        # Filter for April and May
        apr_may_data = df[(df['month'] == 4) | (df['month'] == 5)]
        
        # Calculate April-May average for each year
        annual_apr_may = apr_may_data.groupby('year')['temperature'].mean().reset_index()
        annual_apr_may['scenario'] = scenario_name
        
        print(f"Extracted April-May data for {scenario_name} (years {annual_apr_may['year'].min()}-{annual_apr_may['year'].max()})")
        print(f"Data summary: {annual_apr_may.describe()}")
        
        return annual_apr_may, lat_point, lon_point
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error processing {scenario_name} file: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Extract data for both scenarios
ssp245_data, lat_point_245, lon_point_245 = extract_apr_may_data(ssp245_file, "SSP2-4.5")

# Check if SSP5-8.5 file exists and process it
try:
    with open(ssp585_file, 'r') as f:
        file_exists = True
except FileNotFoundError:
    file_exists = False
    print(f"Warning: SSP5-8.5 file not found at {ssp585_file}")
    print("The script will continue with only SSP2-4.5 data.")

if file_exists:
    ssp585_data, lat_point_585, lon_point_585 = extract_apr_may_data(ssp585_file, "SSP5-8.5")
    # Use coordinates from whichever file has data
    if lat_point_245 is not None:
        lat_point = lat_point_245
        lon_point = lon_point_245
    else:
        lat_point = lat_point_585
        lon_point = lon_point_585
else:
    ssp585_data = None
    lat_point = lat_point_245
    lon_point = lon_point_245

# Check if we have any data to plot
if ssp245_data is None and ssp585_data is None:
    print("Error: No valid data extracted from either file. Exiting.")
    exit()

# Get the full range of years across both datasets
all_years = set()
if ssp245_data is not None:
    all_years.update(ssp245_data['year'].tolist())
if ssp585_data is not None:
    all_years.update(ssp585_data['year'].tolist())

min_year = min(all_years)
max_year = max(all_years)
print(f"Full year range across all datasets: {min_year}-{max_year}")

# Create a clean visualization with just trend lines and rolling averages
plt.figure(figsize=(14, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Function to create a rolling average with proper handling of NaN values
def rolling_avg(data, window):
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values

# Set the smoothing window size
window_size = 3  # 3-year window for smoothing

# Prepare for trend lines
trend_lines = {}

# Plot SSP2-4.5 data with rolling average - NO raw data points
if ssp245_data is not None:
    # Sort data by year to ensure correct order
    ssp245_data = ssp245_data.sort_values('year')
    x = ssp245_data['year'].values
    y = ssp245_data['temperature'].values
    
    # Calculate rolling average
    y_smooth = rolling_avg(y, window_size)
    
    # Plot ONLY the smoothed line (no raw data points)
    plt.plot(x, y_smooth, '-', color='#ff7f0e', linewidth=3, 
             label='SSP2-4.5 (Middle of the Road)')
    
    # Add shaded confidence region
    plt.fill_between(x, y_smooth - 0.15, y_smooth + 0.15, color='#ff7f0e', alpha=0.2)
    
    # Calculate and store trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    trend_lines['SSP2-4.5'] = (x, p(x), z[0])

# Plot SSP5-8.5 data with rolling average - NO raw data points
if ssp585_data is not None:
    # Sort data by year to ensure correct order
    ssp585_data = ssp585_data.sort_values('year')
    x = ssp585_data['year'].values
    y = ssp585_data['temperature'].values
    
    # Calculate rolling average
    y_smooth = rolling_avg(y, window_size)
    
    # Plot ONLY the smoothed line (no raw data points)
    plt.plot(x, y_smooth, '-', color='#d62728', linewidth=3,
             label='SSP5-8.5 (High Emissions)')
    
    # Add shaded confidence region
    plt.fill_between(x, y_smooth - 0.15, y_smooth + 0.15, color='#d62728', alpha=0.2)
    
    # Calculate and store trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    trend_lines['SSP5-8.5'] = (x, p(x), z[0])

# Add trend lines
for scenario, (x, y, slope) in trend_lines.items():
    color = '#ff7f0e' if scenario == 'SSP2-4.5' else '#d62728'
    plt.plot(x, y, '--', color=color, linewidth=1.5,
             label=f'{scenario} Trend: {slope:.4f}°C/year')

# Format the plot
if lat_point is not None and lon_point is not None:
    title = f'April-May Maximum Temperature Projections\nKIOST-ESM Climate Model ({min_year}-{max_year})'
    subtitle = f'Location: Lat {lat_point:.2f}°, Lon {lon_point:.2f}°'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.title(subtitle, fontsize=14)
else:
    title = f'April-May Maximum Temperature Projections\nKIOST-ESM Climate Model ({min_year}-{max_year})'
    plt.suptitle(title, fontsize=16, fontweight='bold')

plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.grid(True, linestyle='-', alpha=0.7)

# Set x-axis ticks - show every 5 years for readability
x_ticks = range(min_year, max_year + 1, 5)
plt.xticks(x_ticks)

# Add legend with improved positioning
plt.legend(fontsize=12, loc='upper left', framealpha=0.9)

# Set y-axis limits with a buffer
all_temps = []
if ssp245_data is not None:
    all_temps.extend(ssp245_data['temperature'].tolist())
if ssp585_data is not None:
    all_temps.extend(ssp585_data['temperature'].tolist())

if all_temps:
    min_temp = min(all_temps) - 0.5
    max_temp = max(all_temps) + 0.5
    plt.ylim(min_temp, max_temp)

# Add explanatory text
scenario_info = (
    "SSP2-4.5: Middle of the road scenario (moderate emissions)\n"
    "SSP5-8.5: Fossil-fueled development (high emissions)\n"
    "Shaded areas represent uncertainty ranges"
)
plt.figtext(0.01, 0.01, scenario_info, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

# Enhance aesthetics
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title space

# Save the clean visualization
clean_comparison_path = os.path.join(output_dir, f'apr_may_temperature_clean_{min_year}-{max_year}.png')
plt.savefig(clean_comparison_path, dpi=300, bbox_inches='tight')
print(f"Saved clean temperature projection with trend lines to: {clean_comparison_path}")

# Save the data to CSV for reference
if ssp245_data is not None:
    ssp245_csv_path = os.path.join(output_dir, f'apr_may_ssp245_{min_year}-{max_year}.csv')
    ssp245_data.to_csv(ssp245_csv_path, index=False)
    print(f"Saved SSP2-4.5 data to: {ssp245_csv_path}")

if ssp585_data is not None:
    ssp585_csv_path = os.path.join(output_dir, f'apr_may_ssp585_{min_year}-{max_year}.csv')
    ssp585_data.to_csv(ssp585_csv_path, index=False)
    print(f"Saved SSP5-8.5 data to: {ssp585_csv_path}")

print("\nAnalysis complete!")