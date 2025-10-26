import os
import tempfile
import xarray as xr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cdsapi
from typing import Dict, List, Tuple, Optional
from config import Config


class WineDemoService:
    """
    Climate data analysis service that downloads data from CDS API 
    and generates temperature projection visualizations.
    """
    
    def __init__(self):
        """Initialize the service with CDS API client."""
        try:
            config = Config()
            self.cds_client = cdsapi.Client(
                url=config.get_cds_url(),
                key=config.get_cds_api_key()
            )
        except Exception as e:
            print(f"Warning: Could not initialize CDS client: {e}")
            self.cds_client = None
    
    def download_cds_data(self, lat: float, lon: float, scenarios: List[str] = None) -> Dict[str, str]:
        """
        Download climate data from CDS API for specified coordinates and scenarios.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate 
            scenarios: List of SSP scenarios (e.g., ['ssp2_4_5', 'ssp5_8_5'])
            
        Returns:
            Dictionary mapping scenario names to downloaded file paths
        """
        if self.cds_client is None:
            raise Exception("CDS client not initialized. Check your CDS API credentials.")
        
        if scenarios is None:
            scenarios = ['ssp2_4_5', 'ssp5_8_5']
        
        downloaded_files = {}
        
        # Create temporary directory for downloads
        temp_dir = tempfile.mkdtemp()
        
        for scenario in scenarios:
            try:
                # Define the area around the coordinates (small bounding box)
                # CDS expects [North, West, South, East] format
                area = [lat + 0.1, lon - 0.1, lat - 0.1, lon + 0.1]
                
                # CDS API request parameters based on your specification
                request_params = {
                    "temporal_resolution": "daily",
                    "experiment": scenario,
                    "variable": "daily_maximum_near_surface_air_temperature",
                    "model": "kiost_esm",
                    "month": [
                        "01", "02", "03", "04", "05", "06",
                        "07", "08", "09", "10", "11", "12"
                    ],
                    "day": [
                        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
                    ],
                    "year": [
                        "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
                        "2031", "2032", "2033", "2034", "2035", "2036", "2037", "2038", "2039", "2040",
                        "2041", "2042", "2043", "2044", "2045", "2046", "2047", "2048", "2049", "2050"
                    ],
                    "area": area
                }
                
                # Download file
                output_file = os.path.join(temp_dir, f'climate_data_{scenario}_{lat}_{lon}.nc')
                
                print(f"Downloading {scenario} data for coordinates ({lat}, {lon})...")
                self.cds_client.retrieve("projections-cmip6", request_params, output_file)
                
                downloaded_files[scenario] = output_file
                print(f"Successfully downloaded {scenario} data to: {output_file}")
                
            except Exception as e:
                print(f"Error downloading {scenario} data: {e}")
                continue
        
        return downloaded_files
    
    def extract_apr_may_data(self, file_path: str, scenario_name: str) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float]]:
        """
        Extract April-May temperature data from NetCDF file.
        Based on the logic from netcdf_to_png.py
        """
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
            
            # Check for temperature variable - use tasmax
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
            
            # Convert Kelvin to Celsius if needed
            sample_value = float(ts_data.isel(time=0).values.flatten()[0])
            if sample_value > 100:  # Likely Kelvin
                print(f"Converting from Kelvin to Celsius (sample value: {sample_value})")
                ts_data_celsius = ts_data - 273.15
            else:  # Likely already Celsius
                print(f"Data appears to be already in Celsius (sample value: {sample_value})")
                ts_data_celsius = ts_data
            
            # Process the data manually to handle cftime objects
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
            
            return annual_apr_may, lat_point, lon_point
            
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None, None, None
        except Exception as e:
            print(f"Error processing {scenario_name} file: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def create_temperature_visualization(self, ssp245_data: pd.DataFrame, ssp585_data: pd.DataFrame, 
                                       lat_point: float, lon_point: float, output_dir: str) -> str:
        """
        Create temperature projection visualization PNG.
        Based on the visualization logic from netcdf_to_png.py
        """
        # Get the full range of years across both datasets
        all_years = set()
        if ssp245_data is not None:
            all_years.update(ssp245_data['year'].tolist())
        if ssp585_data is not None:
            all_years.update(ssp585_data['year'].tolist())

        min_year = min(all_years)
        max_year = max(all_years)
        print(f"Full year range across all datasets: {min_year}-{max_year}")

        # Create visualization
        plt.figure(figsize=(14, 8))
        plt.style.use('seaborn-v0_8-whitegrid')

        # Function to create a rolling average
        def rolling_avg(data, window):
            if len(data) < window:
                return data
            return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values

        window_size = 3  # 3-year window for smoothing
        trend_lines = {}

        # Plot SSP2-4.5 data with rolling average
        if ssp245_data is not None:
            ssp245_data = ssp245_data.sort_values('year')
            x = ssp245_data['year'].values
            y = ssp245_data['temperature'].values
            
            y_smooth = rolling_avg(y, window_size)
            
            plt.plot(x, y_smooth, '-', color='#ff7f0e', linewidth=3, 
                     label='SSP2-4.5 (Middle of the Road)')
            plt.fill_between(x, y_smooth - 0.15, y_smooth + 0.15, color='#ff7f0e', alpha=0.2)
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_lines['SSP2-4.5'] = (x, p(x), z[0])

        # Plot SSP5-8.5 data with rolling average
        if ssp585_data is not None:
            ssp585_data = ssp585_data.sort_values('year')
            x = ssp585_data['year'].values
            y = ssp585_data['temperature'].values
            
            y_smooth = rolling_avg(y, window_size)
            
            plt.plot(x, y_smooth, '-', color='#d62728', linewidth=3,
                     label='SSP5-8.5 (High Emissions)')
            plt.fill_between(x, y_smooth - 0.15, y_smooth + 0.15, color='#d62728', alpha=0.2)
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_lines['SSP5-8.5'] = (x, p(x), z[0])

        # Add trend lines
        for scenario, (x, y, slope) in trend_lines.items():
            color = '#ff7f0e' if scenario == 'SSP2-4.5' else '#d62728'
            plt.plot(x, y, '--', color=color, linewidth=1.5,
                     label=f'{scenario} Trend: {slope:.4f}°C/year')

        # Format the plot
        title = f'April-May Maximum Temperature Projections\nKIOST-ESM Climate Model ({min_year}-{max_year})'
        subtitle = f'Location: Lat {lat_point:.2f}°, Lon {lon_point:.2f}°'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.title(subtitle, fontsize=14)

        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Temperature (°C)', fontsize=14)
        plt.grid(True, linestyle='-', alpha=0.7)

        x_ticks = range(min_year, max_year + 1, 5)
        plt.xticks(x_ticks)
        plt.legend(fontsize=12, loc='upper left', framealpha=0.9)

        # Set y-axis limits
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the visualization
        output_path = os.path.join(output_dir, f'apr_may_temperature_{min_year}-{max_year}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Clean up memory
        
        print(f"Saved temperature projection visualization to: {output_path}")
        return output_path
    
    def generate_landcover_report(self, lat: float, lon: float, id: str, firstname: str) -> Dict:
        """
        Main service method that orchestrates the climate data analysis.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            id: Request identifier
            firstname: User first name
            
        Returns:
            Dictionary containing analysis results and file paths
        """
        try:
            print(f"Starting climate analysis for coordinates ({lat}, {lon})")
            
            # Step 1: Download CDS data
            scenarios = ['ssp2_4_5', 'ssp5_8_5']
            downloaded_files = self.download_cds_data(lat, lon, scenarios)
            
            if not downloaded_files:
                raise Exception("No climate data files were successfully downloaded")
            
            # Step 2: Extract April-May data from downloaded files
            ssp245_data = None
            ssp585_data = None
            lat_point = None
            lon_point = None
            
            if 'ssp2_4_5' in downloaded_files:
                ssp245_data, lat_point, lon_point = self.extract_apr_may_data(
                    downloaded_files['ssp2_4_5'], "SSP2-4.5"
                )
            
            if 'ssp5_8_5' in downloaded_files:
                ssp585_data, lat_point_585, lon_point_585 = self.extract_apr_may_data(
                    downloaded_files['ssp5_8_5'], "SSP5-8.5"
                )
                # Use coordinates from whichever file has data
                if lat_point is None:
                    lat_point = lat_point_585
                    lon_point = lon_point_585
            
            if ssp245_data is None and ssp585_data is None:
                raise Exception("Failed to extract temperature data from any scenario files")
            
            # Step 3: Create temperature visualization
            config = Config()
            output_dir = config.get_string('IMAGES_FOLDER')
            os.makedirs(output_dir, exist_ok=True)
            
            visualization_path = self.create_temperature_visualization(
                ssp245_data, ssp585_data, lat_point, lon_point, output_dir
            )
            
            # Prepare results
            results = {
                'success': True,
                'id': id,
                'firstname': firstname,
                'coordinates': {
                    'requested_lat': lat,
                    'requested_lon': lon,
                    'actual_lat': lat_point,
                    'actual_lon': lon_point
                },
                'visualization_path': visualization_path,
                'data_files': downloaded_files,
                'scenarios_processed': []
            }
            
            if ssp245_data is not None:
                results['scenarios_processed'].append({
                    'scenario': 'SSP2-4.5',
                    'years': f"{ssp245_data['year'].min()}-{ssp245_data['year'].max()}",
                    'data_points': len(ssp245_data)
                })
            
            if ssp585_data is not None:
                results['scenarios_processed'].append({
                    'scenario': 'SSP5-8.5', 
                    'years': f"{ssp585_data['year'].min()}-{ssp585_data['year'].max()}",
                    'data_points': len(ssp585_data)
                })
            
            print("Climate analysis completed successfully!")
            return results
            
        except Exception as e:
            print(f"Error in climate analysis: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'id': id,
                'firstname': firstname,
                'coordinates': {
                    'requested_lat': lat,
                    'requested_lon': lon
                }
            }


# Create a singleton instance for the service
_service_instance = None

def get_service():
    """Get or create the wine demo service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = WineDemoService()
    return _service_instance

# Legacy function for backward compatibility with existing Azure Function
def generate_landcover_report(lat: float, lon: float, id: str, firstname: str) -> Dict:
    """Legacy function that calls the new service implementation."""
    service = get_service()
    return service.generate_landcover_report(lat, lon, id, firstname)
