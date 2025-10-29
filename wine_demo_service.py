import os
import tempfile
import zipfile
import logging
import cdsapi
import xarray as xr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class ClimateAnalysisService:
    """Service for downloading and analyzing climate data from CDS API."""
    
    def __init__(self, cds_url: str, cds_key: str):
        """
        Initialize the climate analysis service.
        
        Args:
            cds_url: CDS API URL
            cds_key: CDS API key
        """
        self.cds_url = cds_url
        self.cds_key = cds_key
        self.client = cdsapi.Client(url=cds_url, key=cds_key)
        
    def download_climate_data(self, lat: float, lon: float, temp_dir: str) -> Dict[str, str]:
        """
        Download climate data for both SSP2-4.5 and SSP5-8.5 scenarios.
        
        Args:
            lat: Latitude
            lon: Longitude
            temp_dir: Temporary directory for downloads
            
        Returns:
            Dictionary with paths to downloaded zip files for each scenario
        """
        logging.info(f"Downloading climate data for coordinates: lat={lat}, lon={lon}")
        
        # Round coordinates to 2 decimal places (CDS API requirement)
        lat = round(lat, 2)
        lon = round(lon, 2)
        
        # Create bounding box around the point (rounded to 2 decimals)
        # Note: CDS API requires a larger area (at least ~2-3 degrees) for grid resolution
        area = [
            round(lat + 1.5, 2),  # North
            round(lon - 1.5, 2),  # West
            round(lat - 1.5, 2),  # South
            round(lon + 1.5, 2)   # East
        ]
        
        logging.info(f"Rounded coordinates and area: lat={lat}, lon={lon}, area={area}")
        
        # Base request parameters
        base_params = {
            "temporal_resolution": "daily",
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
        
        downloaded_files = {}
        scenarios = {
            "ssp2_4_5": "ssp245",
            "ssp5_8_5": "ssp585"
        }
        
        for experiment, scenario_code in scenarios.items():
            logging.info(f"Downloading data for scenario: {experiment}")
            
            # Create request parameters for this scenario
            request_params = base_params.copy()
            request_params["experiment"] = experiment
            
            # Download file path
            download_path = os.path.join(temp_dir, f"{scenario_code}_data.zip")
            
            try:
                # Make the API request
                self.client.retrieve(
                    'projections-cmip6',
                    request_params,
                    download_path
                )
                
                logging.info(f"Successfully downloaded {experiment} data to {download_path}")
                downloaded_files[scenario_code] = download_path
                
            except Exception as e:
                logging.error(f"Failed to download {experiment} data: {e}")
                raise Exception(f"CDS API download failed for {experiment}: {str(e)}")
        
        return downloaded_files
    
    def extract_netcdf_files(self, zip_files: Dict[str, str], extract_dir: str) -> Dict[str, str]:
        """
        Extract NetCDF files from zip archives.
        
        Args:
            zip_files: Dictionary mapping scenario codes to zip file paths
            extract_dir: Directory to extract files to
            
        Returns:
            Dictionary mapping scenario codes to extracted .nc file paths
        """
        logging.info("Extracting NetCDF files from zip archives")
        
        netcdf_files = {}
        
        for scenario_code, zip_path in zip_files.items():
            logging.info(f"Extracting {scenario_code} zip file: {zip_path}")
            
            # Create scenario-specific extraction directory
            scenario_dir = os.path.join(extract_dir, scenario_code)
            os.makedirs(scenario_dir, exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(scenario_dir)
                
                # Find the .nc file in the extracted directory
                nc_files = [f for f in os.listdir(scenario_dir) if f.endswith('.nc')]
                
                if not nc_files:
                    raise Exception(f"No .nc file found in extracted {scenario_code} zip")
                
                if len(nc_files) > 1:
                    logging.warning(f"Multiple .nc files found for {scenario_code}, using first one")
                
                nc_file_path = os.path.join(scenario_dir, nc_files[0])
                netcdf_files[scenario_code] = nc_file_path
                
                logging.info(f"Extracted {scenario_code} NetCDF file: {nc_file_path}")
                
            except Exception as e:
                logging.error(f"Failed to extract {scenario_code} zip file: {e}")
                raise Exception(f"Zip extraction failed for {scenario_code}: {str(e)}")
        
        return netcdf_files
    
    def extract_apr_may_data(self, file_path: str, scenario_name: str) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float]]:
        """
        Extract April-May temperature data from a NetCDF file.
        
        Args:
            file_path: Path to the NetCDF file
            scenario_name: Name of the scenario for logging
            
        Returns:
            Tuple of (DataFrame with April-May averages, latitude, longitude)
        """
        logging.info(f"Processing {scenario_name} file: {file_path}")
        
        try:
            # Open the dataset
            ds = xr.open_dataset(file_path)
            logging.info(f"{scenario_name} file loaded successfully")
            
            # Identify lat/lon variable names
            if 'lat' in ds.dims:
                lat_var = 'lat'
            elif 'latitude' in ds.dims:
                lat_var = 'latitude'
            else:
                lat_var = list(ds.dims)[1]
                
            if 'lon' in ds.dims:
                lon_var = 'lon'
            elif 'longitude' in ds.dims:
                lon_var = 'longitude'
            else:
                lon_var = list(ds.dims)[2]
            
            # Get coordinates
            lat_point = float(ds[lat_var].values[0])
            lon_point = float(ds[lon_var].values[0])
            logging.info(f"Using coordinates: {lat_var}={lat_point}, {lon_var}={lon_point}")
            
            # Identify temperature variable
            if 'tasmax' in ds.data_vars:
                temp_var = 'tasmax'
            elif 'tasmin' in ds.data_vars:
                temp_var = 'tasmin'
            else:
                for var in ds.data_vars:
                    if 'temp' in var.lower() or 'tas' in var.lower():
                        temp_var = var
                        break
                else:
                    raise Exception(f"Could not find temperature variable in {list(ds.data_vars.keys())}")
            
            logging.info(f"Using temperature variable: {temp_var}")
            
            # Extract time series for the selected location
            coords = {lat_var: lat_point, lon_var: lon_point}
            ts_data = ds[temp_var].sel(coords, method='nearest')
            
            # Convert Kelvin to Celsius if needed
            sample_value = float(ts_data.isel(time=0).values.flatten()[0])
            if sample_value > 100:  # Likely Kelvin
                logging.info(f"Converting from Kelvin to Celsius (sample value: {sample_value})")
                ts_data_celsius = ts_data - 273.15
            else:
                logging.info(f"Data appears to be in Celsius (sample value: {sample_value})")
                ts_data_celsius = ts_data
            
            # Process data manually to handle cftime objects
            years = []
            months = []
            temperatures = []
            
            for i in range(len(ds.time)):
                time_obj = ds.time.values[i]
                years.append(time_obj.year)
                months.append(time_obj.month)
                temp_val = float(ts_data_celsius.isel(time=i).values.flatten()[0])
                temperatures.append(temp_val)
            
            # Create DataFrame
            df = pd.DataFrame({
                'year': years,
                'month': months,
                'temperature': temperatures
            })
            
            # Filter for years 2040-2050 only
            df = df[(df['year'] >= 2040) & (df['year'] <= 2050)]
            logging.info(f"Filtered data to years 2040-2050. Available years: {sorted(df['year'].unique())}")
            
            # Filter for April and May
            apr_may_data = df[(df['month'] == 4) | (df['month'] == 5)]
            
            # Calculate April-May average for each year
            annual_apr_may = apr_may_data.groupby('year')['temperature'].mean().reset_index()
            annual_apr_may['scenario'] = scenario_name
            
            logging.info(f"Extracted April-May data for {scenario_name} (years {annual_apr_may['year'].min()}-{annual_apr_may['year'].max()})")
            
            ds.close()
            
            return annual_apr_may, lat_point, lon_point
            
        except Exception as e:
            logging.error(f"Error processing {scenario_name} file: {e}")
            raise Exception(f"NetCDF processing failed for {scenario_name}: {str(e)}")
    
    def create_comparison_visualization(self, ssp245_data: pd.DataFrame, ssp585_data: pd.DataFrame, 
                                       lat_point: float, lon_point: float, output_path: str) -> str:
        """
        Create a comparison visualization of both climate scenarios.
        
        Args:
            ssp245_data: DataFrame with SSP2-4.5 data
            ssp585_data: DataFrame with SSP5-8.5 data
            lat_point: Latitude
            lon_point: Longitude
            output_path: Path to save the output PNG
            
        Returns:
            Path to the saved visualization
        """
        logging.info("Creating comparison visualization")
        
        # Set fixed year range to 2040-2050
        min_year = 2040
        max_year = 2050
        
        # Create figure
        plt.figure(figsize=(14, 8))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Function for rolling average
        def rolling_avg(data, window):
            if len(data) < window:
                return data
            return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
        
        window_size = 3
        trend_lines = {}
        
        # Plot SSP2-4.5 data
        ssp245_data = ssp245_data.sort_values('year')
        x_245 = ssp245_data['year'].values
        y_245 = ssp245_data['temperature'].values
        y_smooth_245 = rolling_avg(y_245, window_size)
        
        plt.plot(x_245, y_smooth_245, '-', color='#ff7f0e', linewidth=3, 
                 label='SSP2-4.5 (Moderate Emissions Forecast)')
        plt.fill_between(x_245, y_smooth_245 - 0.15, y_smooth_245 + 0.15, color='#ff7f0e', alpha=0.2)
        
        z_245 = np.polyfit(x_245, y_245, 1)
        p_245 = np.poly1d(z_245)
        trend_lines['SSP2-4.5'] = (x_245, p_245(x_245), z_245[0])
        
        # Plot SSP5-8.5 data
        ssp585_data = ssp585_data.sort_values('year')
        x_585 = ssp585_data['year'].values
        y_585 = ssp585_data['temperature'].values
        y_smooth_585 = rolling_avg(y_585, window_size)
        
        plt.plot(x_585, y_smooth_585, '-', color='#d62728', linewidth=3,
                 label='SSP5-8.5 (High Emissions Forecast)')
        plt.fill_between(x_585, y_smooth_585 - 0.15, y_smooth_585 + 0.15, color='#d62728', alpha=0.2)
        
        z_585 = np.polyfit(x_585, y_585, 1)
        p_585 = np.poly1d(z_585)
        trend_lines['SSP5-8.5'] = (x_585, p_585(x_585), z_585[0])
        
        # Add trend lines
        for scenario, (x, y, slope) in trend_lines.items():
            color = '#ff7f0e' if scenario == 'SSP2-4.5' else '#d62728'
            plt.plot(x, y, '--', color=color, linewidth=1.5,
                     label=f'{scenario} Trend: {slope:.4f}°C/year')
        
        # Format plot with enhanced styling
        title = f'April-May Maximum Temperature Projections\nKIOST-ESM Climate Model ({min_year}-{max_year})'
        subtitle = f'Location: Lat {lat_point:.2f}°, Lon {lon_point:.2f}°'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.title(subtitle, fontsize=14)
        
        # IMPROVED X-AXIS FORMATTING
        plt.xlabel('Year', fontsize=15, color='black', labelpad=15)
        plt.grid(True, linestyle='-', alpha=0.7)
        
        # Set x-axis ticks - show every year for the 2040-2050 range
        x_ticks = range(min_year, max_year + 1, 1)
        plt.xticks(x_ticks, fontsize=13, color='black')
        ax = plt.gca()
        ax.tick_params(axis='x', pad=10)
        
        # IMPROVED Y-AXIS FORMATTING
        plt.ylabel('Temperature (°C)', fontsize=15, color='black', labelpad=15)
        ax.tick_params(axis='y', labelsize=13, colors='black', pad=8)
        
        # IMPROVED LEGEND FORMATTING
        plt.legend(fontsize=13, loc='upper left', framealpha=0.95, 
                   edgecolor='black', fancybox=True, shadow=False,
                   labelcolor='black', frameon=True,
                   bbox_to_anchor=(0.02, 1.02))
        
        # Set y-axis limits with fixed maximum for better space utilization
        all_temps = ssp245_data['temperature'].tolist() + ssp585_data['temperature'].tolist()
        min_temp = min(all_temps) - 0.5
        max_temp = 25  # Fixed maximum to provide more space for legend
        plt.ylim(min_temp, max_temp)
        
        # Add explanatory text with updated scenario descriptions
        scenario_info = (
            "SSP2-4.5: Moderate Emissions Forecast\n"
            "SSP5-8.5: High Emissions Forecast\n"
            "Shaded areas represent uncertainty ranges"
        )
        plt.figtext(0.01, 0.01, scenario_info, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved comparison visualization to: {output_path}")
        
        return output_path
    
    def generate_climate_analysis(self, lat: float, lon: float, demo_id: str) -> Dict:
        """
        Complete workflow: download, extract, analyze, and visualize climate data.
        
        Args:
            lat: Latitude
            lon: Longitude
            demo_id: Demo identifier for file naming
            
        Returns:
            Dictionary with results including visualization path and metadata
        """
        logging.info(f"Starting climate analysis for demo_id={demo_id}, lat={lat}, lon={lon}")
        
        # Create output filename
        output_filename = f"{demo_id}_climate_comparison.png"
        
        # Create temporary directory for all operations
        temp_dir = tempfile.mkdtemp()
        
        # Define visualization path in temp directory (will persist until cleanup)
        visualization_path = os.path.join(temp_dir, output_filename)
        
        try:
            # Step 1: Download data from CDS API
            zip_files = self.download_climate_data(lat, lon, temp_dir)
            
            # Step 2: Extract NetCDF files
            netcdf_files = self.extract_netcdf_files(zip_files, temp_dir)
            
            # Step 3: Process both scenarios
            ssp245_data, lat_point_245, lon_point_245 = self.extract_apr_may_data(
                netcdf_files['ssp245'], "SSP2-4.5"
            )
            
            ssp585_data, lat_point_585, lon_point_585 = self.extract_apr_may_data(
                netcdf_files['ssp585'], "SSP5-8.5"
            )
            
            # Use coordinates from first dataset
            lat_point = lat_point_245
            lon_point = lon_point_245
            
            # Step 4: Create visualization in temp directory
            created_visualization_path = self.create_comparison_visualization(
                ssp245_data, ssp585_data, lat_point, lon_point, visualization_path
            )
            
            # Step 5: Validate visualization file was created successfully
            if not os.path.exists(created_visualization_path):
                raise Exception(f"Visualization file not found at {created_visualization_path}")
            
            file_size = os.path.getsize(created_visualization_path)
            if file_size == 0:
                raise Exception("Visualization file is empty")
            
            logging.info(f"✓ Successfully created visualization: {created_visualization_path} ({file_size:,} bytes)")
            
            logging.info("Climate analysis completed successfully")
            
            return {
                'success': True,
                'visualization_path': created_visualization_path,
                'temp_dir': temp_dir,  # Return temp_dir so caller can manage cleanup
                'coordinates': {'lat': lat_point, 'lon': lon_point},
                'scenarios_processed': ['SSP2-4.5', 'SSP5-8.5'],
                'year_range': {
                    'min': int(ssp245_data['year'].min()),
                    'max': int(ssp245_data['year'].max())
                }
            }
            
        except Exception as e:
            logging.error(f"Climate analysis failed: {e}")
            # Clean up temp directory on failure
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temp directory after failure: {temp_dir}")
            except Exception as cleanup_error:
                logging.warning(f"Failed to clean up temp directory: {cleanup_error}")
            
            return {
                'success': False,
                'error': str(e)
            }
        # Note: Don't clean up temp_dir here - let the caller handle it after blob upload