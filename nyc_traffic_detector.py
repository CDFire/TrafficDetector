import pandas as pd
import geopandas as gpd # For handling geospatial data (road network)
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import folium
from folium.plugins import MarkerCluster
from IPython.display import display
# from shapely.geometry import Point # Point might be useful for other geometry tasks.

# --- Configuration Parameters ---
# These parameters control simulation, data loading, and detection.
NUM_LINKS_TO_SIMULATE = 50  # Target number of actual road segments for simulation.
DAYS_OF_DATA = 14           # Duration of the simulated data period.
DATA_GRANULARITY_MINUTES = 15 # Time interval for data points.
CITY_CENTER_NYC = (40.7128, -74.0060) # Fallback/default geographic center.
ANOMALY_PROBABILITY = 0.02  # Likelihood of injecting an anomaly at a data point.
ROAD_NETWORK_FILE = "path/to/your/nyc_lion_streets.shp" # IMPORTANT: Update this path to your road network file.

# --- Geospatial Data Handling: Load Road Network ---
def load_road_network(filepath, num_segments_to_select, target_crs="EPSG:4326", bbox=None):
    """
    Loads road network data from a geospatial file (e.g., Shapefile).
    It selects a subset of road segments, extracts their unique identifiers,
    and determines representative geographic coordinates (centroids) for each,
    reprojecting to the target CRS if necessary.

    Parameters:
    - filepath (str): Path to the street centerline geospatial file.
    - num_segments_to_select (int): The desired number of road segments for simulation.
    - target_crs (str): The target Coordinate Reference System for output geometries (e.g., "EPSG:4326" for WGS84).
    - bbox (tuple, optional): A bounding box (minx, miny, maxx, maxy) to filter segments spatially.

    Returns:
    - tuple: (list of link_ids, dict of link_geometries {link_id: (lat, lon)}) or (None, None) on failure.
    """
    print(f"Attempting to load road network from: {filepath}")
    try:
        gdf_roads = gpd.read_file(filepath)
        if gdf_roads.empty:
            print("Road network file loaded, but it contains no features.")
            return None, None
    except Exception as e:
        print(f"Error loading road network file '{filepath}': {e}")
        print("Ensure the path is correct and the file is a valid geospatial format (e.g., Shapefile, GeoJSON).")
        return None, None

    # Identify or create a unique identifier for each road segment ('link_id').
    # Common unique ID columns in NYC LION data include 'PHYSICALID' or 'OBJECTID'.
    # If these are not present, a composite ID or the GeoDataFrame index is used as a fallback.
    if 'PHYSICALID' in gdf_roads.columns:
        gdf_roads['link_id_col'] = gdf_roads['PHYSICALID'].astype(str)
    elif 'StreetCode' in gdf_roads.columns and 'SegmentID' in gdf_roads.columns: # Alternative LION structure
        gdf_roads['link_id_col'] = gdf_roads['StreetCode'].astype(str) + "_" + gdf_roads['SegmentID'].astype(str)
    elif 'OBJECTID' in gdf_roads.columns: # A common Esri Shapefile ID
        gdf_roads['link_id_col'] = gdf_roads['OBJECTID'].astype(str)
    else:
        print("Warning: Standard unique ID column not found. Using GeoDataFrame index as 'link_id'.")
        gdf_roads['link_id_col'] = gdf_roads.index.astype(str)

    # Reproject the entire GeoDataFrame to the target CRS if it's different.
    # This is crucial for consistency, especially for Folium maps (WGS84).
    if gdf_roads.crs and gdf_roads.crs.to_string() != target_crs:
        print(f"Reprojecting road network from {gdf_roads.crs.to_string()} to {target_crs}...")
        try:
            gdf_roads = gdf_roads.to_crs(target_crs)
        except Exception as e:
            print(f"Error during CRS reprojection: {e}. Geometries might not be in {target_crs}.")
            # Proceeding with original CRS if reprojection fails, but map display might be affected.

    # Optionally filter road segments by a specified bounding box.
    if bbox:
        try:
            gdf_roads = gdf_roads.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            if gdf_roads.empty:
                print("No road segments found within the specified bounding box.")
                return None, None
        except Exception as e:
            print(f"Error applying bounding box filter: {e}")


    # Select a sample of road segments if the dataset is larger than desired.
    if len(gdf_roads) > num_segments_to_select:
        selected_roads_gdf = gdf_roads.sample(n=num_segments_to_select, random_state=42) # random_state for reproducibility.
    elif not gdf_roads.empty:
        selected_roads_gdf = gdf_roads
        print(f"Note: Using all {len(selected_roads_gdf)} available road segments (requested {num_segments_to_select}).")
    else: # gdf_roads became empty after filtering or was initially empty
        print("No road segments available for selection.")
        return None, None

    link_geometries_dict = {}
    link_ids_list = []

    print(f"Processing {len(selected_roads_gdf)} selected road segments to extract geometries...")
    for index, row_data in selected_roads_gdf.iterrows():
        link_id = str(row_data['link_id_col'])
        geometry_obj = row_data.geometry

        # We expect LineString geometries for road centerlines.
        if geometry_obj and (geometry_obj.geom_type == 'LineString' or geometry_obj.geom_type == 'MultiLineString'):
            # Use the centroid of the linestring as its representative point for simulation/mapping.
            # Geometry should already be in target_crs due to earlier reprojection.
            representative_point = geometry_obj.centroid
            link_geometries_dict[link_id] = (representative_point.y, representative_point.x) # Store as (latitude, longitude)
            link_ids_list.append(link_id)
        else:
            print(f"Warning: Skipping segment {link_id} due to unexpected or null geometry type: {geometry_obj.geom_type if geometry_obj else 'Null'}.")

    if not link_ids_list:
        print("No valid LineString geometries were processed from the selected road segments.")
        return None, None
        
    return link_ids_list, link_geometries_dict


# --- 1. Data Simulation: NYC-like Traffic Data Generation (Modified for Road Network) ---
def generate_nyc_traffic_data(link_ids_from_network, link_geometries_from_network, days, granularity_minutes):
    """
    Simulates traffic speed data for a provided list of road links and their geometries.
    If road network data is unavailable, it falls back to generating data for generic
    links with random geographic coordinates.
    The core simulation logic for speed patterns and anomalies remains consistent.
    """
    all_simulated_data = []

    # Determine if using actual road network data or fallback.
    if not link_ids_from_network or not link_geometries_from_network:
        print("Warning: Road network data not provided or incomplete. Falling back to random point simulation.")
        num_fallback_links = NUM_LINKS_TO_SIMULATE # Use the global configuration for fallback.
        link_ids_from_network = [f"FALLBACK_LNK_{1001+i}" for i in range(num_fallback_links)]
        link_geometries_from_network = {}
        # Generate random coordinates for fallback links.
        rand_lats, rand_lons = np.random.normal(CITY_CENTER_NYC[0], 0.05, num_fallback_links), \
                               np.random.normal(CITY_CENTER_NYC[1], 0.05, num_fallback_links)
        for i, fallback_lid in enumerate(link_ids_from_network):
            link_geometries_from_network[fallback_lid] = (rand_lats[i], rand_lons[i])
        print(f"Simulating for {num_fallback_links} fallback links with random coordinates.")
    else:
        print(f"Simulating traffic data for {len(link_ids_from_network)} links using provided road network geometries...")

    start_simulation_time = datetime.now() - timedelta(days=days)
    total_time_periods = days * 24 * (60 // granularity_minutes)

    for link_id_current in link_ids_from_network:
        # Assign baseline speed characteristics for the current link.
        base_avg_speed = random.uniform(15, 45) # mph
        rush_hour_reduction_factor = random.uniform(0.4, 0.7)
        weekend_increase_factor = random.uniform(1.1, 1.3)

        # Retrieve the geographic coordinates for this link.
        link_latitude, link_longitude = link_geometries_from_network.get(link_id_current, CITY_CENTER_NYC)

        for i in range(total_time_periods):
            current_sim_time = start_simulation_time + timedelta(minutes=i * granularity_minutes)
            simulated_speed = base_avg_speed
            hour_of_day = current_sim_time.hour
            day_of_week_val = current_sim_time.weekday() # Monday=0, Sunday=6.

            # Apply standard traffic patterns (rush hours, weekends).
            if 7 <= hour_of_day < 10 and day_of_week_val < 5: simulated_speed *= rush_hour_reduction_factor
            elif 16 <= hour_of_day < 19 and day_of_week_val < 5: simulated_speed *= rush_hour_reduction_factor * random.uniform(0.9, 1.1)
            elif hour_of_day < 6 or hour_of_day > 22: simulated_speed *= 1.2
            if day_of_week_val >= 5: simulated_speed *= weekend_increase_factor
            
            # Add random noise and enforce plausible speed limits.
            simulated_speed += np.random.normal(0, simulated_speed * 0.1)
            simulated_speed = max(5, simulated_speed); simulated_speed = min(70, simulated_speed)
            
            # Inject simulated anomalies.
            is_this_a_simulated_anomaly = False
            if random.random() < ANOMALY_PROBABILITY:
                is_this_a_simulated_anomaly = True
                if random.random() < 0.7: simulated_speed *= random.uniform(0.1, 0.4) # Congestion
                else: simulated_speed *= random.uniform(1.5, 2.0) # Unusual clearance
                simulated_speed = max(1, simulated_speed)
            
            all_simulated_data.append({
                "link_id": link_id_current,
                "timestamp": current_sim_time,
                "speed_mph": round(simulated_speed, 2),
                "latitude": link_latitude,  # Using representative coordinates from road network.
                "longitude": link_longitude,
                "is_simulated_anomaly": is_this_a_simulated_anomaly,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week_val
            })
            
    simulated_df = pd.DataFrame(all_simulated_data)
    if not simulated_df.empty:
        simulated_df['timestamp'] = pd.to_datetime(simulated_df['timestamp'])
    return simulated_df

# --- (Anomaly Detection and Visualization sections remain largely the same as your previous "formal comments" version) ---
# --- The main difference is how `traffic_df` is initially populated. ---
# --- We will paste those sections here for completeness, ensuring formal comments. ---

# --- Main Script Execution Flow ---

# 1. Load Road Network Data
# Define an approximate bounding box for Manhattan to focus the road selection, if desired.
# manhattan_bbox = (-74.02, 40.70, -73.93, 40.80) # (min_longitude, min_latitude, max_longitude, max_latitude)
# actual_road_link_ids, actual_road_link_geometries = load_road_network(ROAD_NETWORK_FILE, NUM_LINKS_TO_SIMULATE, bbox=manhattan_bbox)
actual_road_link_ids, actual_road_link_geometries = load_road_network(ROAD_NETWORK_FILE, NUM_LINKS_TO_SIMULATE)


# 2. Generate Traffic Data
# The simulation will use actual road segments if loaded successfully, otherwise it falls back.
traffic_df = generate_nyc_traffic_data(actual_road_link_ids, actual_road_link_geometries, DAYS_OF_DATA, DATA_GRANULARITY_MINUTES)

if traffic_df.empty:
    print("Critical Error: Traffic DataFrame is empty after simulation. Further processing cannot continue. Exiting.")
    exit()

print("\n--- Simulated Traffic Data Generation: Sample Output (Potentially on Real Road Segments) ---")
print(traffic_df.head())


# --- 2. Anomaly Detection Methodologies (Copied from previous formal version for completeness) ---

def detect_anomalies_stl(series, period, robust=True, residual_std_threshold=3.2):
    """
    Detects anomalies in a time series using STL (Seasonal and Trend decomposition using Loess).
    Anomalies are identified as data points where the residual component (after removing
    trend and seasonality) deviates significantly from the mean of the residuals.

    Parameters:
    - series (pd.Series): Time series data with a DatetimeIndex and speed values.
    - period (int): The primary seasonal period in the data (e.g., number of samples per day).
    - robust (bool): If True, the decomposition is robust to the presence of outliers. Recommended.
    - residual_std_threshold (float): The number of standard deviations from the residual mean
                                      required to classify a point as an anomaly. A higher value
                                      results in lower sensitivity.
    Returns:
    - pd.Series: A boolean Series indicating detected anomalies.
    """
    if series.empty or len(series) < 2 * period: # STL requires at least two full periods of data for reliable decomposition.
        return pd.Series([False] * len(series), index=series.index)

    stl_model = STL(series, period=period, robust=robust)
    decomposition_results = stl_model.fit()
    residuals = decomposition_results.resid
    residual_mean = residuals.mean()
    residual_std_dev = residuals.std()
    anomalies = (residuals < residual_mean - residual_std_threshold * residual_std_dev) | \
                (residuals > residual_mean + residual_std_threshold * residual_std_dev)
    return anomalies

def detect_anomalies_isolation_forest(df_segment, features, contamination=0.02):
    """
    Detects anomalies using the Isolation Forest algorithm. Isolation Forest isolates
    observations by randomly selecting a feature and then randomly selecting a split
    value. Anomalies, being rare and different, are expected to be isolated in fewer splits.

    Parameters:
    - df_segment (pd.DataFrame): DataFrame for a single segment, containing the specified features.
    - features (list): List of column names to be used as features for the model.
    - contamination (float or 'auto'): The expected proportion of outliers in the data set.
                                       This parameter influences the sensitivity of the model;
                                       a lower value results in more conservative detection.
    Returns:
    - pd.Series: A boolean Series indicating detected anomalies.
    """
    if df_segment.empty or len(df_segment) < 2:
        return pd.Series([False] * len(df_segment), index=df_segment.index)
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    model.fit(df_segment[features])
    predictions = model.predict(df_segment[features])
    anomalies = predictions == -1
    return pd.Series(anomalies, index=df_segment.index)

# --- Anomaly Detection: Processing Data for Each Link (Copied from previous formal version) ---
all_anomalies_list = []
daily_period = 24 * (60 // DATA_GRANULARITY_MINUTES)
print(f"\n--- Commencing Anomaly Detection (STL Daily Period: {daily_period}) ---")

for link_id_iter, group_data_iter in traffic_df.groupby('link_id'):
    print(f"Processing road segment: {link_id_iter}")
    segment_df_iter = group_data_iter.set_index('timestamp').sort_index()
    min_timestamp_iter, max_timestamp_iter = segment_df_iter.index.min(), segment_df_iter.index.max()
    full_datetime_range_iter = pd.date_range(start=min_timestamp_iter, end=max_timestamp_iter, freq=f'{DATA_GRANULARITY_MINUTES}min')
    segment_df_iter = segment_df_iter.reindex(full_datetime_range_iter)
    segment_df_iter['speed_mph'] = segment_df_iter['speed_mph'].ffill().bfill()
    segment_df_iter['link_id'] = link_id_iter
    for col_name_iter in ['latitude', 'longitude', 'hour_of_day', 'day_of_week', 'is_simulated_anomaly']:
        if col_name_iter in segment_df_iter.columns:
             segment_df_iter[col_name_iter] = segment_df_iter[col_name_iter].ffill().bfill()
    segment_df_iter['hour_of_day'] = segment_df_iter['hour_of_day'].ffill().bfill()
    segment_df_iter['day_of_week'] = segment_df_iter['day_of_week'].ffill().bfill()

    if not segment_df_iter['speed_mph'].dropna().empty and len(segment_df_iter['speed_mph'].dropna()) >= 2 * daily_period:
        stl_anomalies_iter = detect_anomalies_stl(segment_df_iter['speed_mph'].dropna(), period=daily_period, residual_std_threshold=3.2)
        segment_df_iter['stl_anomaly'] = stl_anomalies_iter.reindex(segment_df_iter.index).fillna(False)
    else:
        print(f"  STL skipped for {link_id_iter}: Insufficient data.")
        segment_df_iter['stl_anomaly'] = False
    
    features_for_iso_forest_iter = ['speed_mph', 'hour_of_day', 'day_of_week']
    segment_df_for_iso_forest_iter = segment_df_iter[features_for_iso_forest_iter].dropna()
    if not segment_df_for_iso_forest_iter.empty:
        iso_forest_anomalies_iter = detect_anomalies_isolation_forest(segment_df_for_iso_forest_iter, features_for_iso_forest_iter, contamination=0.02)
        segment_df_iter['iso_forest_anomaly'] = iso_forest_anomalies_iter.reindex(segment_df_iter.index).fillna(False)
    else:
        print(f"  Isolation Forest skipped for {link_id_iter}: Insufficient non-NaN feature data.")
        segment_df_iter['iso_forest_anomaly'] = False
    
    segment_df_iter['any_anomaly'] = segment_df_iter['stl_anomaly'] | segment_df_iter['iso_forest_anomaly']
    anomalies_subset_iter = segment_df_iter[segment_df_iter['any_anomaly']]
    if anomalies_subset_iter.index.name is None:
        anomalies_subset_iter.index.name = 'timestamp' 
    anomalies_detected_this_segment_iter = anomalies_subset_iter.reset_index() 
    if not anomalies_detected_this_segment_iter.empty:
        all_anomalies_list.append(anomalies_detected_this_segment_iter)

if all_anomalies_list:
    all_anomalies_df = pd.concat(all_anomalies_list, ignore_index=True)
    print(f"\n--- Anomaly Detection Results Summary ---")
    print(f"Total potential anomalies detected: {len(all_anomalies_df)}")
    if 'index' in all_anomalies_df.columns and 'timestamp' not in all_anomalies_df.columns:
        all_anomalies_df.rename(columns={'index': 'timestamp'}, inplace=True)
    print("Columns in consolidated anomalies DataFrame:", all_anomalies_df.columns.tolist())
    print("\nSample of Detected Anomalies:")
    display_columns_sample_iter = ['timestamp', 'link_id', 'speed_mph', 'stl_anomaly', 'iso_forest_anomaly']
    if 'is_simulated_anomaly' in all_anomalies_df.columns:
        display_columns_sample_iter.append('is_simulated_anomaly')
    actual_display_columns_iter = [col for col in display_columns_sample_iter if col in all_anomalies_df.columns]
    if actual_display_columns_iter:
        print(all_anomalies_df[actual_display_columns_iter].head())
        if 'is_simulated_anomaly' in all_anomalies_df.columns and 'any_anomaly' in all_anomalies_df.columns:
            performance_comparison_iter = pd.crosstab(all_anomalies_df['is_simulated_anomaly'], all_anomalies_df['any_anomaly'],
                                                 rownames=['Simulated Ground Truth'], colnames=['Detected by System'])
            print("\nDetection Performance Evaluation:")
            print(performance_comparison_iter)
    else:
        print("Unable to display sample: Expected columns missing. Raw head:")
        print(all_anomalies_df.head())
else:
    print("\nNo anomalies detected by the implemented methods.")
    all_anomalies_df = pd.DataFrame()

# --- 3. Visualization of Results (Copied from previous formal version) ---
print("\n--- Generating Visualizations of Detected Anomalies ---")
if not traffic_df.empty:
    example_link_id_list_plot = traffic_df['link_id'].unique()
    if not example_link_id_list_plot.size > 0:
        print("No link IDs for plotting; skipping time series plot.")
    else:
        example_link_id_to_plot_iter = example_link_id_list_plot[0]
        print(f"Generating time series plot for Link ID: {example_link_id_to_plot_iter}")
        example_link_data_original_plot = traffic_df[traffic_df['link_id'] == example_link_id_to_plot_iter].copy()
        example_link_data_indexed_plot = example_link_data_original_plot.set_index('timestamp').sort_index()
        if example_link_data_indexed_plot.empty:
            print(f"No data for example link ID ({example_link_id_to_plot_iter}) for plotting.")
        else:
            min_ts_for_plot_iter, max_ts_for_plot_iter = example_link_data_indexed_plot.index.min(), example_link_data_indexed_plot.index.max()
            full_range_for_plot_iter = pd.date_range(start=min_ts_for_plot_iter, end=max_ts_for_plot_iter, freq=f'{DATA_GRANULARITY_MINUTES}min')
            example_link_data_plot_iter = example_link_data_indexed_plot.reindex(full_range_for_plot_iter)
            example_link_data_plot_iter['speed_mph'] = example_link_data_plot_iter['speed_mph'].ffill().bfill()
            if 'is_simulated_anomaly' in example_link_data_plot_iter.columns:
                example_link_data_plot_iter['is_simulated_anomaly'] = example_link_data_plot_iter['is_simulated_anomaly'].ffill().bfill()
            anomalies_for_link_plot_iter = pd.DataFrame()
            if not all_anomalies_df.empty and 'link_id' in all_anomalies_df.columns:
                anomalies_for_link_plot_iter = all_anomalies_df[all_anomalies_df['link_id'] == example_link_id_to_plot_iter]
            plt.figure(figsize=(18, 7))
            plt.plot(example_link_data_plot_iter.index, example_link_data_plot_iter['speed_mph'], label=f'Speed - {example_link_id_to_plot_iter}', alpha=0.7)
            if not anomalies_for_link_plot_iter.empty and 'timestamp' in anomalies_for_link_plot_iter.columns:
                stl_detected_for_plot_iter = anomalies_for_link_plot_iter[anomalies_for_link_plot_iter['stl_anomaly'] == True]
                iso_detected_for_plot_iter = anomalies_for_link_plot_iter[anomalies_for_link_plot_iter['iso_forest_anomaly'] == True]
                if not stl_detected_for_plot_iter.empty:
                    plt.scatter(stl_detected_for_plot_iter['timestamp'], stl_detected_for_plot_iter['speed_mph'], 
                                color='red', s=80, marker='o', label='STL Anomaly', alpha=0.8, 
                                facecolors='none', edgecolors='red', linewidths=1.5, zorder=5)
                if not iso_detected_for_plot_iter.empty:
                    plt.scatter(iso_detected_for_plot_iter['timestamp'], iso_detected_for_plot_iter['speed_mph'], 
                                color='purple', s=100, marker='x', label='Isolation Forest Anomaly', 
                                alpha=0.8, linewidths=1.5, zorder=5)
            if 'is_simulated_anomaly' in example_link_data_plot_iter.columns:
                simulated_anomalies_for_plot_iter = example_link_data_plot_iter[example_link_data_plot_iter['is_simulated_anomaly'] == True]
                if not simulated_anomalies_for_plot_iter.empty:
                     plt.scatter(simulated_anomalies_for_plot_iter.index, simulated_anomalies_for_plot_iter['speed_mph'],
                                color='green', s=40, marker='s', label='Simulated Anomaly', alpha=0.5, zorder=4)
            plt.title(f'Traffic Speed and Detected Anomalies for Link {example_link_id_to_plot_iter}')
            plt.xlabel('Timestamp'); plt.ylabel('Speed (mph)'); plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); plt.show()

if not all_anomalies_df.empty and all(col in all_anomalies_df.columns for col in ['latitude', 'longitude', 'timestamp', 'link_id']):
    anomalies_for_map_display_iter = all_anomalies_df[all_anomalies_df['timestamp'] > (datetime.now() - timedelta(days=1))]
    if len(anomalies_for_map_display_iter) > 500:
        print(f"Sampling 500 anomalies for map display (from {len(anomalies_for_map_display_iter)} recent).")
        anomalies_for_map_display_iter = anomalies_for_map_display_iter.sample(n=500, random_state=1)
    elif anomalies_for_map_display_iter.empty:
        print("No anomalies from last 24 hours for map display.")
    if not anomalies_for_map_display_iter.empty:
        print(f"Generating map with {len(anomalies_for_map_display_iter)} anomaly points...")
        map_nyc_anomalies_iter = folium.Map(location=CITY_CENTER_NYC, zoom_start=11, tiles="CartoDB positron")
        marker_cluster_group_iter = MarkerCluster(name="Traffic Anomalies").add_to(map_nyc_anomalies_iter)
        for index_iter, anomaly_row_iter in anomalies_for_map_display_iter.iterrows():
            popup_content_html_iter = f"<b>Link ID:</b> {anomaly_row_iter.get('link_id', 'N/A')}<br>" \
                                   f"<b>Timestamp:</b> {anomaly_row_iter['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>" \
                                   f"<b>Speed:</b> {anomaly_row_iter.get('speed_mph', 0):.1f} mph<br>" \
                                   "<b>Detection Method(s):</b>"
            if anomaly_row_iter.get('stl_anomaly'): popup_content_html_iter += " STL"
            if anomaly_row_iter.get('iso_forest_anomaly'): popup_content_html_iter += " IsolationForest"
            if 'is_simulated_anomaly' in anomaly_row_iter and anomaly_row_iter.get('is_simulated_anomaly'): 
                popup_content_html_iter += "<br><b>(Simulated Anomaly)</b>"
            folium.Marker(
                location=[anomaly_row_iter['latitude'], anomaly_row_iter['longitude']],
                popup=folium.Popup(popup_content_html_iter, max_width=300),
                tooltip=f"Anomaly: {anomaly_row_iter.get('link_id', 'N/A')} @ {anomaly_row_iter['speed_mph']:.0f}mph",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(marker_cluster_group_iter)
        folium.LayerControl().add_to(map_nyc_anomalies_iter)
        map_nyc_anomalies_iter.save("nyc_traffic_anomalies_map.html")
        print("\nAnomaly map saved to 'nyc_traffic_anomalies_map.html'.")
        # display(map_nyc_anomalies_iter) # For Jupyter
    else:
        if not all_anomalies_df.empty:
             print("\nNo recent anomalies (last day) for map display; older anomalies may exist.")
else:
    print("\nMap generation skipped: No anomalies detected or missing required data.")
print("\n--- Script Execution Concluded ---")