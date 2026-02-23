# Data loading and validation module

import pandas as pd
import geopandas as gpd
import numpy as np
import zipfile
import tempfile
import os
from shapely.validation import make_valid
from config import REQUIRED_COLS_HARVEST, REQUIRED_COLS_DETECTIONS


def repair_geometry(gdf):
    """
    Repair invalid geometries in a GeoDataFrame.
    """
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )
    return gdf


def load_boundaries_from_zip(zip_file):
    """
    Load all boundary shapefiles from a ZIP file.

    Returns: GeoDataFrame with all boundaries combined, with field names
    """
    boundaries = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract ZIP
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(tmpdir)

        # Find all .shp files
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith('.shp'):
                    shp_path = os.path.join(root, f)
                    try:
                        gdf = gpd.read_file(shp_path)
                        # Extract field name from filename or attributes
                        if 'FIELD_NAME' in gdf.columns:
                            field_name = gdf['FIELD_NAME'].iloc[0]
                        else:
                            field_name = os.path.splitext(f)[0]

                        gdf['boundary_name'] = field_name

                        # Keep useful columns
                        keep_cols = ['geometry', 'boundary_name']
                        for col in ['CLIENT_NAM', 'FARM_NAME', 'FIELD_NAME', 'FIELD_ID']:
                            if col in gdf.columns:
                                keep_cols.append(col)

                        gdf = gdf[[c for c in keep_cols if c in gdf.columns]]
                        boundaries.append(gdf)
                    except Exception as e:
                        print(f"Could not load {f}: {e}")

    if not boundaries:
        return None

    # Combine all boundaries
    combined = pd.concat(boundaries, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")

    return combined


def match_points_to_boundary(points_gdf, boundaries_gdf):
    """
    Find which boundary contains the majority of points.

    Returns: (matched_boundary_gdf, boundary_name, match_percentage)
    """
    if boundaries_gdf is None or len(boundaries_gdf) == 0:
        return None, None, 0

    # Ensure same CRS
    if points_gdf.crs != boundaries_gdf.crs:
        points_gdf = points_gdf.to_crs(boundaries_gdf.crs)

    best_match = None
    best_count = 0
    best_name = None

    for idx, boundary in boundaries_gdf.iterrows():
        # Count points within this boundary
        within = points_gdf.within(boundary.geometry)
        count = within.sum()

        if count > best_count:
            best_count = count
            best_match = boundary
            best_name = boundary.get('boundary_name', f'Boundary {idx}')

    if best_match is None:
        return None, None, 0

    match_pct = (best_count / len(points_gdf)) * 100

    # Return as single-row GeoDataFrame
    matched_gdf = gpd.GeoDataFrame([best_match], crs=boundaries_gdf.crs)

    return matched_gdf, best_name, match_pct


def clip_points_to_boundary(points_gdf, boundary_gdf):
    """
    Clip points to only those within the boundary.
    """
    if boundary_gdf is None:
        return points_gdf

    # Ensure same CRS
    if points_gdf.crs != boundary_gdf.crs:
        points_gdf = points_gdf.to_crs(boundary_gdf.crs)

    # Repair geometries before union
    boundary_gdf = repair_geometry(boundary_gdf)
    boundary_union = boundary_gdf.unary_union
    clipped = points_gdf[points_gdf.within(boundary_union)].copy()

    return clipped


def split_points_by_boundaries(points_gdf, boundaries_gdf):
    """
    Split points into groups based on which boundary they fall within.
    Each point is assigned to exactly one boundary.

    Args:
        points_gdf: GeoDataFrame with point data
        boundaries_gdf: GeoDataFrame with boundary polygons

    Returns: list of dicts: [{'boundary_name': str, 'boundary_gdf': gdf, 'points_gdf': gdf, 'area_ha': float}, ...]
    """
    if boundaries_gdf is None or len(boundaries_gdf) == 0:
        return []

    # Ensure same CRS
    if points_gdf.crs != boundaries_gdf.crs:
        points_gdf = points_gdf.to_crs(boundaries_gdf.crs)

    results = []

    for idx, boundary in boundaries_gdf.iterrows():
        # Find points within this boundary
        within_mask = points_gdf.within(boundary.geometry)
        points_in_boundary = points_gdf[within_mask].copy()

        if len(points_in_boundary) == 0:
            continue

        boundary_name = boundary.get('boundary_name',
                        boundary.get('FIELD_NAME', f'Field_{idx}'))

        # Create single-row boundary GeoDataFrame
        boundary_gdf = gpd.GeoDataFrame([boundary], crs=boundaries_gdf.crs)

        # Calculate area (repair geometry first)
        boundary_gdf = repair_geometry(boundary_gdf)
        boundary_m = boundary_gdf.to_crs(epsg=3857)
        area_ha = boundary_m.unary_union.area / 10000

        results.append({
            'boundary_name': boundary_name,
            'boundary_gdf': boundary_gdf,
            'points_gdf': points_in_boundary,
            'area_ha': area_ha,
            'point_count': len(points_in_boundary)
        })

    # Sort by point count (most points first)
    results.sort(key=lambda x: x['point_count'], reverse=True)

    return results


def get_boundary_area_ha(boundary_gdf):
    """
    Calculate boundary area in hectares.
    """
    if boundary_gdf is None:
        return 0

    # Repair geometry and convert to metric CRS for accurate area
    boundary_gdf = repair_geometry(boundary_gdf)
    boundary_m = boundary_gdf.to_crs(epsg=3857)
    area_m2 = boundary_m.unary_union.area
    return area_m2 / 10000


def detect_file_type(df):
    """
    Detect whether the uploaded file is a harvest (aggregated) or detections file.
    Returns: 'harvest', 'detections', or None if invalid
    """
    has_harvest_cols = all(col in df.columns for col in REQUIRED_COLS_HARVEST)
    has_detection_cols = all(col in df.columns for col in REQUIRED_COLS_DETECTIONS)

    if has_detection_cols and "Confidence" in df.columns:
        return "detections"
    elif has_harvest_cols:
        return "harvest"
    return None


def validate_dataframe(df, file_type):
    """
    Validate that the dataframe has required columns.
    Returns: (is_valid, error_message)
    """
    if file_type == "harvest":
        required = REQUIRED_COLS_HARVEST
    elif file_type == "detections":
        required = REQUIRED_COLS_DETECTIONS
    else:
        return False, "Unknown file type. Please upload a valid SnailCam CSV."

    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, None


def filter_by_confidence(df, min_confidence, max_confidence):
    """
    Filter detections by confidence score.
    """
    if "Confidence" not in df.columns:
        return df

    return df[(df["Confidence"] >= min_confidence) & (df["Confidence"] <= max_confidence)]


def aggregate_detections(df):
    """
    Aggregate individual detections into location-based counts.
    Groups by rounded coordinates and counts snail types.
    """
    # Round coordinates to group nearby detections
    df = df.copy()
    df["lon_round"] = df["Longitude"].round(6)
    df["lat_round"] = df["Latitude"].round(6)

    # Count by snail type
    agg = df.groupby(["lon_round", "lat_round"]).agg({
        "Longitude": "mean",
        "Latitude": "mean",
        "Snail_Type": "count"
    }).reset_index()

    # Also get counts by type
    type_counts = df.pivot_table(
        index=["lon_round", "lat_round"],
        columns="Snail_Type",
        aggfunc="size",
        fill_value=0
    ).reset_index()

    # Merge
    result = agg.merge(type_counts, on=["lon_round", "lat_round"])
    result = result.rename(columns={"Snail_Type": "Total"})

    # Ensure Conical and Italian columns exist
    if "Conical" not in result.columns:
        result["Conical"] = 0
    if "Italian" not in result.columns:
        result["Italian"] = 0

    return result[["Longitude", "Latitude", "Conical", "Italian", "Total"]]


def load_and_prepare_data(df, file_type, snail_option, min_confidence=None, max_confidence=None):
    """
    Load, filter, and prepare data for visualization.

    Returns: (prepared_df, snail_type_column)
    """
    # Apply confidence filtering and aggregation for detection files
    if file_type == "detections":
        if min_confidence is not None:
            df = filter_by_confidence(df, min_confidence, max_confidence)
        # Always aggregate detection files to get Conical/Italian columns
        df = aggregate_detections(df)
    else:
        df = df.copy()

    # Handle snail type selection
    if snail_option == "Both":
        if "Total" not in df.columns:
            df["Total"] = df["Conical"] + df["Italian"]
        snail_type = "Total"
    else:
        snail_type = snail_option

    return df, snail_type


def create_geodataframe(df, snail_type, min_count, max_count):
    """
    Create a GeoDataFrame from the dataframe with filtering.
    Returns both WGS84 and Web Mercator versions.
    """
    filtered_df = df[(df[snail_type] >= min_count) & (df[snail_type] <= max_count)]

    gdf = gpd.GeoDataFrame(
        filtered_df,
        geometry=gpd.points_from_xy(filtered_df.Longitude, filtered_df.Latitude),
        crs="EPSG:4326"
    )
    gdf_m = gdf.to_crs(epsg=3857)

    return gdf, gdf_m


def bboxes_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap.
    bbox format: (min_lon, min_lat, max_lon, max_lat)
    """
    return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
                bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])


def haversine_distance(coord1, coord2):
    """
    Calculate distance in meters between two (lon, lat) coordinates.
    """
    from math import radians, sin, cos, sqrt, atan2

    lon1, lat1 = coord1
    lon2, lat2 = coord2

    R = 6371000  # Earth radius in meters

    lat1, lat2, lon1, lon2 = map(radians, [lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c


def group_files_by_location(dataframes, distance_threshold=500):
    """
    Group files whose bounding boxes overlap or centroids are within threshold (meters).

    Args:
        dataframes: list of tuples (df, filename)
        distance_threshold: max distance in meters to consider same field

    Returns:
        list of dicts: [{files: [...], combined_df: ..., bounds: ..., centroid: ...}, ...]
    """
    groups = []

    for df, filename in dataframes:
        # Calculate bounding box and centroid
        bounds = (
            df['Longitude'].min(), df['Latitude'].min(),
            df['Longitude'].max(), df['Latitude'].max()
        )
        centroid = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

        # Check if this file belongs to an existing group
        matched_group = None
        for group in groups:
            if bboxes_overlap(bounds, group['bounds']) or \
               haversine_distance(centroid, group['centroid']) < distance_threshold:
                matched_group = group
                break

        if matched_group:
            # Add to existing group
            matched_group['files'].append(filename)
            matched_group['combined_df'] = pd.concat([matched_group['combined_df'], df], ignore_index=True)
            # Update group bounds to encompass both
            old_bounds = matched_group['bounds']
            matched_group['bounds'] = (
                min(old_bounds[0], bounds[0]),
                min(old_bounds[1], bounds[1]),
                max(old_bounds[2], bounds[2]),
                max(old_bounds[3], bounds[3])
            )
            # Update centroid
            new_bounds = matched_group['bounds']
            matched_group['centroid'] = (
                (new_bounds[0] + new_bounds[2]) / 2,
                (new_bounds[1] + new_bounds[3]) / 2
            )
        else:
            # Create new group
            groups.append({
                'files': [filename],
                'combined_df': df.copy(),
                'bounds': bounds,
                'centroid': centroid
            })

    return groups


def calculate_field_statistics(gdf_m, df, snail_type):
    """
    Calculate field-level statistics.
    Returns dict with area, total snails, density, etc.
    """
    if len(gdf_m) == 0:
        return {
            "total_area_ha": 0,
            "total_snails": 0,
            "avg_density": 0,
            "max_count": 0,
            "detection_points": 0,
            "conical_total": 0,
            "italian_total": 0
        }

    # Calculate convex hull area for field estimation
    try:
        hull = gdf_m.unary_union.convex_hull
        area_m2 = hull.area
        area_ha = area_m2 / 10000
    except:
        area_ha = 0

    # Calculate statistics
    total_snails = int(df["Conical"].sum() + df["Italian"].sum())
    conical_total = int(df["Conical"].sum())
    italian_total = int(df["Italian"].sum())
    avg_density = total_snails / area_ha if area_ha > 0 else 0
    max_count = int(df[snail_type].max()) if snail_type in df.columns else 0

    return {
        "total_area_ha": area_ha,
        "total_snails": total_snails,
        "avg_density": avg_density,
        "max_count": max_count,
        "detection_points": len(df),
        "conical_total": conical_total,
        "italian_total": italian_total
    }
