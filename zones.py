# Zone classification module
#
# Zone scheme:
#   0 = no data (NaN / outside surveyed area)
#   1 = clear (surveyed, no snails — density <= clear_threshold)
#   2..N = risk zones based on user thresholds
#
# The clear zone boundary is fixed and won't move when thresholds change.
# Only risk zones are affected by threshold adjustments.

import numpy as np
from skimage import measure
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
import io
import zipfile
from config import NO_DATA_COLOR, CLEAR_ZONE_COLOR, RISK_ZONE_COLORS


def fill_gaps(grid_z, max_fill_pixels=20, boundary_mask=None):
    """
    Fill internal NaN gaps in the grid by expanding from surrounding valid data.
    Only fills gaps within boundary (if provided) or within data extent.

    Args:
        grid_z: 2D numpy array with NaN gaps
        max_fill_pixels: maximum distance (in pixels) to fill inward from edges
        boundary_mask: optional boolean mask where True = inside field boundary

    Returns: grid_z with internal gaps filled
    """
    from scipy import ndimage

    grid_filled = grid_z.copy()
    nan_mask = np.isnan(grid_filled)

    if not np.any(nan_mask):
        return grid_filled

    valid_mask = ~nan_mask

    # Determine fillable area
    if boundary_mask is not None:
        # Use the provided boundary - fill all NaN inside boundary
        fillable_mask = nan_mask & boundary_mask
    else:
        # No boundary - use morphological closing to identify internal gaps
        closed = ndimage.binary_closing(valid_mask, iterations=max_fill_pixels)
        fillable_mask = nan_mask & closed

    if not np.any(fillable_mask):
        return grid_filled

    # Iteratively expand valid regions into fillable gaps
    filled_values = grid_filled.copy()
    current_valid = valid_mask.copy()

    for iteration in range(max_fill_pixels * 3):  # More iterations when using boundary
        remaining = fillable_mask & ~current_valid
        if not np.any(remaining):
            break

        # Dilate the valid mask by 1 pixel
        dilated = ndimage.binary_dilation(current_valid)
        new_pixels = dilated & ~current_valid & fillable_mask

        if not np.any(new_pixels):
            break

        # For new pixels, compute average of valid neighbors
        kernel = np.array([[0.5, 1, 0.5],
                          [1, 0, 1],
                          [0.5, 1, 0.5]])
        kernel = kernel / kernel.sum()

        # Compute weighted sum of neighbors
        neighbor_sum = ndimage.convolve(
            np.where(current_valid, filled_values, 0),
            kernel,
            mode='constant',
            cval=0
        )
        neighbor_count = ndimage.convolve(
            current_valid.astype(float),
            kernel,
            mode='constant',
            cval=0
        )

        # Fill new pixels with average of neighbors
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_values = np.where(neighbor_count > 0, neighbor_sum / neighbor_count, 0)

        filled_values = np.where(new_pixels, avg_values, filled_values)
        current_valid = current_valid | new_pixels

    return filled_values


def create_boundary_mask(grid_shape, bounds, boundary_gdf):
    """
    Create a boolean mask from a boundary GeoDataFrame.

    Args:
        grid_shape: tuple (height, width) of the grid
        bounds: tuple (minx, maxx, miny, maxy) in Web Mercator
        boundary_gdf: GeoDataFrame with boundary polygon

    Returns: boolean mask where True = inside boundary
    """
    if boundary_gdf is None:
        return None

    import numpy as np
    import shapely

    minx, maxx, miny, maxy = bounds
    height, width = grid_shape

    # Get boundary polygon in Web Mercator
    from data_loader import repair_geometry
    boundary_union = repair_geometry(boundary_gdf).to_crs(epsg=3857).unary_union

    # Fully vectorized point-in-polygon check
    x_coords = np.linspace(minx, maxx, width)
    y_coords = np.linspace(miny, maxy, height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    flat_points = shapely.points(xx.ravel(), yy.ravel())
    mask = shapely.contains(boundary_union, flat_points).reshape(grid_shape)

    return mask


def classify_zones(grid_z, thresholds, zone_count, clear_threshold=0.1):
    """
    Classify grid cells into zones based on thresholds.

    Zone scheme:
        0 = no data (NaN areas)
        1 = clear (surveyed, density <= clear_threshold) — FIXED boundary
        2..zone_count+1 = risk zones based on thresholds

    The clear zone is determined by clear_threshold only.
    Changing risk zone thresholds will NOT affect the clear zone boundary
    or the field edges.

    Args:
        grid_z: 2D numpy array of interpolated values
        thresholds: list of threshold values for risk zone boundaries
        zone_count: number of risk zones (2 or 3)
        clear_threshold: density below which cells are marked "clear"

    Returns: zone_map (2D array with zone numbers)
    """
    zone_map = np.zeros_like(grid_z, dtype=np.uint8)
    valid = ~np.isnan(grid_z)

    # Clear zone: surveyed but no snails (density <= clear_threshold)
    zone_map[valid & (grid_z <= clear_threshold)] = 1

    # Risk zones: only cells with density > clear_threshold
    positive = valid & (grid_z > clear_threshold)

    if zone_count == 1:
        zone_map[positive] = 2
    elif zone_count == 2:
        t1 = thresholds[0]
        zone_map[positive & (grid_z <= t1)] = 2
        zone_map[positive & (grid_z > t1)] = 3
    else:
        t1 = thresholds[0]
        t2 = thresholds[1] if len(thresholds) > 1 else t1 * 2
        zone_map[positive & (grid_z <= t1)] = 2
        zone_map[positive & (grid_z > t1) & (grid_z <= t2)] = 3
        zone_map[positive & (grid_z > t2)] = 4

    return zone_map


def filter_small_zones(zone_map, min_area_ha, zone_count, pixel_size=5):
    """
    Remove small isolated zone regions (noise reduction).

    Filters both small risk zone patches AND small clear zone holes
    inside risk areas. Each small region is absorbed into the
    surrounding dominant zone. No-data (0) and field edges are
    never changed.

    Args:
        zone_map: 2D array with zone numbers
        min_area_ha: minimum area in hectares (regions smaller are removed)
        zone_count: number of risk zones
        pixel_size: size of each pixel in meters
    """
    if min_area_ha <= 0:
        return zone_map

    zone_map = zone_map.copy()

    # Convert hectares to pixels
    ha_per_pixel = (pixel_size ** 2) / 10000
    min_pixels = min_area_ha / ha_per_pixel

    # Filter risk zones (2 through zone_count+1) AND clear zone (1)
    for z in range(1, zone_count + 2):
        labeled = measure.label(zone_map == z)
        props = measure.regionprops(labeled)
        for region in props:
            if region.area < min_pixels:
                # Find the dominant surrounding zone
                minr, minc, maxr, maxc = region.bbox
                # Expand bbox by 2 pixels to get surrounding context
                minr = max(0, minr - 2)
                minc = max(0, minc - 2)
                maxr = min(zone_map.shape[0], maxr + 2)
                maxc = min(zone_map.shape[1], maxc + 2)

                surrounding = zone_map[minr:maxr, minc:maxc].copy()
                region_mask = labeled[minr:maxr, minc:maxc] == region.label

                # Get values outside this region but in the bbox
                surrounding[region_mask] = 0
                surrounding_values = surrounding[surrounding > 0]

                if len(surrounding_values) > 0:
                    fill_zone = int(np.bincount(surrounding_values).argmax())
                else:
                    # For clear zones, default to keeping clear
                    # For risk zones, default to clear
                    fill_zone = 1

                # Don't replace clear with clear (no-op)
                if fill_zone != z:
                    zone_map[labeled == region.label] = fill_zone

    return zone_map


def get_zone_colors(zone_count):
    """
    Get list of RGB colors for all zones based on the number of risk zones.

    Returns: list of (r,g,b) tuples indexed by zone number
        [0] = no data, [1] = clear, [2..] = risk zones
    """
    colors = [NO_DATA_COLOR, CLEAR_ZONE_COLOR]
    colors.extend(RISK_ZONE_COLORS.get(zone_count, RISK_ZONE_COLORS[3]))
    return colors


def create_zone_rgb(zone_map, zone_count):
    """
    Convert zone map to RGB image for display.

    Returns: 3D numpy array (height, width, 3)
    """
    colors = get_zone_colors(zone_count)
    zone_rgb = np.ones((*zone_map.shape, 3)) * colors[0][0]  # No-data background

    for z in range(len(colors)):
        mask = zone_map == z
        if np.any(mask):
            zone_rgb[mask] = colors[z]

    return zone_rgb


def calculate_zone_areas(zone_map, pixel_size, zone_count):
    """
    Calculate the area of each zone in hectares.
    Includes clear zone (1) and all risk zones (2..zone_count+1).

    Returns: dict mapping zone number to area in hectares
    """
    ha_per_pixel = (pixel_size ** 2) / 10000
    areas = {}

    for z in range(1, zone_count + 2):
        pixel_count = np.sum(zone_map == z)
        areas[z] = pixel_count * ha_per_pixel

    return areas


def get_zone_statistics(zone_map, grid_z, zone_count, pixel_size):
    """
    Get statistics for clear zone and risk zones.

    Returns: list of dicts with zone stats (clear zone first, then risk zones)
    """
    ha_per_pixel = (pixel_size ** 2) / 10000
    stats = []

    for z in range(1, zone_count + 2):
        mask = zone_map == z
        pixel_count = int(np.sum(mask))
        area_ha = pixel_count * ha_per_pixel

        if pixel_count > 0 and z >= 2:
            zone_values = grid_z[mask]
            zone_values = zone_values[~np.isnan(zone_values)]
            avg_density = float(np.mean(zone_values)) if len(zone_values) > 0 else 0
            max_density = float(np.max(zone_values)) if len(zone_values) > 0 else 0
        else:
            avg_density = 0
            max_density = 0

        stats.append({
            "zone": z,
            "zone_type": "clear" if z == 1 else "risk",
            "area_ha": area_ha,
            "avg_density": avg_density,
            "max_density": max_density,
            "pixels": pixel_count
        })

    return stats


def zones_to_polygons(zone_map, bounds, pixel_size, zone_count,
                      zone_names, zone_rates, thresholds, clear_threshold=0.1):
    """
    Convert zone raster to vector polygons for prescription export.

    Uses grid-cell merging (row strips) for reliable, valid polygons.
    Includes clear zone (Rate=0) so the entire field is covered.

    Args:
        zone_map: 2D numpy array with zone numbers
        bounds: tuple (minx, maxx, miny, maxy) in Web Mercator (EPSG:3857)
        pixel_size: size of each pixel in meters
        zone_count: number of risk zones
        zone_names: list of risk zone names
        zone_rates: list of bait rates (kg/ha) for risk zones
        thresholds: list of threshold values
        clear_threshold: density threshold for clear zone

    Returns: GeoDataFrame with zone polygons and attributes (in WGS84)
    """
    minx, maxx, miny, maxy = bounds
    height, width = zone_map.shape

    px_width = (maxx - minx) / width
    px_height = (maxy - miny) / height

    features = []

    # Process all zones: clear (1) + risk zones (2..zone_count+1)
    for zone_num in range(1, zone_count + 2):
        mask = zone_map == zone_num
        if not np.any(mask):
            continue

        # Build polygons from horizontal row strips (much faster than per-cell)
        strips = []
        for row in range(height):
            row_mask = mask[row]
            if not np.any(row_mask):
                continue
            # Find runs of consecutive True values
            padded = np.concatenate([[False], row_mask, [False]])
            changes = np.diff(padded.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            y0 = miny + row * px_height
            y1 = y0 + px_height
            for s, e in zip(starts, ends):
                x0 = minx + s * px_width
                x1 = minx + e * px_width
                strips.append(box(x0, y0, x1, y1))

        if not strips:
            continue

        # Merge strips into zone polygon(s)
        merged = unary_union(strips)

        # Simplify to smooth stair-step edges and reduce vertex count
        tol = min(px_width, px_height) * 1.5
        merged = merged.simplify(tol, preserve_topology=True)

        # Buffer outward slightly so adjacent zones overlap instead of having
        # gaps.  Independent simplification causes zone boundaries to diverge
        # by up to ~tol, leaving slivers with no coverage.  A small overlap
        # (half the simplification tolerance) is invisible in JD Operations
        # Center but eliminates missing-coverage triangles.
        overlap_buf = tol * 0.5
        merged = merged.buffer(overlap_buf, join_style=2)  # mitre join

        if merged.is_empty:
            continue

        # Area in hectares (Web Mercator = meters)
        area_ha = merged.area / 10000

        if zone_num == 1:
            # Clear zone
            features.append({
                'geometry': merged,
                'Zone': 0,
                'Name': 'Clear',
                'Rate': 0.0,
                'Area_ha': round(area_ha, 2),
            })
        else:
            # Risk zone
            risk_idx = zone_num - 2  # 0-based index into zone_names/zone_rates
            name = zone_names[risk_idx] if risk_idx < len(zone_names) else f"Zone {risk_idx+1}"
            rate = float(zone_rates[risk_idx]) if risk_idx < len(zone_rates) else 0.0

            features.append({
                'geometry': merged,
                'Zone': risk_idx + 1,
                'Name': name,
                'Rate': rate,
                'Area_ha': round(area_ha, 2),
            })

    if not features:
        return None

    # Create GeoDataFrame in Web Mercator, convert to WGS84 for export
    gdf = gpd.GeoDataFrame(features, crs="EPSG:3857")
    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def export_zones_to_shapefile(gdf, filename="prescription_zones",
                              product_name="Rate", field_name="",
                              farm_name="", client_name=""):
    """
    Export zone GeoDataFrame to prescription shapefile ZIP.

    Matches the format JD Operations Center expects:
    - Single numeric field named after the product (the rate value)
    - One simple polygon per row (multi-part polygons exploded)
    - .txt metadata file with field/product info
    - No .cpg file

    Args:
        gdf: GeoDataFrame with zone polygons (must have 'Rate' column)
        filename: base filename (without extension)
        product_name: name for the rate field (e.g. "BaitRate", "DAP")
        field_name: field name for metadata
        farm_name: farm name for metadata
        client_name: client/grower name for metadata

    Returns: bytes of ZIP file containing shapefile + metadata
    """
    if gdf is None or len(gdf) == 0:
        return None

    import tempfile
    import os
    from datetime import datetime

    # Sanitize product_name for DBF field (max 10 chars, no special chars)
    field_col = product_name[:10].replace(" ", "_")
    if not field_col:
        field_col = "Rate"

    # Prepare export: keep Zone column for dissolve, rate column for output
    # Exclude Rate=0 polygons — JD treats uncovered area as no-application
    export_gdf = gdf[['geometry', 'Zone', 'Rate']].copy()
    export_gdf = export_gdf[export_gdf['Rate'] > 0].reset_index(drop=True)
    export_gdf = export_gdf.rename(columns={'Rate': field_col})
    export_gdf[field_col] = export_gdf[field_col].astype(float)

    # Explode MultiPolygons into individual Polygon records
    export_gdf = export_gdf.explode(index_parts=False).reset_index(drop=True)

    # Remove holes from polygons — JD expects simple 1-part polygons
    from shapely.geometry import Polygon as ShapelyPolygon
    export_gdf['geometry'] = export_gdf['geometry'].apply(
        lambda g: ShapelyPolygon(g.exterior) if hasattr(g, 'exterior') else g
    )

    # Merge nearby same-zone fragments into larger zones.
    # Dissolve by Zone (not rate) so zones with the same rate stay separate
    # and don't swallow adjacent zones during the buffer step.
    export_projected = export_gdf.to_crs(epsg=3857)
    merge_buf = 3.0  # metres — closes small gaps between same-zone fragments
    export_projected['geometry'] = export_projected.geometry.buffer(merge_buf)
    export_projected = export_projected.dissolve(by='Zone', as_index=False)
    export_projected['geometry'] = export_projected.geometry.buffer(-merge_buf)
    # Remove any geometries that vanished after negative buffer
    export_projected = export_projected[~export_projected.is_empty].reset_index(drop=True)

    # Make zones non-overlapping so JD shows all rate levels.
    # Higher zones take priority — subtract them from lower zones.
    export_projected = export_projected.sort_values('Zone', ascending=False).reset_index(drop=True)
    for i in range(1, len(export_projected)):
        higher_union = unary_union(export_projected.geometry.iloc[:i].tolist())
        diff = export_projected.geometry.iloc[i].difference(higher_union)
        export_projected.at[i, 'geometry'] = diff
    export_projected = export_projected[~export_projected.is_empty].reset_index(drop=True)

    # Drop Zone column — only rate column needed for JD
    export_projected = export_projected.drop(columns=['Zone'])

    export_gdf = export_projected.to_crs(epsg=4326)
    export_gdf = export_gdf.explode(index_parts=False).reset_index(drop=True)

    # Filter out fragments smaller than 0.01 ha (100 m²)
    export_projected = export_gdf.to_crs(epsg=3857)
    min_area_m2 = 100  # 0.01 ha
    keep_mask = export_projected.geometry.area >= min_area_m2
    export_gdf = export_gdf[keep_mask].reset_index(drop=True)

    # If still over 500 features, progressively simplify and merge
    if len(export_gdf) > 500:
        # Aggressive buffer-dissolve-unbuffer to merge nearby fragments
        export_projected = export_gdf.to_crs(epsg=3857)
        export_projected['geometry'] = export_projected.geometry.buffer(5.0)
        export_projected = export_projected.dissolve(by=field_col, as_index=False)
        export_projected['geometry'] = export_projected.geometry.buffer(-5.0)
        export_projected = export_projected[~export_projected.is_empty].reset_index(drop=True)
        export_gdf = export_projected.to_crs(epsg=4326)
        # Simplify geometry more aggressively
        export_gdf['geometry'] = export_gdf['geometry'].simplify(0.0001, preserve_topology=True)
        export_gdf = export_gdf.explode(index_parts=False).reset_index(drop=True)
        # Remove holes again after simplification
        export_gdf['geometry'] = export_gdf['geometry'].apply(
            lambda g: ShapelyPolygon(g.exterior) if hasattr(g, 'exterior') else g
        )
        # Filter small fragments again with higher threshold
        export_projected = export_gdf.to_crs(epsg=3857)
        min_area_m2 = 500  # 0.05 ha
        keep_mask = export_projected.geometry.area >= min_area_m2
        export_gdf = export_gdf[keep_mask].reset_index(drop=True)

    # Calculate stats for metadata
    rates = export_gdf[field_col].values
    total_area = sum(g.area for g in export_gdf.geometry)  # degrees², approximate
    rate_min = float(rates.min())
    rate_max = float(rates.max())
    rate_mean = float(rates.mean())

    # Approximate area in hectares from WGS84 polygons
    # Use the zone areas from the original GDF
    total_area_ha = float(gdf['Area_ha'].sum()) if 'Area_ha' in gdf.columns else 0
    total_product = sum(
        float(row['Rate']) * float(row['Area_ha'])
        for _, row in gdf.iterrows()
    ) if 'Area_ha' in gdf.columns else 0

    # Build .txt metadata (matches PCT prescription format)
    now = datetime.now()
    year = now.year
    meta_lines = [
        f"{field_name} Rx {product_name} {year}",
        "",
        f"Grower:\t{client_name}",
        f"Farm:\t{farm_name}",
        f"Field:\t{field_name}",
        f"Season:\t{year}",
        f"Layer:\tRx Solid",
        f"Product:\t{product_name}",
        f"Crop:\t",
        "",
        f"Field Area:\t{total_area_ha:.2f} ha",
        "",
        f"Layer Area:\t{total_area_ha:.2f} ha",
        f"Range :\t{rate_min:.2f}~{rate_max:.2f} kg/ha",
        f"Mean :\t{rate_mean:.2f} kg/ha",
        f"Total :\t{total_product:.2f} kg",
        "",
        f"Create on {now.strftime('%d/%m/%Y %I:%M:%S %p')}",
        f"by SnailCam Visualizer",
        "",
    ]
    txt_content = "\r\n".join(meta_lines)

    # Create ZIP
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, f"{filename}.shp")
            export_gdf.to_file(shp_path, driver='ESRI Shapefile')

            # Add shapefile components (skip .cpg to match working format)
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                filepath = os.path.join(tmpdir, f"{filename}{ext}")
                if os.path.exists(filepath):
                    zf.write(filepath, f"{filename}{ext}")

            # Add .txt metadata
            zf.writestr(f"{filename}.txt", txt_content)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_jd_operations_center(gdf, snail_type, filename="snail_data", field_name="Unknown"):
    """
    Export data in John Deere Operations Center compatible format.
    Note: This format (doc/ + metadata) is for reference only — JD Files page
    does not accept work record uploads. Use export_zones_to_shapefile instead.
    """
    import json
    from datetime import datetime

    if gdf is None or len(gdf) == 0:
        return None

    export_gdf = gdf.copy()
    if export_gdf.crs and export_gdf.crs.to_epsg() != 4326:
        export_gdf = export_gdf.to_crs("EPSG:4326")

    export_cols = {'geometry': 'geometry'}
    if 'Conical' in export_gdf.columns:
        export_cols['Conical'] = 'Conical'
    if 'Italian' in export_gdf.columns:
        export_cols['Italian'] = 'Italian'
    if 'Total' in export_gdf.columns:
        export_cols['Total'] = 'Total'

    export_gdf = export_gdf[[c for c in export_cols.keys() if c in export_gdf.columns]].copy()

    now = datetime.utcnow().isoformat() + "Z"
    metadata = {
        "Version": "1.0",
        "OrgId": 0,
        "ClientId": "",
        "ClientName": "",
        "FarmId": "",
        "FarmName": "",
        "FieldId": "",
        "FieldName": field_name,
        "FieldOperationId": "",
        "Operation": "Harvest",
        "CropSeason": datetime.now().year,
        "CropToken": "",
        "CropName": "",
        "Work": {
            "WorkPlanId": "00000000-0000-0000-0000-000000000000",
            "WorkOrder": ""
        },
        "Product": {
            "ProductName": "",
            "BrandName": "",
            "TankMix": False
        },
        "FieldOperationStartDate": now,
        "FileCreatedTimeStamp": now,
        "DataAttributes": []
    }

    if 'Conical' in export_gdf.columns:
        metadata["DataAttributes"].append({
            "Name": "Conical", "Unit": "count", "Description": "Conical Snail Count"
        })
    if 'Italian' in export_gdf.columns:
        metadata["DataAttributes"].append({
            "Name": "Italian", "Unit": "count", "Description": "Italian Snail Count"
        })
    if 'Total' in export_gdf.columns:
        metadata["DataAttributes"].append({
            "Name": "Total", "Unit": "count", "Description": "Total Snail Count"
        })

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            doc_dir = os.path.join(tmpdir, "doc")
            os.makedirs(doc_dir)
            shp_path = os.path.join(doc_dir, f"{filename}.shp")
            export_gdf.to_file(shp_path, driver='ESRI Shapefile')
            meta_path = os.path.join(doc_dir, f"{filename}-Deere-Metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                filepath = os.path.join(doc_dir, f"{filename}{ext}")
                if os.path.exists(filepath):
                    zf.write(filepath, f"doc/{filename}{ext}")
            zf.write(meta_path, f"doc/{filename}-Deere-Metadata.json")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_points_to_shapefile(gdf, snail_type, filename="snail_data"):
    """Export detection points with snail counts as shapefile."""
    if gdf is None or len(gdf) == 0:
        return None

    export_cols = ['geometry']
    for col in ['Conical', 'Italian', 'Total']:
        if col in gdf.columns and col not in export_cols:
            export_cols.append(col)

    export_gdf = gdf[export_cols].copy()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            shp_path = os.path.join(tmpdir, f"{filename}.shp")
            export_gdf.to_file(shp_path, driver='ESRI Shapefile')
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                filepath = os.path.join(tmpdir, f"{filename}{ext}")
                if os.path.exists(filepath):
                    zf.write(filepath, f"{filename}{ext}")

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
