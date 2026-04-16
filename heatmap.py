# Heatmap interpolation module

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from config import HEATMAP_COLORS
import io


def get_colormap_and_norm(min_count, max_count):
    """Create a custom colormap for snail density visualization."""
    cmap = mcolors.LinearSegmentedColormap.from_list("snail_density", HEATMAP_COLORS)
    norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
    return cmap, norm


def interpolate_heatmap(gdf_m, snail_type, pixel_size, search_radius,
                        extrapolation_limit, smoothing, apply_blur=False, blur_sigma=1.0):
    """
    Perform spatial interpolation to create a heatmap grid.

    Uses inverse distance weighting with exponential decay.

    Returns: (grid_z, grid_x, grid_y, bounds)
    """
    points = np.array([(geom.x, geom.y) for geom in gdf_m.geometry])
    values = gdf_m[snail_type].values

    minx, miny, maxx, maxy = gdf_m.total_bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)

    # Ensure minimum grid size
    width = max(width, 10)
    height = max(height, 10)

    xi = np.linspace(minx, maxx, width)
    yi = np.linspace(miny, maxy, height)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Build KDTree for efficient nearest neighbor queries
    tree = cKDTree(points)
    dists, idxs = tree.query(grid_coords, distance_upper_bound=search_radius, k=8)

    # Interpolate using exponential decay weighting
    grid_z = np.full(grid_coords.shape[0], np.nan)
    for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
        valid = np.isfinite(dist_row) & (dist_row < extrapolation_limit)
        if np.any(valid):
            w = np.exp(-dist_row[valid] / smoothing)
            grid_z[i] = np.sum(w * values[idx_row[valid]]) / np.sum(w)

    grid_z = grid_z.reshape(grid_x.shape)

    # Apply Gaussian blur if requested
    if apply_blur:
        mask = np.isnan(grid_z)
        grid_z_filled = np.nan_to_num(grid_z, nan=0)
        grid_z_blurred = gaussian_filter(grid_z_filled, sigma=blur_sigma)
        grid_z = np.where(mask, np.nan, grid_z_blurred)

    bounds = (minx, maxx, miny, maxy)
    return grid_z, grid_x, grid_y, bounds


def render_heatmap_figure(grid_z, bounds, cmap, norm, title, zone_rgb=None,
                          zone_map=None, zone_names=None, zone_colors=None,
                          pixel_size=3, thresholds=None, zone_rates=None,
                          snail_type=None, hide_field_name=False,
                          legend_entries=None):
    """
    Render the heatmap and zone map side by side.

    Args:
        hide_field_name: If True, only show snail type in title (no field/client name)
        legend_entries: list of (color_tuple, label_string) for zone legend.
                        If provided, overrides the auto-generated legend.

    Returns: matplotlib figure
    """
    minx, maxx, miny, maxy = bounds

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    # Configure colormap for NaN values
    cmap_copy = cmap.copy()
    cmap_copy.set_bad('white')

    # Build title based on hide_field_name option
    if hide_field_name:
        if snail_type and snail_type != "Total":
            heatmap_title = f"{snail_type} Snail Density"
        elif snail_type == "Total":
            heatmap_title = "All Snails (Combined) Density"
        else:
            heatmap_title = "Snail Density Heatmap"
    else:
        if snail_type and snail_type != "Total":
            heatmap_title = f"{snail_type} Snail Density\n{title}"
        elif snail_type == "Total":
            heatmap_title = f"All Snails (Combined) Density\n{title}"
        else:
            heatmap_title = f"Snail Density Heatmap\n{title}"

    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(grid_z, extent=(minx, maxx, miny, maxy), origin='lower',
                    cmap=cmap_copy, norm=norm)
    ax1.set_title(heatmap_title, fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('Snails per sample point', fontsize=10)

    # Zone map
    ax2 = axes[1]
    if zone_rgb is not None:
        ax2.imshow(zone_rgb, extent=(minx, maxx, miny, maxy), origin='lower')
        ax2.set_title("Treatment Zones", fontsize=12, fontweight='bold')

        # Add zone legend
        from matplotlib.patches import Patch
        if legend_entries:
            # Use pre-built legend entries (color, label) tuples
            legend_elements = [Patch(facecolor=c, label=l) for c, l in legend_entries]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=7)
        elif zone_names and zone_colors and thresholds:
            # Fallback to auto-generated legend
            legend_elements = []
            zone_count = len(zone_names)
            for i in range(zone_count):
                if i == 0:
                    label = f"{zone_names[i]}: 0 - {thresholds[0]:.1f}"
                elif i == zone_count - 1:
                    label = f"{zone_names[i]}: > {thresholds[-1]:.1f}"
                else:
                    label = f"{zone_names[i]}: {thresholds[i-1]:.1f} - {thresholds[i]:.1f}"

                if zone_rates:
                    label += f" ({zone_rates[i]} kg/ha)"

                legend_elements.append(Patch(facecolor=zone_colors[i+1], label=label))

            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Configure zones in sidebar", ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
    ax2.axis('off')

    plt.tight_layout()
    return fig


def mask_grid_to_boundary(grid_z, bounds, boundary_gdf):
    """
    Mask grid values outside the boundary polygon to NaN.

    Args:
        grid_z: 2D numpy array
        bounds: (minx, maxx, miny, maxy) in the same CRS as boundary
        boundary_gdf: GeoDataFrame with boundary polygon (should be in EPSG:3857)

    Returns: masked grid_z
    """
    if boundary_gdf is None:
        return grid_z

    import shapely

    minx, maxx, miny, maxy = bounds
    height, width = grid_z.shape

    # Get boundary polygon
    boundary_union = boundary_gdf.to_crs(epsg=3857).unary_union

    # Vectorized point-in-polygon check
    x_coords = np.linspace(minx, maxx, width)
    y_coords = np.linspace(miny, maxy, height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    flat_points = shapely.points(xx.ravel(), yy.ravel())
    mask = shapely.contains(boundary_union, flat_points).reshape(grid_z.shape)

    # Apply mask - set outside values to NaN
    masked_grid = grid_z.copy()
    masked_grid[~mask] = np.nan

    return masked_grid


def render_heatmap_only(grid_z, bounds, cmap, norm, title, snail_type=None, boundary_gdf=None, hide_field_name=False):
    """
    Render just the heatmap without zone map (for batch export).

    Args:
        hide_field_name: If True, only show snail type in title (no field/client name)

    Returns: matplotlib figure
    """
    minx, maxx, miny, maxy = bounds

    # Calculate aspect ratio
    width = maxx - minx
    height = maxy - miny
    aspect = width / height if height > 0 else 1

    # Create single-panel figure with appropriate size
    fig_width = 10
    fig_height = fig_width / aspect
    fig_height = max(6, min(fig_height, 12))  # Clamp between 6-12 inches

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Configure colormap for NaN values
    cmap_copy = cmap.copy()
    cmap_copy.set_bad('white')

    # Render heatmap
    im = ax.imshow(grid_z, extent=(minx, maxx, miny, maxy), origin='lower',
                   cmap=cmap_copy, norm=norm)

    # Draw boundary outline if provided
    if boundary_gdf is not None:
        try:
            boundary_m = boundary_gdf.to_crs(epsg=3857)
            for geom in boundary_m.geometry:
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='black', linewidth=2)
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color='black', linewidth=2)
        except Exception:
            pass

    # Build title with snail type label
    if hide_field_name:
        # Only show snail type, no field name
        if snail_type and snail_type != "Total":
            full_title = f"{snail_type} Snail Density"
        elif snail_type == "Total":
            full_title = "All Snails (Combined) Density"
        else:
            full_title = "Snail Density Heatmap"
    else:
        # Include field name in title
        if snail_type and snail_type != "Total":
            full_title = f"{snail_type} Snail Density\n{title}"
        elif snail_type == "Total":
            full_title = f"All Snails (Combined) Density\n{title}"
        else:
            full_title = f"Snail Density Heatmap\n{title}"

    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Snails per sample point', fontsize=11)

    plt.tight_layout()
    return fig


def render_zone_map_figure(zone_rgb, bounds, zone_stats, zone_names, zone_count,
                           legend_entries=None, title="Treatment Zones",
                           clear_threshold=0.1, hide_field_name=False):
    """
    Render only the zone map with zone area labels (in ha) for PNG export.

    Args:
        zone_rgb: 3D numpy array (height, width, 3) of zone colors
        bounds: (minx, maxx, miny, maxy) in Web Mercator
        zone_stats: list of zone stat dicts (from get_zone_statistics)
        zone_names: list of risk zone names
        zone_count: number of risk zones
        legend_entries: list of (color_tuple, label_string) for legend
        title: figure title
        clear_threshold: clear zone threshold
        hide_field_name: if True, use generic title

    Returns: matplotlib figure
    """
    from matplotlib.patches import Patch

    minx, maxx, miny, maxy = bounds

    # Calculate aspect ratio for the figure
    width = maxx - minx
    height = maxy - miny
    aspect = width / height if height > 0 else 1

    fig_width = 10
    fig_height = fig_width / aspect
    fig_height = max(6, min(fig_height, 12))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    ax.imshow(zone_rgb, extent=(minx, maxx, miny, maxy), origin='lower')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add area labels on each zone
    # Build labels: clear zone + risk zones
    label_items = []
    for stat in zone_stats:
        if stat['zone_type'] == 'clear':
            label_items.append((1, f"Clear\n{stat['area_ha']:.2f} ha"))
        else:
            risk_idx = stat['zone'] - 2
            name = zone_names[risk_idx] if risk_idx < len(zone_names) else f"Zone {risk_idx+1}"
            label_items.append((stat['zone'], f"{name}\n{stat['area_ha']:.2f} ha"))

    # Place labels at the centroid of each zone's largest region
    from skimage import measure
    zone_map_2d = np.zeros(zone_rgb.shape[:2], dtype=np.uint8)
    # Reconstruct zone_map from zone_rgb by matching colors
    # Instead, use a simpler approach: for each zone number, find pixels
    # We can derive zone numbers from zone_stats
    from config import CLEAR_ZONE_COLOR, RISK_ZONE_COLORS, NO_DATA_COLOR

    all_colors = [NO_DATA_COLOR, CLEAR_ZONE_COLOR]
    all_colors.extend(RISK_ZONE_COLORS.get(zone_count, RISK_ZONE_COLORS.get(3, [])))

    # Reconstruct zone_map from RGB by color matching
    for z, color in enumerate(all_colors):
        color_arr = np.array(color)
        match = np.all(np.abs(zone_rgb - color_arr) < 0.01, axis=2)
        zone_map_2d[match] = z

    h, w = zone_map_2d.shape
    for zone_num, label_text in label_items:
        mask = zone_map_2d == zone_num
        if not np.any(mask):
            continue

        # Find largest connected component
        labeled = measure.label(mask)
        props = measure.regionprops(labeled)
        if not props:
            continue

        largest = max(props, key=lambda r: r.area)
        cy, cx = largest.centroid  # row, col

        # Convert pixel coords to map coords
        map_x = minx + (cx / w) * (maxx - minx)
        map_y = miny + (cy / h) * (maxy - miny)

        ax.text(map_x, map_y, label_text, ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='gray'))

    # Add legend
    if legend_entries:
        legend_elements = [Patch(facecolor=c, label=l) for c, l in legend_entries]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def figure_to_bytes(fig, format='png', dpi=150):
    """Convert matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf.getvalue()
