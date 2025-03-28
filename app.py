# SnailCam Visualizer v1.1.15

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import leafmap.foliumap as leafmap
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.color import label2rgb
import folium
import streamlit.components.v1 as components
import os
import tempfile
import zipfile

st.set_page_config(page_title="SnailCam Visualizer", layout="wide")
st.title("🐌 SnailCam Data Viewer")
st.markdown("Upload SnailCam detection data and visualize snail counts with exponential decay IDW.")

def get_colormap_and_norm(min_count, max_count):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["green", "yellow", "orange", "red"])
    norm = mcolors.Normalize(vmin=min_count, vmax=max_count)
    return cmap, norm

with st.sidebar.expander("📁 Upload CSV", expanded=True):
    uploaded_file = st.file_uploader("Upload SnailCam CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    filename = os.path.splitext(uploaded_file.name)[0]

    required_columns = ["Longitude", "Latitude", "Conical", "Italian"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"Uploaded CSV must contain the following columns: {', '.join(required_columns)}")
        st.stop()

    mode = st.radio("Visualization Mode", ["Heatmap", "Detection Map"], horizontal=True)

    st.sidebar.header("Global Settings")
    snail_option = st.sidebar.radio("Snail Type", ["Conical", "Italian", "Both"], key="snail_type")
    if snail_option == "Both":
        df["Total"] = df["Conical"] + df["Italian"]
        snail_type = "Total"
    else:
        snail_type = snail_option

    min_count = st.sidebar.number_input("Minimum snail count (green)", value=0, key="min_count")
    max_count = st.sidebar.number_input("Maximum snail count (red)", value=6, key="max_count")
    cmap, norm = get_colormap_and_norm(min_count, max_count)

    filtered_df = df[(df[snail_type] >= min_count) & (df[snail_type] <= max_count)]
    gdf = gpd.GeoDataFrame(
        filtered_df,
        geometry=gpd.points_from_xy(filtered_df.Longitude, filtered_df.Latitude),
        crs="EPSG:4326"
    )
    gdf_m = gdf.to_crs(epsg=3857)

    if mode == "Heatmap":
        with st.sidebar.expander("Heatmap Settings", expanded=True):
            decay_constant = st.number_input("Exponential Decay Constant (λ)", value=0.23, key="decay")
            pixel_size = st.slider("Pixel size (m)", 1, 30, 3, key="pix")
            search_radius = st.slider("Search radius (m)", 5, 100, 30, key="rad")
            extrapolation_limit = st.slider("Extrapolation distance (m)", 5, 100, 30, key="ext")
            blur = st.checkbox("Apply Gaussian blur", value=False, key="blur")
            sigma = st.slider("Blur intensity (σ)", 0.5, 5.0, 1.0, key="sig")

        with st.sidebar.expander("Zoning Settings", expanded=False):
            zone_count = st.radio("Number of Zones", [2, 3], index=1)
            thresholds = []
            zone_names = []
            zone_rates = []
            for i in range(zone_count - 1):
                thresholds.append(st.number_input(f"Threshold {i+1} (max for Zone {i+1})", value=(i+1)*2.0, step=0.1))

            for i in range(zone_count):
                col = st.columns([2, 2, 2])
                with col[0]:
                    zone_names.append(st.text_input(f"Zone {i+1} Name", value=f"Zone {i+1}"))
                with col[1]:
                    zone_rates.append(st.number_input(f"{zone_names[-1]} Rate (kg/ha)", value=0))
            min_area = st.slider("Minimum zone area (pixels)", 0, 50, 0, step=1, key="min_area")
            download = st.sidebar.button("📥 Download Zone Shapefile")

        # === HEATMAP INTERPOLATION ===
        points = np.array([(geom.x, geom.y) for geom in gdf_m.geometry])
        values = gdf_m[snail_type].values

        minx, miny, maxx, maxy = gdf_m.total_bounds
        width = int((maxx - minx) / pixel_size)
        height = int((maxy - miny) / pixel_size)

        xi = np.linspace(minx, maxx, width)
        yi = np.linspace(miny, maxy, height)
        grid_x, grid_y = np.meshgrid(xi, yi)
        grid_coords = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        tree = cKDTree(points)
        dists, idxs = tree.query(grid_coords, distance_upper_bound=search_radius, k=8)

        grid_z = np.full(grid_coords.shape[0], np.nan)
        for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
            valid = np.isfinite(dist_row) & (dist_row < extrapolation_limit)
            if np.any(valid):
                w = np.exp(-dist_row[valid] / decay_constant)
                grid_z[i] = np.sum(w * values[idx_row[valid]]) / np.sum(w)

        grid_z = grid_z.reshape(grid_x.shape)
        if blur:
            mask = np.isnan(grid_z)
            grid_z[~mask] = gaussian_filter(grid_z[~mask], sigma=sigma)

        cmap.set_bad('white')

        # ZONE MAP
        zone_map = np.zeros_like(grid_z, dtype=np.uint8)
        if zone_count == 2:
            t1 = thresholds[0]
            zone_map[grid_z <= t1] = 1
            zone_map[grid_z > t1] = 2
        else:
            t1, t2 = thresholds
            zone_map[grid_z <= t1] = 1
            zone_map[(grid_z > t1) & (grid_z <= t2)] = 2
            zone_map[grid_z > t2] = 3

        if min_area > 0:
            for z in range(1, zone_count + 1):
                labeled = measure.label(zone_map == z)
                props = measure.regionprops(labeled)
                for region in props:
                    if region.area < min_area:
                        zone_map[labeled == region.label] = 1

        zone_colors = [
            (0.85, 0.85, 0.85),
            (0.0, 0.8, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ]

        # Initialize background as light gray
        zone_rgb = np.ones((*zone_map.shape, 3)) * 0.95  # light gray background

        # Overwrite with zone colors
        for z in range(1, zone_count + 1):
            zone_rgb[zone_map == z] = zone_colors[z]


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.3), dpi=70)
        ax1.imshow(grid_z, extent=(minx, maxx, miny, maxy), origin='lower', cmap=cmap, norm=norm)
        ax1.set_title(f"Heatmap: {filename}")
        ax1.axis('off')

        ax2.imshow(zone_rgb, extent=(minx, maxx, miny, maxy), origin='lower')
        ax2.set_title("Zone Map")
        ax2.axis('off')

        st.pyplot(fig, use_container_width=False)

        # Show zone areas with correct color patches
        st.subheader("Zone Areas")
        ha_per_pixel = (pixel_size ** 2) / 10000
        for z in range(1, zone_count + 1):
            area_ha = np.sum(zone_map == z) * ha_per_pixel
            color_hex = mcolors.to_hex(zone_colors[z])
            st.markdown(
                f"<div style='display:flex;align-items:center;'>"
                f"<div style='width:16px;height:16px;background:{color_hex};border:1px solid #ccc;margin-right:8px;'></div>"
                f"<b>{zone_names[z - 1]}</b> — ≥ {thresholds[z - 1] if z-1 < len(thresholds) else thresholds[-1]} — {zone_rates[z - 1]} kg/ha — <i>{area_ha:.2f} ha</i>"
                f"</div>",
                unsafe_allow_html=True
            )

        if download:
            polys = []
            zones = []
            names = []
            rates = []
            for z in range(1, zone_count + 1):
                mask = zone_map == z
                contours = measure.find_contours(mask, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    coords = np.array([
                        (grid_x[int(y), int(x)], grid_y[int(y), int(x)])
                        for x, y in contour if 0 <= int(x) < width and 0 <= int(y) < height
                    ])
                    if len(coords) > 2:
                        poly = Polygon(coords)
                        if poly.is_valid:
                            polys.append(poly)
                            zones.append(z)
                            names.append(zone_names[z-1])
                            rates.append(zone_rates[z-1])

            shp_gdf = gpd.GeoDataFrame({
                "Zone": zones,
                "Name": names,
                "Rate_kg_ha": rates,
                "geometry": polys
            }, crs="EPSG:3857")

            with tempfile.TemporaryDirectory() as tmpdir:
                shapefile_path = os.path.join(tmpdir, "zones.shp")
                shp_gdf.to_file(shapefile_path)
                zip_path = os.path.join(tmpdir, "zones.zip")
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for ext in [".shp", ".shx", ".dbf", ".prj"]:
                        filepath = os.path.join(tmpdir, "zones" + ext)
                        if os.path.exists(filepath):
                            zf.write(filepath, arcname="zones" + ext)
                with open(zip_path, "rb") as f:
                    st.sidebar.download_button("📦 Download Zoning Shapefile", f.read(), file_name="zones.zip")

    elif mode == "Detection Map":
        with st.sidebar.expander("Detection Map Settings", expanded=True):
            marker_radius = st.slider("Detection marker radius (m)", 1, 20, 3, key="radius")
            zoom_level = st.slider("Map Zoom Level", 10, 20, 17, key="zoom")

        gdf_latlon = gdf.to_crs(epsg=4326)
        gdf_latlon["lon"] = gdf_latlon.geometry.x
        gdf_latlon["lat"] = gdf_latlon.geometry.y

        center = [gdf_latlon["lat"].mean(), gdf_latlon["lon"].mean()]
        m = leafmap.Map(center=center, zoom=zoom_level)
        gdf_latlon["color"] = gdf_latlon[snail_type].apply(lambda x: mcolors.to_hex(cmap(norm(x))))

        for _, row in gdf_latlon.iterrows():
            folium.Circle(
                location=(row["lat"], row["lon"]),
                radius=marker_radius,
                color=row["color"],
                fill=True,
                fill_color=row["color"],
                fill_opacity=0.8,
                popup=str(row[snail_type]),
                weight=1,
                opacity=1
            ).add_to(m)

        components.html(m.to_html(), height=650)

else:
    st.info("Upload a CSV file to begin.")