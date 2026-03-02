# Data Splitter — assign CSV data to field boundaries
# Run with: streamlit run data_splitter.py

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import io
import zipfile
from data_loader import (
    load_boundaries_from_zip, detect_file_type, validate_dataframe,
    aggregate_detections, filter_by_confidence, repair_geometry
)

st.set_page_config(page_title="Data Splitter", layout="wide")
st.title("Data Splitter")
st.caption("Assign snail data to field boundaries and export clean CSVs for SnailViz")

# --- Sidebar: file uploads ---
with st.sidebar:
    st.header("1. Upload CSV Files")
    csv_files = st.file_uploader(
        "Upload all snail data CSVs",
        type=["csv"], accept_multiple_files=True,
        help="Upload all your CSV files — data will be combined and sorted into boundaries"
    )

    st.header("2. Upload Boundaries")
    boundary_zip = st.file_uploader(
        "Upload boundary shapefiles (ZIP)",
        type=["zip"],
        help="ZIP containing shapefiles (.shp + .dbf + .prj etc.) for all fields"
    )

    st.divider()
    min_confidence = st.slider("Min confidence (detection files)", 0.0, 1.0, 0.3, 0.05)
    max_confidence = st.slider("Max confidence (detection files)", 0.0, 1.0, 1.0, 0.05)

if not csv_files or not boundary_zip:
    st.info("Upload CSV files and a boundary ZIP in the sidebar to begin.")
    st.stop()

# --- Load all CSVs ---
all_dfs = []
file_summary = []

for f in csv_files:
    try:
        df = pd.read_csv(f)
        file_type = detect_file_type(df)
        if file_type is None:
            st.warning(f"Skipped **{f.name}** — missing required columns")
            continue

        valid, err = validate_dataframe(df, file_type)
        if not valid:
            st.warning(f"Skipped **{f.name}** — {err}")
            continue

        # Convert detection files to harvest format
        if file_type == "detections":
            df = filter_by_confidence(df, min_confidence, max_confidence)
            df = aggregate_detections(df)

        # Ensure Total column
        if "Total" not in df.columns:
            df["Total"] = df.get("Conical", 0) + df.get("Italian", 0)

        file_summary.append({
            "File": f.name,
            "Type": file_type,
            "Points": len(df),
        })
        all_dfs.append(df)
    except Exception as e:
        st.warning(f"Error reading **{f.name}**: {e}")

if not all_dfs:
    st.error("No valid CSV files loaded.")
    st.stop()

# Combine all data into one DataFrame
combined_df = pd.concat(all_dfs, ignore_index=True)

# --- Load boundaries ---
boundaries_gdf = load_boundaries_from_zip(boundary_zip)
if boundaries_gdf is None or len(boundaries_gdf) == 0:
    st.error("No boundaries found in ZIP file.")
    st.stop()

boundaries_gdf = repair_geometry(boundaries_gdf)

# --- Show upload summary ---
st.header("Upload Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CSV Files Loaded", len(all_dfs))
with col2:
    st.metric("Total Data Points", f"{len(combined_df):,}")
with col3:
    st.metric("Boundaries Found", len(boundaries_gdf))

with st.expander("File details"):
    st.table(pd.DataFrame(file_summary))

# --- Spatial matching ---
st.header("Assigning Points to Boundaries")

with st.spinner("Matching points to boundaries..."):
    # Create GeoDataFrame from all points
    points_gdf = gpd.GeoDataFrame(
        combined_df,
        geometry=gpd.points_from_xy(combined_df.Longitude, combined_df.Latitude),
        crs="EPSG:4326"
    )

    # Ensure same CRS
    if points_gdf.crs != boundaries_gdf.crs:
        points_gdf = points_gdf.to_crs(boundaries_gdf.crs)

    # Assign each point to a boundary
    assigned = np.full(len(points_gdf), -1, dtype=int)

    for idx, boundary in boundaries_gdf.iterrows():
        mask = points_gdf.within(boundary.geometry)
        # Only assign if not already assigned (first boundary wins)
        new_assignments = mask & (assigned == -1)
        assigned[new_assignments.values] = idx

    # Build results per boundary
    results = []
    for idx, boundary in boundaries_gdf.iterrows():
        point_mask = assigned == idx
        count = point_mask.sum()

        boundary_name = boundary.get('boundary_name',
                        boundary.get('FIELD_NAME', f'Field_{idx}'))

        # Calculate area
        b_gdf = gpd.GeoDataFrame([boundary], crs=boundaries_gdf.crs)
        b_gdf = repair_geometry(b_gdf)
        try:
            area_ha = b_gdf.to_crs(epsg=3857).unary_union.area / 10000
        except Exception:
            area_ha = 0

        # Extract matching points as DataFrame
        if count > 0:
            matched_df = combined_df.loc[point_mask].copy()
        else:
            matched_df = pd.DataFrame(columns=combined_df.columns)

        results.append({
            'name': boundary_name,
            'count': int(count),
            'area_ha': area_ha,
            'df': matched_df,
            'boundary': boundary.geometry,
        })

    # Unmatched points
    unmatched_mask = assigned == -1
    unmatched_count = unmatched_mask.sum()

# --- Results table ---
st.header("Results")

summary_data = []
for r in results:
    summary_data.append({
        "Field": r['name'],
        "Points": r['count'],
        "Area (ha)": f"{r['area_ha']:.2f}",
        "Conical": int(r['df']['Conical'].sum()) if len(r['df']) > 0 else 0,
        "Italian": int(r['df']['Italian'].sum()) if len(r['df']) > 0 else 0,
    })

summary_data.append({
    "Field": "Unmatched",
    "Points": int(unmatched_count),
    "Area (ha)": "—",
    "Conical": int(combined_df.loc[unmatched_mask, 'Conical'].sum()) if unmatched_count > 0 else 0,
    "Italian": int(combined_df.loc[unmatched_mask, 'Italian'].sum()) if unmatched_count > 0 else 0,
})

st.table(pd.DataFrame(summary_data))

if unmatched_count > 0:
    st.warning(f"{unmatched_count:,} points did not match any boundary.")

# --- Map preview ---
st.header("Map Preview")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # Plot boundaries
    boundaries_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)

    # Label boundaries
    for idx, boundary in boundaries_gdf.iterrows():
        name = boundary.get('boundary_name', f'Field_{idx}')
        centroid = boundary.geometry.centroid
        ax.text(centroid.x, centroid.y, name, ha='center', va='center',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Color each boundary's points differently
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        if r['count'] == 0:
            continue
        color = colors[i % len(colors)]
        ax.scatter(
            r['df']['Longitude'], r['df']['Latitude'],
            s=5, c=[color], label=f"{r['name']} ({r['count']})", alpha=0.6
        )

    # Plot unmatched in gray
    if unmatched_count > 0:
        unmatched_df = combined_df.loc[unmatched_mask]
        ax.scatter(
            unmatched_df['Longitude'], unmatched_df['Latitude'],
            s=5, c='gray', label=f"Unmatched ({unmatched_count})", alpha=0.3
        )

    ax.legend(loc='best', fontsize=7, markerscale=3)
    ax.set_title("Data Points by Boundary Assignment")
    ax.axis('off')
    plt.tight_layout()

    st.pyplot(fig, width="stretch")
    plt.close(fig)
except Exception as e:
    st.caption(f"Could not render preview: {e}")

# --- Downloads ---
st.header("Download Clean CSVs")

# Build ZIP of all CSVs
zip_buffer = io.BytesIO()
export_cols = ["Longitude", "Latitude", "Conical", "Italian"]

with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
    for r in results:
        if r['count'] == 0:
            continue
        # Keep only SnailViz-compatible columns
        out_df = r['df'][export_cols].copy()
        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
        safe_name = r['name'].replace(' ', '_').replace('/', '_')
        zf.writestr(f"{safe_name}.csv", csv_bytes)

zip_buffer.seek(0)

st.download_button(
    label="Download All CSVs (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="split_data.zip",
    mime="application/zip"
)

# Individual downloads
st.subheader("Individual Files")
cols = st.columns(min(len(results), 4))

for i, r in enumerate(results):
    if r['count'] == 0:
        continue
    with cols[i % len(cols)]:
        out_df = r['df'][export_cols].copy()
        csv_bytes = out_df.to_csv(index=False).encode('utf-8')
        safe_name = r['name'].replace(' ', '_').replace('/', '_')
        st.download_button(
            label=f"{r['name']} ({r['count']} pts)",
            data=csv_bytes,
            file_name=f"{safe_name}.csv",
            mime="text/csv",
            key=f"dl_{i}"
        )
