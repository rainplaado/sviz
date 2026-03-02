# SnailCam Visualizer v2.0.0
# Refactored with improved UX for farmers

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import leafmap.foliumap as leafmap
import folium
import streamlit.components.v1 as components
import io
import zipfile

from config import (
    APP_VERSION, APP_TITLE,
    DEFAULT_SMOOTHING, DEFAULT_PIXEL_SIZE, DEFAULT_SEARCH_RADIUS,
    DEFAULT_EXTRAPOLATION, DEFAULT_BLUR_SIGMA, DEFAULT_APPLY_BLUR,
    DEFAULT_ZONE_COUNT, DEFAULT_THRESHOLDS, DEFAULT_MIN_AREA,
    DEFAULT_CLEAR_THRESHOLD,
    DEFAULT_MARKER_RADIUS, DEFAULT_ZOOM,
    DEFAULT_MIN_COUNT, DEFAULT_MAX_COUNT,
    DEFAULT_MIN_CONFIDENCE, DEFAULT_MAX_CONFIDENCE,
    ZONE_COLORS, CLEAR_ZONE_COLOR
)
from data_loader import (
    detect_file_type, validate_dataframe, load_and_prepare_data,
    create_geodataframe, calculate_field_statistics, group_files_by_location,
    load_boundaries_from_zip, match_points_to_boundary, clip_points_to_boundary,
    get_boundary_area_ha, split_points_by_boundaries
)
from heatmap import (
    get_colormap_and_norm, interpolate_heatmap,
    render_heatmap_figure, render_heatmap_only, render_zone_map_figure,
    figure_to_bytes
)
from zones import (
    classify_zones, filter_small_zones, create_zone_rgb,
    calculate_zone_areas, get_zone_statistics, get_zone_colors,
    zones_to_polygons, export_zones_to_shapefile, export_points_to_shapefile,
    export_jd_operations_center, fill_gaps, create_boundary_mask
)


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"🐌 {APP_TITLE}")
st.markdown(f"*Version {APP_VERSION}* — Upload snail detection data to visualize and plan treatments")


# Cache data loading for performance
@st.cache_data
def load_csv(uploaded_file):
    """Load and cache CSV data."""
    return pd.read_csv(uploaded_file)


def create_zip(files_dict):
    """Create a ZIP file from a dictionary of {filename: png_bytes}."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename, png_bytes in files_dict.items():
            zf.writestr(f"{filename}_heatmap.png", png_bytes)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# Sidebar - File Upload
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload SnailCam CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload one or more Harvest summary or Detections files"
    )

    st.header("🗺️ Field Boundaries (Optional)")
    boundary_file = st.file_uploader(
        "Upload boundaries ZIP",
        type="zip",
        help="Upload a ZIP with boundary shapefiles - data will be clipped to matching field"
    )

# Load boundaries if provided
boundaries_gdf = None
if boundary_file:
    try:
        boundaries_gdf = load_boundaries_from_zip(boundary_file)
        if boundaries_gdf is not None:
            st.sidebar.success(f"✓ Loaded {len(boundaries_gdf)} boundaries")
    except Exception as e:
        st.sidebar.error(f"Could not load boundaries: {e}")


if uploaded_files:
    # Determine if batch mode (multiple files)
    batch_mode = len(uploaded_files) > 1

    if batch_mode:
        # === BATCH MODE ===
        st.sidebar.success(f"✓ {len(uploaded_files)} files uploaded")

        # === BATCH SETTINGS IN SIDEBAR ===
        st.sidebar.header("⚙️ Batch Settings")

        snail_option = st.sidebar.radio(
            "Snail Type",
            ["Conical", "Italian", "Both", "All (separate)"],
            horizontal=True,
            help="'All (separate)' generates Conical + Italian as separate files"
        )

        st.sidebar.subheader("🔗 Field Merging")
        merge_options = ["Auto-detect", "Manual Select", "Keep Separate"]
        if boundaries_gdf is not None:
            merge_options.insert(0, "Split by Boundaries")
        merge_mode = st.sidebar.radio(
            "Merge Mode",
            merge_options,
            help="Split by Boundaries: auto-assign points to correct field. Auto: merge nearby. Separate: each file = 1 map"
        )

        st.sidebar.subheader("🎨 Color Scale")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_count = st.number_input("Min (green)", value=DEFAULT_MIN_COUNT, min_value=0)
        with col2:
            max_count = st.number_input("Max (red)", value=DEFAULT_MAX_COUNT, min_value=1)

        st.sidebar.subheader("🔒 Privacy")
        hide_field_name = st.sidebar.checkbox(
            "Hide field name from image",
            value=False,
            help="Remove field/client name from map title - useful for sharing with others"
        )

        with st.sidebar.expander("🗺️ Heatmap Settings", expanded=False):
            smoothing = st.slider("Smoothing intensity", 0.1, 1.0, DEFAULT_SMOOTHING, 0.05)
            pixel_size = st.slider("Detail level (m)", 1, 30, DEFAULT_PIXEL_SIZE)
            search_radius = st.slider("Search radius (m)", 5, 100, DEFAULT_SEARCH_RADIUS)
            extrapolation_limit = st.slider("Fill distance (m)", 5, 100, DEFAULT_EXTRAPOLATION)
            apply_blur = st.checkbox("Smooth edges", value=False)
            blur_sigma = st.slider("Edge smoothing", 0.5, 5.0, DEFAULT_BLUR_SIGMA) if apply_blur else DEFAULT_BLUR_SIGMA

        with st.sidebar.expander("🎯 Batch Zone Settings", expanded=False):
            st.caption("Zone settings applied to all fields in batch mode.")

            batch_zone_count = st.radio(
                "Number of risk zones", [1, 2, 3], index=2, horizontal=True,
                help="1 = Bait zone only. 2 = Low + High. 3 = Low + Medium + High.",
                key="batch_zone_count"
            )

            batch_clear_threshold = st.slider(
                "Clear threshold (snails/point)",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_CLEAR_THRESHOLD, step=0.05,
                key="batch_clear_thresh"
            )

            batch_thresholds = []
            for i in range(batch_zone_count - 1):
                batch_thresholds.append(st.number_input(
                    f"Threshold {i+1} (snails/point)",
                    value=DEFAULT_THRESHOLDS[i] if i < len(DEFAULT_THRESHOLDS) else (i+1)*0.5,
                    step=0.1, format="%.1f",
                    key=f"batch_thresh_{i}"
                ))

            st.divider()
            batch_default_names = (
                ["Bait Zone"] if batch_zone_count == 1
                else ["Low Risk", "High Risk"] if batch_zone_count == 2
                else ["Low Risk", "Medium Risk", "High Risk"]
            )

            batch_zone_names = []
            batch_zone_rates = []
            for i in range(batch_zone_count):
                cols = st.columns([3, 2])
                with cols[0]:
                    batch_zone_names.append(st.text_input(
                        f"Risk zone {i+1} name",
                        value=batch_default_names[i],
                        key=f"batch_zname_{i}"
                    ))
                with cols[1]:
                    batch_zone_rates.append(st.number_input(
                        f"Bait (kg/ha)",
                        value=0,
                        key=f"batch_zrate_{i}"
                    ))

            st.divider()
            batch_fill_gaps = st.checkbox("Fill gaps in coverage", value=True, key="batch_fill")
            batch_min_area = st.slider(
                "Min risk zone size (ha)",
                min_value=0.0, max_value=5.0,
                value=0.5, step=0.1,
                key="batch_min_area"
            )
            batch_rx_product = st.text_input(
                "Product name", value="BaitRate",
                help="Rate column name in prescription shapefile (max 10 chars)",
                key="batch_rx_product"
            )

        # Load and validate all files
        st.header("📂 Batch Processing")

        dataframes = []
        errors = []
        for uf in uploaded_files:
            try:
                df = load_csv(uf)
                filename = uf.name.rsplit('.', 1)[0]
                file_type = detect_file_type(df)
                is_valid, error_msg = validate_dataframe(df, file_type)
                if is_valid:
                    # Prepare data
                    prepared_df, snail_type = load_and_prepare_data(
                        df, file_type, snail_option, None, None
                    )
                    dataframes.append((prepared_df, filename))
                else:
                    errors.append(f"{uf.name}: {error_msg}")
            except Exception as e:
                errors.append(f"{uf.name}: {str(e)}")

        if errors:
            for err in errors:
                st.warning(f"⚠️ {err}")

        if not dataframes:
            st.error("No valid files to process.")
            st.stop()

        # Create groups based on merge mode
        if merge_mode == "Split by Boundaries" and boundaries_gdf is not None:
            # Combine all data and split by boundaries
            all_dfs = [df for df, _ in dataframes]
            combined_all = pd.concat(all_dfs, ignore_index=True)

            # Ensure Total column exists
            if "Total" not in combined_all.columns:
                combined_all["Total"] = combined_all.get("Conical", 0) + combined_all.get("Italian", 0)

            # Create GeoDataFrame from combined data
            combined_gdf = gpd.GeoDataFrame(
                combined_all,
                geometry=gpd.points_from_xy(combined_all.Longitude, combined_all.Latitude),
                crs="EPSG:4326"
            )

            # Split by boundaries
            boundary_splits = split_points_by_boundaries(combined_gdf, boundaries_gdf)

            if boundary_splits:
                groups = []
                for split in boundary_splits:
                    # Convert points GeoDataFrame back to regular DataFrame
                    points_df = pd.DataFrame(split['points_gdf'].drop(columns='geometry'))
                    groups.append({
                        'files': [split['boundary_name']],
                        'combined_df': points_df,
                        'boundary_gdf': split['boundary_gdf'],
                        'boundary_name': split['boundary_name'],
                        'area_ha': split['area_ha']
                    })
                st.success(f"✓ Split data into {len(groups)} fields based on boundaries")
            else:
                st.warning("No points fell within any boundary. Using auto-detect instead.")
                groups = group_files_by_location(dataframes)
        elif merge_mode == "Keep Separate":
            # Each file is its own group
            groups = [
                {'files': [filename], 'combined_df': df.copy()}
                for df, filename in dataframes
            ]
        elif merge_mode == "Auto-detect":
            # Auto-detect same-field files
            groups = group_files_by_location(dataframes)
        else:
            # Manual Select mode - will be handled with selection UI
            groups = group_files_by_location(dataframes)

        # === PREVIEW SECTION ===
        st.subheader("👁️ Preview All Fields")
        st.caption("Quick preview of each uploaded file to help you decide which to merge")

        # Generate quick previews for each file
        cmap_preview, norm_preview = get_colormap_and_norm(min_count, max_count)
        preview_cols = st.columns(min(4, len(dataframes)))

        # Determine snail type for preview
        if snail_option == "Both":
            preview_snail_type = "Total"
        elif snail_option == "All (separate)":
            preview_snail_type = "Conical"  # Just show Conical for preview
        else:
            preview_snail_type = snail_option

        preview_images = {}
        for idx, (df, filename) in enumerate(dataframes):
            col_idx = idx % 4
            with preview_cols[col_idx]:
                # Ensure Total column exists for "Both" option
                df_preview = df.copy()
                if preview_snail_type == "Total" and "Total" not in df_preview.columns:
                    df_preview["Total"] = df_preview.get("Conical", 0) + df_preview.get("Italian", 0)

                # Create quick preview heatmap
                gdf_preview, gdf_m_preview = create_geodataframe(df_preview, preview_snail_type, min_count, max_count + 100)
                if len(gdf_m_preview) > 0:
                    try:
                        grid_z, grid_x, grid_y, bounds = interpolate_heatmap(
                            gdf_m_preview, preview_snail_type,
                            pixel_size=10, search_radius=30, extrapolation_limit=30,
                            smoothing=0.3, apply_blur=False, blur_sigma=1.0
                        )
                        fig_preview = render_heatmap_only(grid_z, bounds, cmap_preview, norm_preview, filename, snail_type=preview_snail_type, hide_field_name=hide_field_name)
                        png_preview = figure_to_bytes(fig_preview, format='png', dpi=100)
                        preview_images[filename] = png_preview
                        st.image(png_preview, caption=filename, use_container_width=True)
                        plt.close(fig_preview)
                    except Exception as e:
                        st.caption(f"⚠️ {filename}")
                else:
                    st.caption(f"⚠️ {filename} (no data)")

        st.divider()

        # === MANUAL MERGE SELECTION ===
        if merge_mode == "Manual Select":
            st.subheader("🔗 Select Fields to Merge")
            st.caption("Check the boxes to group files together. Files in the same group will be merged into one map.")

            file_list = [filename for _, filename in dataframes]

            # Initialize merge groups in session state
            if 'merge_groups' not in st.session_state:
                st.session_state.merge_groups = {}

            # Create merge group selector
            num_groups = st.number_input("Number of output maps", min_value=1, max_value=len(dataframes), value=len(groups))

            merge_assignments = {}
            assign_cols = st.columns(min(3, len(dataframes)))

            for idx, filename in enumerate(file_list):
                col_idx = idx % 3
                with assign_cols[col_idx]:
                    group_num = st.selectbox(
                        f"📄 {filename}",
                        options=list(range(1, num_groups + 1)),
                        key=f"merge_{filename}",
                        help=f"Assign to output map group"
                    )
                    merge_assignments[filename] = group_num

            # Rebuild groups based on manual selection
            manual_groups = {}
            for filename, group_num in merge_assignments.items():
                if group_num not in manual_groups:
                    manual_groups[group_num] = {'files': [], 'combined_df': None}
                manual_groups[group_num]['files'].append(filename)

            # Combine dataframes for each group
            df_lookup = {filename: df for df, filename in dataframes}
            for group_num, group_data in manual_groups.items():
                dfs_to_merge = [df_lookup[f] for f in group_data['files']]
                group_data['combined_df'] = pd.concat(dfs_to_merge, ignore_index=True)

            groups = list(manual_groups.values())

            st.divider()

        # === DETECTED FIELDS SUMMARY ===
        st.subheader("🔍 Output Maps")
        for i, group in enumerate(groups):
            area_info = f" ({group['area_ha']:.1f} ha)" if 'area_ha' in group else ""
            if len(group['files']) > 1:
                st.info(f"**Map {i+1}**: {len(group['files'])} files merged → {', '.join(group['files'])}{area_info}")
            else:
                st.write(f"**Map {i+1}**: {group['files'][0]}{area_info}")

        st.write(f"Total: **{len(groups)} map(s)** from {len(dataframes)} files")
        st.divider()

        # Process button
        if st.button("🚀 Process All & Generate Heatmaps", type="primary"):
            cmap, norm = get_colormap_and_norm(min_count, max_count)
            png_dict = {}
            rx_dict = {}  # prescription ZIPs per field

            # Determine which snail types to process
            if snail_option == "All (separate)":
                snail_types_to_process = ["Conical", "Italian"]
            elif snail_option == "Both":
                snail_types_to_process = ["Total"]
            else:
                snail_types_to_process = [snail_option]

            total_steps = len(groups) * len(snail_types_to_process)
            progress = st.progress(0, text="Processing...")
            step = 0

            for idx, group in enumerate(groups):
                combined_df = group['combined_df']
                group_name = group.get('boundary_name', group['files'][0] if len(group['files']) == 1 else f"merged_{'+'.join(group['files'][:2])}")
                group_boundary = group.get('boundary_gdf', None)

                # Ensure Total column exists if needed
                if "Total" not in combined_df.columns:
                    combined_df["Total"] = combined_df.get("Conical", 0) + combined_df.get("Italian", 0)

                for snail_type in snail_types_to_process:
                    step += 1
                    progress.progress(step / total_steps, text=f"Processing {group_name} ({snail_type})...")

                    # Create GeoDataFrame
                    gdf, gdf_m = create_geodataframe(combined_df, snail_type, min_count, max_count)

                    if len(gdf) == 0:
                        st.warning(f"⚠️ No data for {group_name} ({snail_type})")
                        continue

                    # Generate heatmap
                    grid_z, grid_x, grid_y, bounds = interpolate_heatmap(
                        gdf_m, snail_type, pixel_size, search_radius,
                        extrapolation_limit, smoothing, apply_blur, blur_sigma
                    )

                    # Mask to boundary if available
                    if group_boundary is not None:
                        from heatmap import mask_grid_to_boundary
                        grid_z = mask_grid_to_boundary(grid_z, bounds, group_boundary)

                    # Fill gaps if enabled
                    if batch_fill_gaps:
                        boundary_mask = None
                        if group_boundary is not None:
                            boundary_mask = create_boundary_mask(grid_z.shape, bounds, group_boundary)
                        grid_z = fill_gaps(grid_z, max_fill_pixels=30, boundary_mask=boundary_mask)
                        if boundary_mask is not None:
                            grid_z = np.where(boundary_mask, grid_z, np.nan)

                    # Render heatmap-only figure
                    if snail_option == "All (separate)":
                        output_name = f"{group_name}_{snail_type}"
                    else:
                        output_name = group_name

                    fig = render_heatmap_only(grid_z, bounds, cmap, norm, group_name, snail_type=snail_type, boundary_gdf=group_boundary, hide_field_name=hide_field_name)
                    png_bytes = figure_to_bytes(fig, format='png', dpi=300)
                    png_dict[output_name] = png_bytes
                    plt.close(fig)

                    # Generate prescription shapefile
                    try:
                        zone_map = classify_zones(grid_z, batch_thresholds, batch_zone_count, batch_clear_threshold)
                        zone_map = filter_small_zones(zone_map, batch_min_area, batch_zone_count, pixel_size)

                        zone_gdf = zones_to_polygons(
                            zone_map, bounds, pixel_size, batch_zone_count,
                            batch_zone_names, batch_zone_rates, batch_thresholds, batch_clear_threshold
                        )

                        if zone_gdf is not None:
                            rx_field = group_name
                            rx_farm = ""
                            rx_client = ""
                            if 'boundary_gdf' in group and group['boundary_gdf'] is not None:
                                bg = group['boundary_gdf'].iloc[0] if len(group['boundary_gdf']) > 0 else None
                                if bg is not None:
                                    rx_farm = str(bg.get('FARM_NAME', '')) if 'FARM_NAME' in group['boundary_gdf'].columns else ""
                                    rx_client = str(bg.get('CLIENT_NAM', '')) if 'CLIENT_NAM' in group['boundary_gdf'].columns else ""

                            safe_name = output_name.replace(' ', '_').replace('/', '_')
                            shp_zip = export_zones_to_shapefile(
                                zone_gdf, f"{safe_name}_prescription",
                                product_name=batch_rx_product,
                                field_name=rx_field,
                                farm_name=rx_farm,
                                client_name=rx_client
                            )
                            if shp_zip:
                                rx_dict[output_name] = shp_zip
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate prescription for {output_name}: {e}")

            progress.empty()

            if png_dict:
                st.success(f"✅ Generated {len(png_dict)} heatmap(s) and {len(rx_dict)} prescription(s)!")

                # Create ZIP and download
                zip_bytes = create_zip(png_dict)
                st.download_button(
                    label=f"📥 Download All Heatmaps ({len(png_dict)} files) as ZIP",
                    data=zip_bytes,
                    file_name="snail_heatmaps.zip",
                    mime="application/zip",
                    type="primary"
                )

                # Prescription downloads
                if rx_dict:
                    rx_zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(rx_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for name, shp_bytes in rx_dict.items():
                            safe = name.replace(' ', '_').replace('/', '_')
                            zf.writestr(f"{safe}_prescription.zip", shp_bytes)
                    rx_zip_buffer.seek(0)

                    st.download_button(
                        label=f"📦 Download All Prescriptions ({len(rx_dict)} files) as ZIP",
                        data=rx_zip_buffer.getvalue(),
                        file_name="snail_prescriptions.zip",
                        mime="application/zip"
                    )

                # Preview first heatmap
                with st.expander("👀 Preview first heatmap"):
                    first_key = list(png_dict.keys())[0]
                    st.image(png_dict[first_key], caption=first_key)
            else:
                st.error("No heatmaps generated. Check your data.")

    else:
        # === SINGLE FILE MODE (original behavior) ===
        uploaded_file = uploaded_files[0]

        # Load data with caching
        df = load_csv(uploaded_file)
        filename = uploaded_file.name.rsplit('.', 1)[0]

        # Detect file type
        file_type = detect_file_type(df)
        is_valid, error_msg = validate_dataframe(df, file_type)

        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.stop()

        # Show file type detected
        file_type_label = "Detection Details" if file_type == "detections" else "Harvest Summary"
        st.sidebar.success(f"✓ Loaded: {file_type_label}")

        # === GLOBAL SETTINGS ===
        st.sidebar.header("⚙️ Settings")

        # Snail type selection
        snail_option = st.sidebar.radio(
            "Snail Type",
            ["Conical", "Italian", "Both"],
            horizontal=True,
            help="Choose which snail type to visualize"
        )

        # Confidence filtering (only for detection files)
        min_confidence = None
        max_confidence = None

        if file_type == "detections":
            st.sidebar.subheader("🎯 Confidence Filter")
            confidence_range = st.sidebar.slider(
                "Detection confidence range",
                min_value=0.0,
                max_value=1.0,
                value=(DEFAULT_MIN_CONFIDENCE, 1.0),
                step=0.05,
                help="Filter out low-confidence detections. Higher values = more certain detections only."
            )
            min_confidence = confidence_range[0]
            max_confidence = confidence_range[1]

            # Show filtered count
            original_count = len(df)
            filtered_df_preview = df[(df["Confidence"] >= min_confidence) & (df["Confidence"] <= max_confidence)]
            st.sidebar.caption(f"Showing {len(filtered_df_preview):,} of {original_count:,} detections")

        # Prepare data with filtering
        with st.spinner("Processing data..."):
            prepared_df, snail_type = load_and_prepare_data(
                df, file_type, snail_option, min_confidence, max_confidence
            )

        # Count range settings
        st.sidebar.subheader("🎨 Color Scale")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_count = st.number_input("Min (green)", value=DEFAULT_MIN_COUNT, min_value=0, key="single_min")
        with col2:
            max_count = st.number_input("Max (red)", value=DEFAULT_MAX_COUNT, min_value=1, key="single_max")

        st.sidebar.subheader("🔒 Privacy")
        hide_field_name = st.sidebar.checkbox(
            "Hide field name from image",
            value=False,
            help="Remove field/client name from map title - useful for sharing with others",
            key="single_hide_field"
        )

        cmap, norm = get_colormap_and_norm(min_count, max_count)

        # Create GeoDataFrames
        gdf, gdf_m = create_geodataframe(prepared_df, snail_type, min_count, max_count)

        if len(gdf) == 0:
            st.warning("⚠️ No data points match the current filter settings. Try adjusting the filters.")
            st.stop()

        # Match and clip to boundary if provided
        matched_boundary = None
        boundary_name = None
        boundary_area_ha = None

        if boundaries_gdf is not None:
            matched_boundary, boundary_name, match_pct = match_points_to_boundary(gdf, boundaries_gdf)
            if matched_boundary is not None:
                st.success(f"✓ Matched to boundary: **{boundary_name}** ({match_pct:.0f}% of points inside)")
                # Clip points to boundary
                gdf = clip_points_to_boundary(gdf, matched_boundary)
                gdf_m = gdf.to_crs(epsg=3857)
                boundary_area_ha = get_boundary_area_ha(matched_boundary)
            else:
                st.warning("⚠️ Could not match data to any boundary")

        # Calculate and display field statistics
        stats = calculate_field_statistics(gdf_m, prepared_df, snail_type)

        # Keep boundary area separate — total_area_ha will be updated
        # after zone processing to reflect actual mapped coverage
        if boundary_area_ha:
            stats['boundary_area_ha'] = boundary_area_ha

        # === FIELD SUMMARY ===
        st.header("📊 Field Summary")
        if boundary_name:
            st.caption(f"Field: {boundary_name}")
        summary_cols = st.columns(5)

        with summary_cols[0]:
            area_label = "Total Field Area"
            if boundary_area_ha:
                area_label = "Boundary Area"
            st.metric(area_label, f"{stats.get('boundary_area_ha', stats['total_area_ha']):.2f} ha")
        with summary_cols[1]:
            st.metric("Total Snails Detected", f"{stats['total_snails']:,}")
        with summary_cols[2]:
            st.metric("Avg Density", f"{stats['avg_density']:.1f} /ha")
        with summary_cols[3]:
            st.metric("Conical Snails", f"{stats['conical_total']:,}")
        with summary_cols[4]:
            st.metric("Italian Snails", f"{stats['italian_total']:,}")

        st.divider()

        # === VISUALIZATION MODE ===
        mode = st.radio(
            "📍 Visualization Mode",
            ["Heatmap", "Detection Map"],
            horizontal=True,
            help="Heatmap shows density patterns. Detection Map shows individual points."
        )

        if mode == "Heatmap":
            # === HEATMAP SETTINGS ===
            with st.sidebar.expander("🗺️ Heatmap Settings", expanded=True):
                smoothing = st.slider(
                    "Smoothing intensity",
                    min_value=0.1,
                    max_value=1.0,
                    value=DEFAULT_SMOOTHING,
                    step=0.05,
                    help="Higher values create smoother transitions between areas"
                )

                pixel_size = st.slider(
                    "Detail level (m)",
                    min_value=1,
                    max_value=30,
                    value=DEFAULT_PIXEL_SIZE,
                    help="Smaller = more detail but slower. Larger = faster but less detail."
                )

                search_radius = st.slider(
                    "Search radius (m)",
                    min_value=5,
                    max_value=100,
                    value=DEFAULT_SEARCH_RADIUS,
                    help="How far to look for nearby detections when creating the heatmap"
                )

                extrapolation_limit = st.slider(
                    "Fill distance (m)",
                    min_value=5,
                    max_value=100,
                    value=DEFAULT_EXTRAPOLATION,
                    help="Maximum distance to extend data beyond detection points"
                )

                apply_blur = st.checkbox("Smooth edges", value=DEFAULT_APPLY_BLUR)
                if apply_blur:
                    blur_sigma = st.slider("Edge smoothing", 0.5, 5.0, DEFAULT_BLUR_SIGMA)
                else:
                    blur_sigma = DEFAULT_BLUR_SIGMA

            # === ZONE SETTINGS ===
            with st.sidebar.expander("🎯 Treatment Zones", expanded=True):
                st.caption(
                    "The field is split into a **Clear** zone (no snails) "
                    "and **risk zones** where snails were found. "
                    "Adjusting thresholds only changes risk zones — "
                    "the clear area and field edges stay fixed."
                )

                zone_count = st.radio(
                    "Number of risk zones", [1, 2, 3], index=2, horizontal=True,
                    help="1 = Bait zone only. 2 = Low + High risk. 3 = Low + Medium + High risk."
                )

                thresholds = []
                zone_names = []
                zone_rates = []

                # Show data range to help set thresholds
                data_values = gdf_m[snail_type].dropna()

                st.divider()
                st.caption("**Clear zone** — areas with no snails detected:")
                clear_threshold = st.slider(
                    "Clear threshold (snails/point)",
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_CLEAR_THRESHOLD,
                    step=0.05,
                    help="Areas below this density are marked 'Clear' — no bait needed. "
                         "Increase to expand the clear zone, decrease to be more sensitive."
                )

                if len(data_values) > 0:
                    n_clear = int((data_values <= clear_threshold).sum())
                    n_with_snails = int((data_values > clear_threshold).sum())
                    st.caption(
                        f"📊 {n_clear} survey points clear, "
                        f"{n_with_snails} with snails "
                        f"(range: {data_values.min():.1f} - {data_values.max():.1f})"
                    )

                st.divider()
                st.caption("**Risk zone thresholds** (snails per survey point):")

                # Only consider positive-density values for threshold defaults
                if len(data_values) > 0:
                    positive_values = data_values[data_values > clear_threshold]
                else:
                    positive_values = pd.Series(dtype=float)

                for i in range(zone_count - 1):
                    if len(positive_values) > 0:
                        default_thresh = float(positive_values.quantile((i + 1) / zone_count))
                    else:
                        default_thresh = DEFAULT_THRESHOLDS[i] if i < len(DEFAULT_THRESHOLDS) else (i+1)*0.5

                    thresholds.append(st.number_input(
                        f"Threshold {i+1} (snails/point)",
                        value=default_thresh,
                        step=0.1,
                        format="%.1f",
                        help=f"Boundary between risk zone {i+1} and {i+2}"
                    ))

                st.divider()
                st.caption("**Risk zone names and bait rates:**")

                default_names = (
                    ["Bait Zone"] if zone_count == 1
                    else ["Low Risk", "High Risk"] if zone_count == 2
                    else ["Low Risk", "Medium Risk", "High Risk"]
                )

                for i in range(zone_count):
                    cols = st.columns([3, 2])
                    with cols[0]:
                        zone_names.append(st.text_input(
                            f"Risk zone {i+1} name",
                            value=default_names[i],
                            key=f"zname_{i}"
                        ))
                    with cols[1]:
                        zone_rates.append(st.number_input(
                            f"Bait (kg/ha)",
                            value=0,
                            key=f"zrate_{i}"
                        ))

                st.divider()
                st.caption("**Zone cleanup:**")

                fill_gaps_option = st.checkbox(
                    "Fill gaps in coverage",
                    value=True,
                    help="Fill missed/uncovered areas with surrounding values"
                )

                min_area_ha = st.slider(
                    "Min risk zone size (ha)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    format="%.1f ha",
                    help="Remove isolated risk zones smaller than this. "
                         "They'll be absorbed into the surrounding zone. "
                         "The clear zone is never affected."
                )

            # === GENERATE HEATMAP ===
            with st.spinner("🔄 Generating heatmap..."):
                grid_z, grid_x, grid_y, bounds = interpolate_heatmap(
                    gdf_m, snail_type, pixel_size, search_radius,
                    extrapolation_limit, smoothing, apply_blur, blur_sigma
                )

                # Create boundary mask if boundary is available
                boundary_mask = None
                if matched_boundary is not None:
                    boundary_mask = create_boundary_mask(grid_z.shape, bounds, matched_boundary)

                # Fill gaps if enabled (uses boundary mask to stay within field)
                if fill_gaps_option:
                    # max_fill_pixels based on typical gap size (20 pixels = ~60m at 3m resolution)
                    grid_z = fill_gaps(grid_z, max_fill_pixels=30, boundary_mask=boundary_mask)

                # Mask to boundary after filling (ensure nothing outside boundary)
                if boundary_mask is not None:
                    grid_z = np.where(boundary_mask, grid_z, np.nan)

                # Create zone map with clear zone separation
                zone_map = classify_zones(grid_z, thresholds, zone_count, clear_threshold)
                zone_map = filter_small_zones(zone_map, min_area_ha, zone_count, pixel_size)
                zone_rgb = create_zone_rgb(zone_map, zone_count)

                # Calculate zone areas - use boundary area if available for accurate percentages
                zone_areas = calculate_zone_areas(zone_map, pixel_size, zone_count)
                zone_stats = get_zone_statistics(zone_map, grid_z, zone_count, pixel_size)

                # Zone areas are calculated directly from pixel counts
                # (actual mapped/surveyed area, not scaled to boundary)

            # Build legend entries for the zone map
            zone_colors = get_zone_colors(zone_count)
            legend_entries = []
            # Clear zone - zone_stats[0] is always the clear zone (zone 1)
            clear_area = zone_stats[0]['area_ha'] if len(zone_stats) > 0 else 0
            legend_entries.append((
                CLEAR_ZONE_COLOR,
                f"Clear (0 snails): 0 kg/ha — {clear_area:.2f} ha"
            ))
            # Risk zones
            for i in range(zone_count):
                color = zone_colors[i + 2]
                if zone_count == 1:
                    range_text = f">{clear_threshold:.1f}"
                elif i == 0:
                    range_text = f"{clear_threshold:.1f}-{thresholds[0]:.1f}"
                elif i == zone_count - 1:
                    range_text = f">{thresholds[-1]:.1f}"
                else:
                    range_text = f"{thresholds[i-1]:.1f}-{thresholds[i]:.1f}"
                # zone_stats[i+1] corresponds to risk zone i (index 0 is clear)
                risk_area = zone_stats[i + 1]['area_ha'] if (i + 1) < len(zone_stats) else 0
                legend_entries.append((
                    color,
                    f"{zone_names[i]} ({range_text}): {zone_rates[i]} kg/ha — {risk_area:.2f} ha"
                ))

            # Render the figure
            fig = render_heatmap_figure(
                grid_z, bounds, cmap, norm, filename,
                zone_rgb=zone_rgb, zone_map=zone_map,
                legend_entries=legend_entries,
                snail_type=snail_type, hide_field_name=hide_field_name
            )

            # Display the figure
            st.pyplot(fig, width="stretch")

            # === ZONE SUMMARY TABLE ===
            st.subheader("📋 Zone Summary")

            zone_data = []
            total_bait = 0
            for stat in zone_stats:
                if stat['zone_type'] == 'clear':
                    zone_data.append({
                        "Zone": "Clear (No Snails)",
                        "Snail Range": f"<= {clear_threshold:.1f}",
                        "Area (ha)": f"{stat['area_ha']:.2f}",
                        "Avg Density": "0",
                        "Bait Rate (kg/ha)": 0,
                        "Total Bait (kg)": "0"
                    })
                else:
                    risk_idx = stat['zone'] - 2  # 0-based index into zone_names/zone_rates
                    if zone_count == 1:
                        range_text = f"> {clear_threshold:.1f}"
                    elif risk_idx == 0:
                        range_text = f"{clear_threshold:.1f} - {thresholds[0]:.1f}"
                    elif risk_idx == zone_count - 1:
                        range_text = f"> {thresholds[-1]:.1f}"
                    else:
                        range_text = f"{thresholds[risk_idx-1]:.1f} - {thresholds[risk_idx]:.1f}"

                    rate = zone_rates[risk_idx] if risk_idx < len(zone_rates) else 0
                    bait = stat['area_ha'] * rate
                    total_bait += bait

                    zone_data.append({
                        "Zone": zone_names[risk_idx] if risk_idx < len(zone_names) else f"Zone {risk_idx+1}",
                        "Snail Range": range_text,
                        "Area (ha)": f"{stat['area_ha']:.2f}",
                        "Avg Density": f"{stat['avg_density']:.1f}",
                        "Bait Rate (kg/ha)": rate,
                        "Total Bait (kg)": f"{bait:.1f}"
                    })

            # Add total mapped area row
            mapped_area = sum(s['area_ha'] for s in zone_stats)
            zone_data.append({
                "Zone": "**Total Mapped Area**",
                "Snail Range": "",
                "Area (ha)": f"{mapped_area:.2f}",
                "Avg Density": "",
                "Bait Rate (kg/ha)": "",
                "Total Bait (kg)": f"{total_bait:.1f}"
            })

            st.table(pd.DataFrame(zone_data))
            st.info(f"💰 **Total bait required: {total_bait:.1f} kg**")

            # === DOWNLOAD BUTTON ===
            st.subheader("📥 Download Map")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Download as PNG
                png_bytes = figure_to_bytes(fig, format='png', dpi=200)
                st.download_button(
                    label="⬇️ Download as PNG",
                    data=png_bytes,
                    file_name=f"{filename}_snail_map.png",
                    mime="image/png"
                )

            with col2:
                # Download as high-res PNG
                png_hires = figure_to_bytes(fig, format='png', dpi=300)
                st.download_button(
                    label="⬇️ Download High-Res PNG",
                    data=png_hires,
                    file_name=f"{filename}_snail_map_hires.png",
                    mime="image/png"
                )

            with col3:
                # Download zone map only with area labels
                zone_fig = render_zone_map_figure(
                    zone_rgb, bounds, zone_stats, zone_names, zone_count,
                    legend_entries=legend_entries,
                    title="Treatment Zones",
                    clear_threshold=clear_threshold,
                    hide_field_name=hide_field_name
                )
                zone_png = figure_to_bytes(zone_fig, format='png', dpi=200)
                plt.close(zone_fig)
                st.download_button(
                    label="⬇️ Download Zone Map PNG",
                    data=zone_png,
                    file_name=f"{filename}_zone_map.png",
                    mime="image/png"
                )

            # === PRESCRIPTION EXPORT FOR JD OPERATIONS CENTER ===
            st.subheader("🚜 Export Prescription for Operations Center")

            rx_product = st.text_input(
                "Product name (for prescription field name)",
                value="BaitRate",
                help="This becomes the rate column name in the shapefile. "
                     "JD uses this to identify the prescription product. Max 10 characters."
            )

            zone_gdf = zones_to_polygons(
                zone_map, bounds, pixel_size, zone_count,
                zone_names, zone_rates, thresholds, clear_threshold
            )

            # Get field/farm/client from the data if available
            rx_field = filename
            rx_farm = str(prepared_df['Farm'].iloc[0]) if 'Farm' in prepared_df.columns else ""
            rx_client = str(prepared_df['Client'].iloc[0]) if 'Client' in prepared_df.columns else ""

            if zone_gdf is not None:
                shp_zip = export_zones_to_shapefile(
                    zone_gdf, f"{filename}_prescription",
                    product_name=rx_product,
                    field_name=rx_field,
                    farm_name=rx_farm,
                    client_name=rx_client
                )
                if shp_zip:
                    st.download_button(
                        label="📦 Download Prescription Shapefile",
                        data=shp_zip,
                        file_name=f"{filename}_prescription.zip",
                        mime="application/zip",
                        type="primary"
                    )
                    st.info("""
                    **How to upload to Operations Center:**
                    1. Go to **Files** page in Operations Center
                    2. Click **Upload** and select the ZIP file
                    3. Choose file type: **Prescription**
                    4. Assign to the correct field and season
                    """)
            else:
                st.warning("Could not generate prescription. Try adjusting zone settings.")

        elif mode == "Detection Map":
            # === DETECTION MAP SETTINGS ===
            with st.sidebar.expander("📍 Map Settings", expanded=True):
                marker_radius = st.slider(
                    "Marker size (m)",
                    min_value=1,
                    max_value=20,
                    value=DEFAULT_MARKER_RADIUS,
                    help="Size of each detection marker on the map"
                )
                zoom_level = st.slider(
                    "Zoom level",
                    min_value=10,
                    max_value=20,
                    value=DEFAULT_ZOOM,
                    help="Initial zoom level of the map"
                )

            # === GENERATE MAP ===
            with st.spinner("🔄 Creating detection map..."):
                gdf_latlon = gdf.to_crs(epsg=4326)
                gdf_latlon["lon"] = gdf_latlon.geometry.x
                gdf_latlon["lat"] = gdf_latlon.geometry.y

                center = [gdf_latlon["lat"].mean(), gdf_latlon["lon"].mean()]
                m = leafmap.Map(center=center, zoom=zoom_level)

                # Add colored markers
                gdf_latlon["color"] = gdf_latlon[snail_type].apply(
                    lambda x: mcolors.to_hex(cmap(norm(x)))
                )

                # Limit markers for performance (warn if too many)
                max_markers = 5000
                if len(gdf_latlon) > max_markers:
                    st.warning(f"⚠️ Showing first {max_markers:,} of {len(gdf_latlon):,} points for performance. Use Heatmap mode for full data.")
                    gdf_latlon = gdf_latlon.head(max_markers)

                for _, row in gdf_latlon.iterrows():
                    folium.Circle(
                        location=(row["lat"], row["lon"]),
                        radius=marker_radius,
                        color=row["color"],
                        fill=True,
                        fill_color=row["color"],
                        fill_opacity=0.8,
                        popup=f"{snail_type}: {row[snail_type]}",
                        weight=1,
                        opacity=1
                    ).add_to(m)

            # Display the map
            st.subheader("🗺️ Detection Map")
            components.html(m.to_html(), height=650)

            # Color legend
            st.caption("🟢 Low snail count → 🟡 Medium → 🔴 High snail count")

            # Note about screenshot
            st.info("💡 **Tip:** Use your browser's screenshot tool or right-click to save the map image.")

else:
    # No file uploaded - show instructions
    st.info("👆 Upload a CSV file in the sidebar to begin.")

    with st.expander("📖 How to use this tool"):
        st.markdown("""
        ### Getting Started

        1. **Upload your SnailCam data** using the sidebar
        2. **Choose visualization mode**:
           - **Heatmap**: See snail density patterns across your field
           - **Detection Map**: See individual detection points
        3. **Adjust settings** to refine the visualization
        4. **Download** your maps as images

        ### Supported File Types

        - **Harvest Summary**: Aggregated counts per location
        - **Detection Details**: Individual detections with confidence scores

        ### Tips

        - Use **confidence filtering** to exclude uncertain detections
        - Adjust **zone thresholds** to match your treatment strategy
        - The **total bait calculator** helps estimate costs
        """)
