import streamlit as st
from pathlib import Path
import pandas as pd
import networkx as nx

from geometric_features import (
    load_crack_mask,
    calculate_width_features,
    calculate_crack_coverage,
    calculate_crack_area,
    calculate_crack_orientation
)

from skeletonization import skeletonize_crack

from feature_extraction import (
    get_clustered_endpoints,
    get_clustered_branch_points,
    build_skeleton_graph,
    calculate_skeleton_length,
    calculate_branch_density,
    calculate_endpoint_density
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Pavement Crack Severity Analyzer",
    page_icon="🛣️",
    layout="wide"
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("About the Project")

    st.write(
        "Pavement Crack Severity Analysis "
        "using the CRACK500 dataset."
    )

    st.divider()

    st.subheader("Current Version")

    st.write(
        "Ground-truth based analysis"
    )

    st.divider()

    st.subheader("Dataset")

    st.write(
        "CRACK500"
    )

    st.write(
        "1,896 paired images and masks"
    )

    st.divider()

    st.subheader("Important")

    st.warning(
        "This dashboard currently supports "
        "CRACK500 dataset images only. "
        "Ground-truth masks are used for "
        "feature extraction and CSI analysis."
    )


# ==================================================
# DATASET PATHS
# ==================================================

DATASET_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "CRACK500"
    / "traincrop"
)

CSI_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "stage3"
    / "features"
    / "final_csi"
    / "crack500_csi_results.csv"
)


# ==================================================
# MAIN HEADER
# ==================================================

st.title(
    "🛣️ Pavement Crack Severity Analyzer"
)

st.markdown(
    """
    ### CRACK500 Ground-Truth Based Analysis

    Analyze pavement crack characteristics using the
    corresponding ground-truth mask and the validated
    Crack Severity Index (CSI).
    """
)

st.divider()


# ==================================================
# DATASET CHECK
# ==================================================

if DATASET_DIR.exists():

    st.success(
        "CRACK500 dataset found successfully."
    )

    jpg_count = len(
        list(DATASET_DIR.glob("*.jpg"))
    )

    png_count = len(
        list(DATASET_DIR.glob("*.png"))
    )

    st.write(
        f"JPG images found: **{jpg_count}**"
    )

    st.write(
        f"PNG masks found: **{png_count}**"
    )

else:

    st.error(
        "CRACK500 dataset folder was not found."
    )


# ==================================================
# 1. INPUT IMAGE
# ==================================================

st.header(
    "🔹 1. Input Image"
)

st.write(
    "Upload an image from the CRACK500 dataset."
)

uploaded_file = st.file_uploader(
    "Choose a CRACK500 JPG image",
    type=["jpg", "jpeg"]
)


# ==================================================
# PROCESS UPLOADED IMAGE
# ==================================================

if uploaded_file is not None:

    st.write(
        f"Uploaded image: **{uploaded_file.name}**"
    )

    # --------------------------------------------------
    # Find matching image in CRACK500
    # --------------------------------------------------

    matching_images = list(
        DATASET_DIR.rglob(
            uploaded_file.name
        )
    )

    if matching_images:

        image_path = matching_images[0]

        # Corresponding ground-truth mask
        mask_path = image_path.with_suffix(
            ".png"
        )

        st.success(
            "Matching CRACK500 image found."
        )

        # ==================================================
        # 2. INPUT IMAGE + GROUND-TRUTH MASK
        # ==================================================

        if mask_path.exists():

            st.header(
                "🔹 2. Input Image & Ground-Truth Mask"
            )

            st.success(
                "Ground-truth mask found."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "2A. Input Image"
                )

                st.image(
                    str(image_path),
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "2B. Ground-Truth Mask"
                )

                st.image(
                    str(mask_path),
                    use_container_width=True
                )


            # ==================================================
            # LOAD GROUND-TRUTH MASK
            # ==================================================

            mask = load_crack_mask(
                str(mask_path)
            )


            # ==================================================
            # GEOMETRIC FEATURES
            # ==================================================

            width_features = (
                calculate_width_features(
                    mask
                )
            )

            coverage = (
                calculate_crack_coverage(
                    mask
                )
            )

            area = (
                calculate_crack_area(
                    mask
                )
            )

            orientation = (
                calculate_crack_orientation(
                    mask
                )
            )


            # ==================================================
            # 3. CRACK FEATURES
            # ==================================================

            st.header(
                "🔹 3. Crack Features"
            )

            st.write(
                "Geometric characteristics extracted "
                "from the ground-truth crack mask."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Average Width",
                    f"{width_features['average_width_pixels']:.2f} px"
                )

                st.metric(
                    "Crack Coverage",
                    f"{coverage:.2f}%"
                )

            with col2:

                st.metric(
                    "Maximum Width",
                    f"{width_features['maximum_width_pixels']:.2f} px"
                )

                st.metric(
                    "Crack Area",
                    f"{area:,} pixels"
                )

            with col3:

                st.metric(
                    "Dominant Orientation",
                    f"{orientation:.2f}°"
                )


            # ==================================================
            # SKELETONIZATION
            # ==================================================

            skeleton_image = (
                skeletonize_crack(
                    mask
                )
            )

            skeleton = (
                skeleton_image > 0
            )


            # ==================================================
            # ENDPOINTS AND BRANCH POINTS
            # ==================================================

            endpoints = (
                get_clustered_endpoints(
                    skeleton,
                    radius=3
                )
            )

            branch_points = (
                get_clustered_branch_points(
                    skeleton,
                    radius=3
                )
            )


            # ==================================================
            # BUILD SKELETON GRAPH
            # ==================================================

            graph = (
                build_skeleton_graph(
                    skeleton
                )
            )


            # ==================================================
            # GRAPH FEATURES
            # ==================================================

            connected_components = (
                nx.number_connected_components(
                    graph
                )
            )

            skeleton_length = (
                calculate_skeleton_length(
                    graph
                )
            )

            branch_density = (
                calculate_branch_density(
                    len(branch_points),
                    skeleton_length
                )
            )

            endpoint_density = (
                calculate_endpoint_density(
                    len(endpoints),
                    skeleton_length
                )
            )


            # ==================================================
            # 4. SKELETON & GRAPH FEATURES
            # ==================================================

            st.header(
                "🔹 4. Skeleton & Graph Features"
            )

            st.write(
                "Structural characteristics extracted "
                "from the skeletonized ground-truth mask."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Skeleton Length",
                    f"{skeleton_length:.2f} px"
                )

                st.metric(
                    "Endpoint Count",
                    len(endpoints)
                )

            with col2:

                st.metric(
                    "Branch Point Count",
                    len(branch_points)
                )

                st.metric(
                    "Connected Components",
                    connected_components
                )

            with col3:

                st.metric(
                    "Branch Density",
                    f"{branch_density:.6f}"
                )

                st.metric(
                    "Endpoint Density",
                    f"{endpoint_density:.6f}"
                )


            # ==================================================
            # 5. CRACK SEVERITY INDEX
            # ==================================================

            st.header(
                "🔹 5. Crack Severity Index"
            )

            st.write(
                "Validated CSI result obtained from the "
                "CRACK500 research analysis."
            )


            # --------------------------------------------------
            # Load validated CSI results
            # --------------------------------------------------

            if CSI_PATH.exists():

                csi_df = pd.read_csv(
                    CSI_PATH
                )

                image_name = (
                    image_path.stem
                )

                csi_row = csi_df[
                    csi_df["image"] == image_name
                ]


                # --------------------------------------------------
                # Display CSI
                # --------------------------------------------------

                if not csi_row.empty:

                    csi_value = float(
                        csi_row.iloc[0]["CSI"]
                    )

                    severity = str(
                        csi_row.iloc[0]["severity"]
                    )

                    st.markdown("---")

                    st.subheader(
                        "Final Crack Severity Assessment"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Crack Severity Index (CSI)",
                            f"{csi_value:.2f} / 100"
                        )

                    with col2:

                        st.metric(
                            "Severity Classification",
                            severity
                        )


                    # ------------------------------------------
                    # Severity interpretation
                    # ------------------------------------------

                    if severity == "Low":

                        st.info(
                            "Low Severity — the crack characteristics "
                            "correspond to the low-severity range."
                        )

                    elif severity == "Moderate":

                        st.warning(
                            "Moderate Severity — the crack characteristics "
                            "correspond to the moderate-severity range."
                        )

                    elif severity == "High":

                        st.error(
                            "High Severity — the crack characteristics "
                            "correspond to the high-severity range."
                        )

                    st.caption(
                        "CSI thresholds: Low < 27.73 | "
                        "Moderate: 27.73–46.83 | "
                        "High > 46.83"
                    )

                else:

                    st.warning(
                        "Validated CSI result not found "
                        "for this image."
                    )

            else:

                st.error(
                    "CSI results file was not found."
                )


        else:

            st.error(
                "Matching image was found, "
                "but its ground-truth mask is missing."
            )


    else:

        st.error(
            "This image was not found in the "
            "CRACK500 dataset."
        )