
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_DIR / "src"

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "best_model.pth"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# IMPORT MODEL / INFERENCE
# ============================================================

from model import SiameseFCSDiff

from inference import (
    predict_from_paths
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 256

THRESHOLD = 0.70

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = SiameseFCSDiff().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Satellite Building Change Detection",
    page_icon="🛰️",
    layout="wide"
)


st.title(
    "🛰️ Satellite Building Change Detection"
)

st.markdown(
    """
Compare two satellite images from different time periods
and detect areas where building changes occurred.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Architecture:** Siamese FC-Siam-Diff"
    )

    st.write(
        "**Dataset:** LEVIR-CD"
    )

    st.write(
        "**Patch Size:** 256 × 256"
    )

    st.write(
        "**Detection Threshold:** 0.70"
    )

    st.write(
        f"**Best Epoch:** {checkpoint['epoch']}"
    )

    st.write(
        f"**Best Validation IoU:** "
        f"{checkpoint['best_iou']:.4f}"
    )

    st.divider()

    st.write(
        f"**Device:** {DEVICE}"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "Upload Satellite Images"
)

col1, col2 = st.columns(2)

with col1:

    before_file = st.file_uploader(
        "Before Image",
        type=["png", "jpg", "jpeg"],
        key="before"
    )

with col2:

    after_file = st.file_uploader(
        "After Image",
        type=["png", "jpg", "jpeg"],
        key="after"
    )


# ============================================================
# DISPLAY UPLOADED IMAGES
# ============================================================

if before_file and after_file:

    before_image = Image.open(
        before_file
    ).convert("RGB")

    after_image = Image.open(
        after_file
    ).convert("RGB")

    if before_image.size != after_image.size:

        st.error(
            "Before and After images must have "
            "the same dimensions."
        )

        st.stop()

    st.subheader(
        "Input Images"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            before_image,
            caption="Before",
            width=450
        )

    with col2:

        st.image(
            after_image,
            caption="After",
            width=450
        )


    # ========================================================
    # VALIDATE IMAGE SIZE
    # ========================================================

    width, height = before_image.size

    if (
        width % PATCH_SIZE != 0
        or height % PATCH_SIZE != 0
    ):

        st.error(
            f"Image dimensions must be divisible "
            f"by {PATCH_SIZE}. "
            f"Received {width} × {height}."
        )

        st.stop()


    # ========================================================
    # DETECT BUTTON
    # ========================================================

    detect = st.button(
        "🔍 Detect Changes",
        type="primary",
        use_container_width=True
    )


    if detect:

        with tempfile.TemporaryDirectory() as temp_dir:

            before_path = os.path.join(
                temp_dir,
                "before.png"
            )

            after_path = os.path.join(
                temp_dir,
                "after.png"
            )

            before_image.save(
                before_path
            )

            after_image.save(
                after_path
            )


            # =================================================
            # INFERENCE
            # =================================================

            with st.spinner(
                "Running change detection..."
            ):

                (
                    image_a,
                    image_b,
                    probability_map,
                    change_mask
                ) = predict_from_paths(

                    model=model,

                    before_path=before_path,

                    after_path=after_path,

                    device=DEVICE,

                    threshold=THRESHOLD,

                    patch_size=PATCH_SIZE
                )


        # ====================================================
        # RESULTS
        # ====================================================

        changed_pixels = int(
            change_mask.sum()
        )

        total_pixels = int(
            change_mask.size
        )

        changed_percentage = (
            changed_pixels /
            total_pixels
        ) * 100


        st.success(
            "Change detection completed."
        )


        # ====================================================
        # METRICS
        # ====================================================

        st.subheader(
            "Detection Summary"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Changed Pixels",
                f"{changed_pixels:,}"
            )

        with col2:

            st.metric(
                "Changed Area",
                f"{changed_percentage:.2f}%"
            )

        with col3:

            st.metric(
                "Threshold",
                f"{THRESHOLD:.2f}"
            )


        # ====================================================
        # CHANGE MASK
        # ====================================================

        st.subheader(
            "Predicted Change Mask"
        )

        st.image(
            change_mask * 255,
            caption="Detected Building Changes",
            clamp=True,
            width=600
        )


        # ====================================================
        # PROBABILITY MAP
        # ====================================================

        st.subheader(
            "Change Probability Map"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        im = ax.imshow(
            probability_map,
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            "Pixel-wise Change Probability"
        )

        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

        st.pyplot(
            fig,
            width=600
        )

        plt.close(fig)


        # ====================================================
        # OVERLAY
        # ====================================================

        st.subheader(
            "Change Overlay"
        )

        overlay = (
            image_b.astype(
                np.float32
            ) / 255.0
        )

        overlay = overlay.copy()

        overlay[
            change_mask == 1
        ] = (
            1.0,
            0.0,
            0.0
        )

        st.image(
            overlay,
            caption="Detected Changes Highlighted in Red",
            width=600
        )


        # ====================================================
        # DOWNLOAD MASK
        # ====================================================

        mask_bytes = (
            (change_mask * 255)
            .astype(np.uint8)
        )

        mask_image = Image.fromarray(
            mask_bytes
        )

        import io

        buffer = io.BytesIO()

        mask_image.save(
            buffer,
            format="PNG"
        )

        st.download_button(
            label="⬇️ Download Change Mask",
            data=buffer.getvalue(),
            file_name="change_mask.png",
            mime="image/png",
            use_container_width=True
        )


else:

    st.info(
        "Upload both a Before and After satellite image "
        "to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Satellite Building Change Detection | "
    "Siamese FC-Siam-Diff | LEVIR-CD"
)
