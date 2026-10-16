
import io
import os
import hashlib
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
IMAGE_SIZE = 128

MODEL_CANDIDATES = [
    MODEL_DIR / "best_colorization_model.keras",
    MODEL_DIR / "colorization_model.keras",
]

st.set_page_config(
    page_title="Image Colorization Using a CNN",
    page_icon="🎨",
    layout="wide",
)

# =====================================================
# HEADER
# =====================================================

st.title("Image Colorization Using a CNN")
st.write(
    "Colorize grayscale human portraits and landscape "
    "photographs using a trained convolutional neural network."
)

st.info(
    "The model predicts plausible colors. Exact original colors "
    "cannot always be recovered from grayscale images."
)

# =====================================================
# MODEL LOADING
# =====================================================

def find_model_path():
    return next(
        (path for path in MODEL_CANDIDATES if path.is_file()),
        None,
    )


@st.cache_resource(show_spinner="Loading CNN model...")
def load_colorization_model(model_path_string):
    from tensorflow import keras

    model = keras.models.load_model(
        model_path_string,
        compile=False,
    )

    if isinstance(model.input_shape, list) or isinstance(
        model.output_shape, list
    ):
        raise ValueError(
            "The model must have one input and one output."
        )

    if model.input_shape[-1] != 1 or model.output_shape[-1] != 3:
        raise ValueError(
            f"Incompatible model shapes: {model.input_shape} -> "
            f"{model.output_shape}"
        )

    return model


# =====================================================
# IMAGE UTILITIES
# =====================================================

def image_to_png_bytes(image):
    """Convert a PIL image into PNG bytes for display and download."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def display_image(image, caption):
    """Display PNG bytes rather than passing a PIL object directly."""
    png_bytes = image_to_png_bytes(image)

    st.image(
        png_bytes,
        caption=caption,
        output_format="PNG",
        use_container_width=True,
    )


def prepare_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as opened_image:
        opened_image.load()
        original = ImageOps.exif_transpose(opened_image).convert("RGB")

    original_size = original.size

    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    grayscale = ImageOps.grayscale(resized)

    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    model_input = array[np.newaxis, ..., np.newaxis]

    return original, grayscale, model_input, original_size


# =====================================================
# COLORIZATION
# =====================================================

def colorize_image(model, model_input, original_size):
    prediction = model(model_input, training=False)

    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()

    prediction = np.asarray(prediction)

    if (
        prediction.ndim != 4
        or prediction.shape[0] != 1
        or prediction.shape[-1] != 3
    ):
        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    prediction = np.nan_to_num(
        prediction[0],
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    prediction = np.clip(prediction, 0.0, 1.0)
    prediction = np.round(prediction * 255).astype(np.uint8)

    result = Image.fromarray(prediction, mode="RGB")

    return result.resize(
        original_size,
        Image.Resampling.LANCZOS,
    )


# =====================================================
# SIDEBAR
# =====================================================

model_path = find_model_path()

with st.sidebar:
    st.header("About the Project")
    st.write(
        "A CNN-based image colorization system for portraits "
        "and landscapes."
    )

    if model_path:
        st.success(f"Model file found: {model_path.name}")
    else:
        st.error("No model found in the models/ folder.")

    st.write(
        f"**Target Input:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 1) Grayscale"
    )
    st.write(
        f"**Target Output:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 3) RGB"
    )
    st.caption("Colors are predictions, not guaranteed original colors.")


# =====================================================
# UPLOAD AND PREVIEW
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a grayscale portrait or landscape",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    key="image_upload",
)

if uploaded_file is None:
    st.write("Upload an image to begin.")

else:
    try:
        image_bytes = uploaded_file.getvalue()

        if not image_bytes:
            st.error("The uploaded file is empty.")
            st.stop()

        upload_hash = hashlib.sha256(image_bytes).hexdigest()

        if st.session_state.get("upload_hash") != upload_hash:
            st.session_state["upload_hash"] = upload_hash
            st.session_state.pop("colorized_png", None)

        original, grayscale, model_input, original_size = prepare_image(
            image_bytes
        )

        st.subheader("Input Preview")

        before_col, gray_col = st.columns(2)

        with before_col:
            display_image(original, "Uploaded image")

        with gray_col:
            display_image(grayscale, "Grayscale input used by the CNN")

        # =================================================
        # COLORIZE BUTTON
        # =================================================

        if st.button(
            "Colorize Image",
            type="primary",
            use_container_width=True,
        ):
            model_path = find_model_path()

            if model_path is None:
                st.error(
                    "Model not found. Check the repository's models/ folder."
                )
            else:
                try:
                    with st.spinner("Colorizing image with CNN..."):
                        model = load_colorization_model(str(model_path))

                        result = colorize_image(
                            model,
                            model_input,
                            original_size,
                        )

                        st.session_state["colorized_png"] = (
                            image_to_png_bytes(result)
                        )

                except Exception as error:
                    st.error(f"Colorization failed: {error}")

        # =================================================
        # RESULT AND DOWNLOAD
        # =================================================

        colorized_png = st.session_state.get("colorized_png")

        if colorized_png:
            with Image.open(io.BytesIO(colorized_png)) as saved_image:
                result = saved_image.convert("RGB")

            st.divider()
            st.subheader("Colorization Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                display_image(grayscale, "Before: Grayscale")

            with result_col2:
                display_image(result, "After: CNN Colorization")

            st.download_button(
                label="Download Colorized Image",
                data=colorized_png,
                file_name="colorized_image.png",
                mime="image/png",
                key=f"download_{upload_hash}",
                use_container_width=True,
            )

    except (UnidentifiedImageError, OSError):
        st.error(
            "Unable to read this image. Upload a valid JPG, JPEG, "
            "PNG, BMP, or WebP image."
        )

    except Exception as error:
        st.error(f"Image processing failed: {error}")
