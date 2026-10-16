
import io
import os
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
# PAGE HEADER
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


@st.cache_resource(show_spinner="Loading colorization model...")
def load_colorization_model():
    from tensorflow import keras

    model_path = find_model_path()

    if model_path is None:
        raise FileNotFoundError(
            "No trained model found in models/. "
            "Ensure best_colorization_model.keras or "
            "colorization_model.keras exists in the repository."
        )

    model = keras.models.load_model(
        str(model_path),
        compile=False,
    )

    if isinstance(model.input_shape, list) or isinstance(
        model.output_shape, list
    ):
        raise ValueError("Expected a single-input, single-output model.")

    if model.input_shape[-1] != 1 or model.output_shape[-1] != 3:
        raise ValueError(
            f"Incompatible model shapes: {model.input_shape} -> "
            f"{model.output_shape}"
        )

    return model, model_path.name


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

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

    st.write(f"**Target Input:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 1) Grayscale")
    st.write(f"**Target Output:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 3) RGB")
    st.caption("Colors are predictions, not guaranteed original colors.")


# =====================================================
# IMAGE UPLOAD AND PREVIEW
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

        original, grayscale, model_input, original_size = prepare_image(
            image_bytes
        )

        upload_key = (
            uploaded_file.name,
            len(image_bytes),
            hash(image_bytes),
        )

        if st.session_state.get("upload_key") != upload_key:
            st.session_state["upload_key"] = upload_key
            st.session_state.pop("colorized_image_bytes", None)

        st.subheader("Input Preview")

        before_col, gray_col = st.columns(2)

        with before_col:
            st.image(
                original,
                caption="Uploaded image",
                use_container_width=True,
            )

        with gray_col:
            st.image(
                grayscale,
                caption="Grayscale input used by the CNN",
                use_container_width=True,
            )

        # =================================================
        # RUN COLORIZATION
        # =================================================

        if st.button(
            "Colorize Image",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Generating colorized image..."):
                model, model_name = load_colorization_model()

                result = colorize_image(
                    model,
                    model_input,
                    original_size,
                )

                # Store PNG bytes instead of relying on a temporary image.
                output_buffer = io.BytesIO()
                result.save(output_buffer, format="PNG")

                st.session_state["colorized_image_bytes"] = (
                    output_buffer.getvalue()
                )

        # =================================================
        # RESULT AND DOWNLOAD
        # =================================================

        png_data = st.session_state.get("colorized_image_bytes")

        if png_data:
            with Image.open(io.BytesIO(png_data)) as saved_image:
                result = saved_image.convert("RGB")

            st.divider()
            st.subheader("Colorization Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.image(
                    grayscale,
                    caption="Before: Grayscale",
                    use_container_width=True,
                )

            with result_col2:
                st.image(
                    result,
                    caption="After: CNN Colorization",
                    use_container_width=True,
                )

            st.download_button(
                label="Download Colorized Image",
                data=png_data,
                file_name="colorized_image.png",
                mime="image/png",
                key="download_colorized_image",
                use_container_width=True,
            )

    except (UnidentifiedImageError, OSError):
        st.error(
            "Unable to read this image. Please upload a valid "
            "JPG, JPEG, PNG, BMP, or WebP file."
        )

    except Exception as error:
        st.error(f"Image processing failed: {error}")
