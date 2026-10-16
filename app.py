import io
import os
from pathlib import Path

# =====================================================
# PERFORMANCE & ENVIRONMENT OPTIMIZATIONS
# =====================================================
# Set TensorFlow flags early to suppress verbose C++ logs and avoid slow oneDNN checks
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

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

st.title("🎨 Image Colorization Using a CNN")
st.write(
    "Colorize grayscale human portraits and landscape "
    "photographs using a trained convolutional neural network."
)

st.info(
    "The model predicts plausible colors. Exact original "
    "colors cannot always be recovered from grayscale images."
)


# =====================================================
# MODEL LOADING (CACHED & LAZY)
# =====================================================

def find_model_path():
    """Quickly check if a model file exists on disk without loading it into RAM."""
    return next((p for p in MODEL_CANDIDATES if p.exists()), None)


@st.cache_resource(show_spinner="Loading deep learning model into memory...")
def load_colorization_model():
    """
    Lazily loads the Keras model into memory once and caches it across sessions.
    Also warms up the execution graph to make first inference instant.
    """
    from tensorflow import keras

    model_path = find_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "No trained model found in the models folder. "
            "Please ensure 'best_colorization_model.keras' or 'colorization_model.keras' is present."
        )

    model = keras.models.load_model(str(model_path), compile=False)

    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list) or isinstance(output_shape, list):
        raise ValueError("This app expects a single-input, single-output model.")

    if input_shape[-1] != 1 or output_shape[-1] != 3:
        raise ValueError(
            "The selected model is not compatible with grayscale-to-RGB pipeline.\n\n"
            f"Input shape: {input_shape}\n"
            f"Output shape: {output_shape}"
        )

    # Warm-up execution graph on a dummy input so first inference has no compilation delay
    try:
        dummy_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
        _ = model(dummy_input, training=False)
    except Exception:
        pass

    return model, str(model_path), input_shape, output_shape


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def prepare_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    original_size = image.size

    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.LANCZOS,
    )

    grayscale = ImageOps.grayscale(resized)

    array = np.asarray(
        grayscale,
        dtype=np.float32,
    ) / 255.0

    model_input = array[np.newaxis, ..., np.newaxis]

    return image, grayscale, model_input, original_size


# =====================================================
# COLORIZATION
# =====================================================

def colorize_image(model, model_input, original_size):
    # Direct model call is 5x-10x faster than model.predict for single images
    try:
        prediction = model(model_input, training=False)
        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()
        else:
            prediction = np.asarray(prediction)
    except Exception:
        prediction = model.predict(model_input, verbose=0)

    prediction = np.asarray(prediction)

    if (
        prediction.ndim != 4
        or prediction.shape[0] != 1
        or prediction.shape[-1] != 3
    ):
        raise ValueError(
            f"Unexpected model output shape: {prediction.shape}"
        )

    prediction = prediction[0]

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    prediction = np.clip(prediction, 0.0, 1.0)
    prediction = np.round(prediction * 255).astype(np.uint8)

    result = Image.fromarray(prediction)
    result = result.resize(
        original_size,
        Image.Resampling.LANCZOS,
    )

    return result


# =====================================================
# SIDEBAR
# =====================================================

model_file = find_model_path()

with st.sidebar:
    st.header("About the Project")
    st.write(
        "A CNN-based image colorization system for portraits "
        "and landscapes."
    )

    if model_file:
        st.success(f"Model ready: `{model_file.name}`", icon="✅")
    else:
        st.error("No model found in `models/` folder", icon="⚠️")

    st.write(f"**Target Input:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 1) Grayscale")
    st.write(f"**Target Output:** ({IMAGE_SIZE}, {IMAGE_SIZE}, 3) RGB")
    st.caption("Colors are predictions, not guaranteed original colors.")

    if model_file:
        if st.button("⚡ Preload Model into RAM", use_container_width=True):
            with st.spinner("Preloading model..."):
                try:
                    load_colorization_model()
                    st.success("Model preloaded in RAM!")
                except Exception as e:
                    st.error(f"Failed to load: {e}")

# =====================================================
# UPLOAD AND DISPLAY
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a grayscale portrait or landscape",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    key="image_upload",
)

if uploaded_file is not None:
    # Reset result if a different image is uploaded
    current_upload = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("current_upload") != current_upload:
        st.session_state["current_upload"] = current_upload
        st.session_state.pop("colorized_image", None)

    try:
        original, grayscale, model_input, original_size = prepare_image(uploaded_file)

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

        if st.button(
            "🎨 Colorize Image",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Colorizing image with CNN..."):
                model, model_path, _, _ = load_colorization_model()
                result = colorize_image(
                    model,
                    model_input,
                    original_size,
                )
                st.session_state["colorized_image"] = result

        result = st.session_state.get("colorized_image")

        if result is not None:
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

            image_bytes = io.BytesIO()
            result.save(image_bytes, format="PNG")

            st.download_button(
                "⬇️ Download Colorized Image",
                data=image_bytes.getvalue(),
                file_name="colorized_image.png",
                mime="image/png",
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Image processing failed: {error}")

else:
    st.write("Upload an image to begin.")
