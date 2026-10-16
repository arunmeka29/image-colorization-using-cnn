
import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps

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

st.title("🎨Image Colorization Using a CNN")
st.write(
    "Colorize grayscale human portraits and landscape "
    "photographs using a trained convolutional neural network."
)

st.info(
    "The model predicts plausible colors. Exact original "
    "colors cannot always be recovered from grayscale images."
)

# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_colorization_model():
    model_path = next(
        (p for p in MODEL_CANDIDATES if p.exists()),
        None,
    )

    if model_path is None:
        raise FileNotFoundError(
            "No trained model found in the models folder."
        )

    model = keras.models.load_model(
        str(model_path),
        compile=False,
    )

    return model, str(model_path)


try:
    model, model_path = load_colorization_model()
    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list) or isinstance(output_shape, list):
        st.error("This app expects a single-input, single-output model.")
        st.stop()

    input_channels = input_shape[-1]
    output_channels = output_shape[-1]

    if input_channels != 1 or output_channels != 3:
        st.error(
            "The selected model is not compatible with this app's "
            "grayscale-to-RGB pipeline.\n\n"
            f"Input shape: {input_shape}\n"
            f"Output shape: {output_shape}\n\n"
            "Your improved CIELAB model requires different "
            "preprocessing and output reconstruction."
        )
        st.stop()

    st.success(f"Loaded model: {Path(model_path).name}")

except Exception as error:
    st.error(f"Unable to load compatible model: {error}")
    st.stop()

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

with st.sidebar:
    st.header("About the project")
    st.write(
        "A CNN-based image colorization system for portraits "
        "and landscapes."
    )
    st.write(f"**Input:** {input_shape}")
    st.write(f"**Output:** {output_shape}")
    st.caption(
        "Colors are predictions, not guaranteed original colors."
    )

# =====================================================
# UPLOAD AND DISPLAY
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a grayscale portrait or landscape",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    key="image_upload",
)

if uploaded_file is not None:
    # Reset the result whenever a different image is uploaded.
    current_upload = (
        uploaded_file.name,
        uploaded_file.getvalue(),
    )
    if st.session_state.get("current_upload") != current_upload:
        st.session_state["current_upload"] = current_upload
        st.session_state.pop("colorized_image", None)

    try:
        original, grayscale, model_input, original_size = (
            prepare_image(uploaded_file)
        )

        st.subheader("Input image")

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
            with st.spinner("Predicting colors..."):
                result = colorize_image(
                    model,
                    model_input,
                    original_size,
                )

                st.session_state["colorized_image"] = result

        result = st.session_state.get("colorized_image")

        if result is not None:
            st.divider()
            st.subheader("Colorization result")

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
                    caption="After: CNN colorization",
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
