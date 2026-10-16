
import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 25

# Start with a manageable subset.
MAX_HUMAN_IMAGES = 3000
MAX_LANDSCAPE_IMAGES = 500

HUMAN_DIR = os.path.join("dataset", "human")
LANDSCAPE_DIR = os.path.join("dataset", "landscape")

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR, "best_colorization_model.keras"
)
FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "colorization_model.keras"
)

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Avoid allocating all GPU memory immediately.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

print("TensorFlow:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))


# ==========================================
# FIND IMAGES RECURSIVELY
# ==========================================

def find_images(folder):
    extensions = ["jpg", "jpeg", "png", "webp"]
    paths = []

    for ext in extensions:
        paths.extend(
            glob.glob(
                os.path.join(folder, "**", f"*.{ext}"),
                recursive=True
            )
        )
        paths.extend(
            glob.glob(
                os.path.join(folder, "**", f"*.{ext.upper()}"),
                recursive=True
            )
        )

    return sorted(set(paths))


def select_images(folder, limit):
    paths = find_images(folder)

    if not paths:
        raise FileNotFoundError(
            f"No images found inside: {folder}"
        )

    random.shuffle(paths)
    selected = paths[:limit]

    print(f"{folder}: found {len(paths)} images")
    print(f"Selected {len(selected)} images")

    return selected


# ==========================================
# LOAD AND PREPROCESS IMAGES
# ==========================================

def load_pair(path):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.convert_image_dtype(
        image, tf.float32
    )

    # Resize to a fixed size for batch training.
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="area"
    )

    image = tf.clip_by_value(image, 0.0, 1.0)

    # Data augmentation: transform the color target
    # before generating its corresponding grayscale input.
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    # RGB target.
    target = image

    # One-channel grayscale input.
    gray = tf.image.rgb_to_grayscale(target)

    return gray, target


# ==========================================
# CNN ENCODER-DECODER
# ==========================================

def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def build_model():
    inputs = keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        name="grayscale_image"
    )

    # Encoder.
    e1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(e1)

    e2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(e2)

    e3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(e3)

    # Bottleneck.
    b = conv_block(p3, 256)

    # Decoder with skip connections.
    d3 = layers.UpSampling2D(
        interpolation="bilinear"
    )(b)
    d3 = layers.Concatenate()([d3, e3])
    d3 = conv_block(d3, 128)

    d2 = layers.UpSampling2D(
        interpolation="bilinear"
    )(d3)
    d2 = layers.Concatenate()([d2, e2])
    d2 = conv_block(d2, 64)

    d1 = layers.UpSampling2D(
        interpolation="bilinear"
    )(d2)
    d1 = layers.Concatenate()([d1, e1])
    d1 = conv_block(d1, 32)

    # Predict three color channels.
    outputs = layers.Conv2D(
        3, 1,
        activation="sigmoid",
        name="predicted_rgb"
    )(d1)

    return keras.Model(
        inputs, outputs,
        name="HumanLandscapeColorizationCNN"
    )


# ==========================================
# METRICS
# ==========================================

def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.psnr(
            y_true, y_pred, max_val=1.0
        )
    )


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.ssim(
            y_true, y_pred, max_val=1.0
        )
    )


# ==========================================
# TRAINING
# ==========================================

def main():
    human_paths = select_images(
        HUMAN_DIR, MAX_HUMAN_IMAGES
    )

    landscape_paths = select_images(
        LANDSCAPE_DIR, MAX_LANDSCAPE_IMAGES
    )

    # Keep a validation split from each domain.
    random.shuffle(human_paths)
    random.shuffle(landscape_paths)

    def split_paths(paths):
        if len(paths) < 10:
            raise ValueError(
                "Each dataset needs at least 10 usable images."
            )

        split_at = max(1, int(len(paths) * 0.9))
        return paths[:split_at], paths[split_at:]

    human_train, human_val = split_paths(human_paths)
    landscape_train, landscape_val = split_paths(
        landscape_paths
    )

    train_paths = human_train + landscape_train
    val_paths = human_val + landscape_val

    random.shuffle(train_paths)
    random.shuffle(val_paths)

    print("\nTraining images:", len(train_paths))
    print("Validation images:", len(val_paths))

    # Train on randomly sampled image patches represented
    # as complete 128x128 resized images.
    train_ds = tf.data.Dataset.from_tensor_slices(
        train_paths
    )

    train_ds = train_ds.shuffle(
        len(train_paths),
        seed=SEED,
        reshuffle_each_iteration=True
    )

    train_ds = train_ds.map(
        load_pair,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        val_paths
    )

    val_ds = val_ds.map(
        load_pair,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mae",
        metrics=[psnr_metric, ssim_metric]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(MODEL_DIR, "training_log.csv")
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save(FINAL_MODEL_PATH)

    print("\nTraining complete!")
    print("Best model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
