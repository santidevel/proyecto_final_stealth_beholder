from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    Modelo CNN para Stealth Beholder.

    Compatible con:
    - Pipeline existente del proyecto
    - Clasificación binaria (2 clases)
    - Imágenes ya normalizadas en el pipeline
    """

    model = models.Sequential(name="stealth_beholder_cnn")

    # Entrada
    model.add(layers.Input(shape=input_shape))

    # BLOQUE 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # BLOQUE 4
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # REDUCCIÓN
    model.add(layers.GlobalAveragePooling2D())

    # CLASIFICACIÓN
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))

    # SALIDA BINARIA (muy importante para vuestro dataset)
    model.add(layers.Dense(1, activation="sigmoid"))

    return model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:

    """
    Compilación compatible con el pipeline actual del proyecto.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model