from __future__ import annotations
import tensorflow as tf
from keras import layers, models, regularizers

def create_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """
    Modelo CNN Optimizado para Objetivo 90% - Rol 4.
    Equilibrio entre potencia de cálculo y regularización.
    """
    model = models.Sequential(name="stealth_beholder_target_90")

    # BLOQUES DE CONVOLUCIÓN CON BATCH NORMALIZATION
    # Bloque 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloque 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Bloque 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # REDUCCIÓN DE DIMENSIONALIDAD
    model.add(layers.GlobalAveragePooling2D())

    # CLASIFICADOR FINAL (Aumentamos a 128 neuronas para dar potencia)
    model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4)) 

    # SALIDA (Sigmoid para clasificación binaria)
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

def compile_model(model: tf.keras.Model, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Compilación con Adam y métricas de seguimiento.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model
