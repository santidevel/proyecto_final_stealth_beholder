from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

# ============================================================
# Configuración base
# ============================================================

IMG_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SEED: int = 42
CLASS_NAMES = ["capturas_juego", "capturas_real"]
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Validaciones
# ============================================================

def _validate_dataset_structure(data_dir: Path) -> None:
    """
    Comprueba que existan las carpetas esperadas dentro de data_dir.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la ruta del dataset: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"La ruta no es un directorio válido: {data_dir}")

    missing = [class_name for class_name in CLASS_NAMES if not (data_dir / class_name).exists()]
    if missing:
        raise ValueError(
            f"Estructura incorrecta en {data_dir}. "
            f"Se esperaban las carpetas: {CLASS_NAMES}. "
            f"Faltan: {missing}"
        )


# ============================================================
# Preprocesado
# ============================================================

def _preprocess(images: tf.Tensor, labels: tf.Tensor):
    """
    Normaliza las imágenes al rango [0, 1].
    """
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


# ============================================================
# Carga principal
# ============================================================

def load_datasets(
    data_dir: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    image_size: Tuple[int, int] = IMG_SIZE,
    validation_split: float = 0.2,
    seed: int = DEFAULT_SEED,
    cache: bool = False,
    augment_train: bool = False,  # se deja por compatibilidad futura
):
    """
    Carga train y validation desde una carpeta con dos subcarpetas:
      - capturas_juego
      - capturas_real

    Ejemplo esperado:
        data/sample/
            capturas_juego/
            capturas_real/

    Retorna:
        train_ds, val_ds, class_names
    """
    data_path = Path(data_dir)
    _validate_dataset_structure(data_path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_ds = train_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, CLASS_NAMES