from src.datapipeline import load_datasets


def main():
    # Cambia esta ruta a "data/full" cuando queráis probar el dataset grande
    DATA_DIR = "data/dataset"

    print("Cargando datasets...")

    train_ds, val_ds, class_names = load_datasets(
        data_dir=DATA_DIR,
        batch_size=32,
        image_size=(224, 224),
        validation_split=0.2,
        cache=False,
        augment_train=False,
    )

    print("\nCarga completada correctamente")
    print("Clases detectadas:", class_names)

    # Comprobación rápida del dataset de entrenamiento
    for images, labels in train_ds.take(1):
        print("\n[TRAIN]")
        print("Shape imágenes:", images.shape)
        print("Shape etiquetas:", labels.shape)
        print("Primeras etiquetas:", labels[:10].numpy().flatten())
        print("Valor mínimo imagen:", images.numpy().min())
        print("Valor máximo imagen:", images.numpy().max())

    # Comprobación rápida del dataset de validación
    for images, labels in val_ds.take(1):
        print("\n[VALIDATION]")
        print("Shape imágenes:", images.shape)
        print("Shape etiquetas:", labels.shape)
        print("Primeras etiquetas:", labels[:10].numpy().flatten())
        print("Valor mínimo imagen:", images.numpy().min())
        print("Valor máximo imagen:", images.numpy().max())

    print("\nPipeline OK")


if __name__ == "__main__":
    main()