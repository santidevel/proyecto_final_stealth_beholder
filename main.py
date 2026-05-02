from src.datapipeline import load_datasets
from src.predict import ejecutar_prediccion


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

    # Ruta relativa desde la raíz del proyecto
    foto = "data/dataset/capturas_real/animal_0001.jpg"
    modelo = "outputs/models/modelo_final.onnx"

    print(f"--- Iniciando Sistema Stealth Beholder ---")
    resultado, score = ejecutar_prediccion(foto, modelo)

    if score:
        print(f"La IA dice: {resultado} con un {score:.2f}% de confianza.")
    else:
        print(resultado)  # Imprime el error si no hay score

if __name__ == "__main__":
    main()