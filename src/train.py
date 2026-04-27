import tensorflow as tf
import os
from src.datapipeline import load_datasets
from src.model import create_model, compile_model

def train_and_evaluate():
    DATA_DIR = "data/dataset" 
    BATCH_SIZE = 16  
    EPOCHS = 40
    
    if not os.path.exists("models"):
        os.makedirs("models")

    print("----- ENTRENAMIENTO DEL MODELO -----")
    
    train_ds, val_ds, class_names = load_datasets(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        validation_split=0.2
    )

    model = create_model()
    model = compile_model(model, learning_rate=0.0001)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5, 
            patience=3, 
            min_lr=1e-6, 
            verbose=1
        )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    print("\n" + "="*30)
    print("   RESULTADOS FINALES")
    print("="*30)
    
    metrics = model.evaluate(val_ds, verbose=0)
    acc = metrics[1]
    
    print(f"\nPrecisión en Validación Final: {acc*100:.2f}%\n")
    model.save("models/optimized_model.h5")

if __name__ == "__main__":
    train_and_evaluate()