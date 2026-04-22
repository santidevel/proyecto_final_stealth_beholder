from src.datapipeline import load_datasets
from src.model import create_model, compile_model

train_ds, val_ds, class_names = load_datasets(
    data_dir="data/dataset",
    batch_size=8,
    image_size=(224, 224),
    validation_split=0.2,
    cache=False,
    augment_train=False,
)

print("Clases:", class_names)

model = create_model()
model = compile_model(model)

model.fit(train_ds, validation_data=val_ds, epochs=1)