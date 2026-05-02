import tensorflow as tf
import tf2onnx
import os

# 1. Configuración de rutas
model_path = '../models/optimized_model.h5'
output_path = '../output/models/modelo_final.onnx'

# Asegurar que la carpeta models existe
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

print(f"--- Cargando: {model_path} ---")
model = tf.keras.models.load_model(model_path)

# 2. Requisito del proyecto: Imágenes de 224x224 y 3 canales (RGB)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

print("--- Convirtiendo a ONNX (workaround estable para capas de aumento) ---")

# 🔥 CLAVE: Envolver el modelo en una tf.function fija los nodos.
# Forzamos training=False para desactivar capas aleatorias (RandomFlip/Dropout).
@tf.function(input_signature=spec)
def model_fn(x):
    return model(x, training=False)

# 3. Conversión usando la función concreta en lugar del objeto Keras directo
model_proto, _ = tf2onnx.convert.from_function(
    model_fn,
    input_signature=spec,
    opset=13
)

# 4. Serialización del "cerebro" final
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"✅ ¡Hecho! El modelo final está en: {output_path}")