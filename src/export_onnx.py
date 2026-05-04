import tensorflow as tf
import tf2onnx
import onnx
import os


def ejecutar_exportacion():
    """
    Convierte el modelo Keras (.h5) a formato ONNX (.onnx).
    Esta función es llamada por el main.py central.
    """
    # 1. Configuración de rutas simplificada para Docker (/app)
    # Usamos rutas relativas al directorio de trabajo actual para evitar errores de permisos.
    model_path = "models/optimized_model.h5"
    output_path = "output/models/modelo_final.onnx"

    # Asegurar que la carpeta de salida existe dentro del volumen mapeado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"--- 🧠 Cargando modelo Keras: {model_path} ---")

    # Verificación de existencia del modelo previo a la carga
    if not os.path.exists(model_path):
        print(f"❌ Error crítico: No se encuentra el archivo {model_path}")
        print(f"Directorio actual: {os.getcwd()}")
        return

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error al cargar el modelo H5: {e}")
        return

    # 2. Requisito del proyecto: Imágenes de 224x224 y 3 canales (RGB)[cite: 1]
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

    print("--- 🔄 Convirtiendo a ONNX (Modo Inferencia: training=False) ---")

    # Envolver el modelo en una tf.function fija los nodos.
    # Forzamos training=False para desactivar Dropout/BatchNormalization.[cite: 1]
    @tf.function(input_signature=spec)
    def model_fn(x):
        return model(x, training=False)

    # 3. Conversión usando la función concreta[cite: 1]
    try:
        model_proto, _ = tf2onnx.convert.from_function(
            model_fn,
            input_signature=spec,
            opset=13,
            output_path=output_path
        )
        print(f"✅ ¡Conversión finalizada!")
    except Exception as e:
        print(f"❌ Error durante la conversión de tf2onnx: {e}")
        return

    # 4. Verificación de integridad (Punto clave de tu Rol 7)[cite: 1]
    try:
        # Cargamos el modelo recién creado para validar que no está corrupto[cite: 1]
        check_model = onnx.load(output_path)
        onnx.checker.check_model(check_model)
        print(f"🛡️ Verificación exitosa: El archivo ONNX en {output_path} es válido.")
    except Exception as e:
        print(f"❌ Error de validación: El modelo exportado tiene problemas. Detalle: {e}")

    print(f"--- Proceso Stealth Beholder (Export) Completado ---")


if __name__ == "__main__":
    ejecutar_exportacion()