import onnxruntime as ort
import numpy as np
import cv2
import os


def ejecutar_prediccion(image_path, model_path="outputs/models/modelo_final.onnx"):
    """
    Realiza la inferencia de una imagen usando el modelo ONNX para el sistema Stealth Beholder.
    """
    classes = ['capturas_juego', 'capturas_real']

    # 1. Verificación de existencia del modelo
    if not os.path.exists(model_path):
        # Intento de ruta alternativa por si el entorno Docker o la raíz cambian
        alt_path = os.path.join("models", "modelo_final.onnx")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return f"❌ Error: El modelo no existe en {model_path}", None

    # 2. Cargar sesión de ONNX
    try:
        # Cargamos el modelo en CPU (por defecto en ONNX Runtime si no hay GPU configurada)
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
    except Exception as e:
        return f"❌ Error al iniciar sesión ONNX: {e}", None

    # 3. Preprocesado de la imagen
    if not os.path.exists(image_path):
        return f"❌ Error: No encuentro la imagen en {image_path}", None

    img = cv2.imread(image_path)
    if img is None:
        return "❌ Error: El archivo no es una imagen válida", None

    # Conversión de color (OpenCV usa BGR, el modelo espera RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    # Normalización: exactamente igual a como se hizo en el datapipeline (0.0 a 1.0)
    img_final = img_resized.astype(np.float32) / 255.0
    # Añadir dimensión de batch: de (224, 224, 3) a (1, 224, 224, 3)
    img_final = np.expand_dims(img_final, axis=0)

    # 4. Inferencia
    # El resultado de session.run es una lista de outputs. Tomamos el primero [0].
    prediction = session.run(None, {input_name: img_final})

    # Extraemos el valor numérico. Usamos flatten() para asegurar que
    # obtenemos un array de una dimensión independientemente de los corchetes.
    raw_output = prediction[0].flatten()

    # 5. Lógica de Decisión para clasificación binaria
    # Si hay una sola neurona de salida (Sigmoid):
    # Valores cercanos a 0.0 -> capturas_juego
    # Valores cercanos a 1.0 -> capturas_real
    if len(raw_output) == 1:
        prob_real = float(raw_output[0])
        prob_juego = 1.0 - prob_real
    else:
        # En caso de que el modelo tenga dos neuronas de salida (Softmax)
        prob_juego = float(raw_output[0])
        prob_real = float(raw_output[1])

    # 6. Selección de resultado y confianza
    if prob_juego > prob_real:
        resultado = classes[0]
        confianza = prob_juego * 100
    else:
        resultado = classes[1]
        confianza = prob_real * 100

    # 7. Nota informativa de incertidumbre técnica
    if 45 <= confianza <= 55:
        print(f"⚠️ Nota: El modelo está muy indeciso (Confianza: {confianza:.2f}%)")

    return resultado, confianza