import onnxruntime as ort
import numpy as np
import cv2
import os


def softmax(x):
    """Convierte los valores brutos del modelo en probabilidades reales."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def ejecutar_prediccion(image_path, model_path="outputs/models/modelo_final.onnx"):
    classes = ['capturas_juego', 'capturas_real']

    if not os.path.exists(model_path):
        return f"❌ Error: El modelo no está en {model_path}", None

    # Cargar sesión de ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Preprocesado con OpenCV (como en tu código original)
    if not os.path.exists(image_path):
        return f"❌ Error: No encuentro la imagen en {image_path}", None

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    img_final = img_resized.astype(np.float32) / 255.0
    img_final = np.expand_dims(img_final, axis=0)

    # Inferencia
    prediction = session.run(None, {input_name: img_final})
    raw_scores = prediction[0][0]

    # Tu lógica de Softmax
    probabilidades = softmax(raw_scores)
    result_index = np.argmax(probabilidades)
    confianza = probabilidades[result_index] * 100

    return classes[result_index], confianza