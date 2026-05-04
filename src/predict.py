import onnxruntime as ort
import numpy as np
import cv2
import os

def ejecutar_prediccion(image_path, model_path="outputs/models/modelo_final.onnx"):
    classes = ['capturas_juego', 'capturas_real']

    # 1. Verificación de existencia del modelo
    if not os.path.exists(model_path):
        alt_path = "data/models/modelo_final.onnx"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return f"❌ Error: El modelo no está en {model_path}", None

    # 2. Cargar sesión de ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # 3. Preprocesado de la imagen
    if not os.path.exists(image_path):
        return f"❌ Error: No encuentro la imagen en {image_path}", None

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))

    # Normalización idéntica al entrenamiento (0.0 a 1.0)
    img_final = img_resized.astype(np.float32) / 255.0
    img_final = np.expand_dims(img_final, axis=0)

    # 4. Inferencia
    prediction = session.run(None, {input_name: img_final})
    raw_output = prediction[0][0]

    # 5. Lógica de Decisión (Sigmoid)
    # En tu modelo, el valor tiende a 0.0 para 'juego' y a 1.0 para 'real'
    if np.isscalar(raw_output) or len(raw_output.flatten()) == 1:
        prob_real = float(raw_output)
        prob_juego = 1.0 - prob_real
    else:
        # Soporte para modelos multiclase (Softmax)
        prob_juego = float(raw_output[0])
        prob_real = float(raw_output[1])

    # 6. Clasificación por Mayoría (Objetiva)
    # Ahora el modelo elegirá simplemente la que tenga mayor probabilidad
    if prob_juego > prob_real:
        result_index = 0  # capturas_juego
        confianza = prob_juego * 100
    else:
        result_index = 1  # capturas_real
        confianza = prob_real * 100

    # 7. Aviso de incertidumbre (Solo para informarte hoy que el modelo es débil)
    if 45 <= confianza <= 55:
        print(f"⚠️ Nota: El modelo está muy indeciso (Confianza: {confianza:.2f}%)")

    return classes[result_index], confianza