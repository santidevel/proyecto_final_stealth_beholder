import os
from src.train import train_and_evaluate
from src.export_onnx import ejecutar_exportacion
from src.predict import ejecutar_prediccion


def main():
    # Elige tu carpeta aquí
    DATA_DIR = "data/dataset"

    # Rutas relativas a la raíz del proyecto (/app en Docker)
    PATH_H5 = "models/optimized_model.h5"
    PATH_ONNX = "output/models/modelo_final.onnx"
    FOTO_TEST = "data/dataset/capturas_real/arquitectura_0001.jpg"

    print(f"--- 🛡️ SISTEMA STEALTH BEHOLDER ---")

    # FASE 1: Entrenamiento
    if not os.path.exists(PATH_H5):
        train_and_evaluate(data_path=DATA_DIR)

    # FASE 2: Exportación
    # Forzamos la exportación para refrescar el ONNX siempre
    ejecutar_exportacion()

    # FASE 3: Predicción
    if os.path.exists(FOTO_TEST):
        print(f"\n[TEST FINAL] Analizando: {FOTO_TEST}")
        resultado, score = ejecutar_prediccion(FOTO_TEST, PATH_ONNX)
        print(f"\nRESULTADO: {resultado} ({score:.2f}%)")
    else:
        print(f"❌ No se encontró la imagen de prueba en {FOTO_TEST}")


if __name__ == "__main__":
    main()