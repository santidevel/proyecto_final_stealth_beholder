import onnx
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent[1]
model_path = str(ROOT / "output" / "models" / "modelo_final.onnx")

print(f"--- Verificando integridad de: {model_path} ---")

if not os.path.exists(model_path):
    print(f"❌ Error: No se encuentra el archivo en {model_path}. Ejecuta primero export_onnx.py")
else:
    try:
        # Cargar el modelo
        model = onnx.load(model_path)
        # Comprobar la estructura del grafo
        onnx.checker.check_model(model)
        print("✅ ¡Éxito! El modelo ONNX es válido y la estructura es consistente.")

        # Opcional: Imprimir información básica para documentar
        print(f"Versión del Opset: {model.opset_import[0].version}")
    except Exception as e:
        print(f"❌ Error: El modelo parece estar corrupto o es inválido. Detalle: {e}")