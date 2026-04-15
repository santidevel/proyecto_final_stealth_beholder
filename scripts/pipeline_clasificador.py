import os
import shutil
import cv2
import gc
import torch
import torch.nn.functional as F
import easyocr
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ==========================================
# RUTAS
# ==========================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROYECTO_DIR = os.path.dirname(SCRIPT_DIR)

def ruta(*partes):
    return os.path.join(PROYECTO_DIR, *partes)

RAW_DIRS = {
    "apex":  ruta("data", "raw_samples", "Apex_Legends"),
    "forza": ruta("data", "raw_samples", "Forza_Horizon"),
}

SEED_DIRS = {
    "apex_menus":    ruta("data", "semillas", "semillas_apex_menus"),
    "apex_victoria": ruta("data", "semillas", "semillas_apex_victoria"),
    "apex_jugando":  ruta("data", "semillas", "semillas_apex_jugando"),
    "forza_carrera": ruta("data", "semillas", "semillas_forza_carrera"),
    "forza_menus":   ruta("data", "semillas", "semillas_forza_menus"),
    "forza_primero": ruta("data", "semillas", "semillas_forza_primero"),
    "forza_victoria":ruta("data", "semillas", "semillas_forza_victoria"),
}

OUT_DIRS = {
    "apex_menus":       ruta("data", "sample", "apex_menus"),
    "apex_victoria":    ruta("data", "sample", "apex_victoria"),
    "apex_jugando":     ruta("data", "sample", "apex_jugando"),
    "apex_racha":       ruta("data", "sample", "apex_racha"),
    "apex_kill_leader": ruta("data", "sample", "apex_kill_leader"),
    "forza_carrera":    ruta("data", "sample", "forza_carrera"),
    "forza_menus":      ruta("data", "sample", "forza_menus"),
    "forza_primero":    ruta("data", "sample", "forza_primero"),
    "forza_victoria":   ruta("data", "sample", "forza_victoria"),
}

for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

# ==========================================
# PARÁMETROS
# ==========================================
KILLS_RACHA_MIN  = 3    # kills mínimos para racha
LIMPIAR_CADA_N   = 50   # limpiar VRAM cada N imágenes

# Color de la llama de Kill Leader en Apex: naranja/dorado brillante
# En BGR: B bajo, G medio, R alto
KL_COLOR_LOW  = np.array([0,  80, 180], dtype=np.uint8)   # BGR min
KL_COLOR_HIGH = np.array([80, 200, 255], dtype=np.uint8)  # BGR max

# Color de la calavera de kills: blanco/gris claro en HUD
SKULL_COLOR_LOW  = np.array([140, 140, 140], dtype=np.uint8)
SKULL_COLOR_HIGH = np.array([255, 255, 255], dtype=np.uint8)

dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# CARGAR MODELOS
# ==========================================
print(f"Cargando CLIP en {dispositivo.upper()}...")
modelo_clip     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dispositivo)
procesador_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Cargando EasyOCR...")
lector_ocr = easyocr.Reader(['en'], gpu=(dispositivo == "cuda"))

# ==========================================
# SEMILLAS CLIP
# ==========================================
print("\nEstudiando semillas...")
huellas_medias = {}

for categoria, ruta_carpeta in SEED_DIRS.items():
    if not os.path.exists(ruta_carpeta):
        print(f"  AVISO: no encontrada -> {ruta_carpeta}")
        continue
    imagenes = [f for f in os.listdir(ruta_carpeta)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imagenes:
        print(f"  AVISO: vacia -> {ruta_carpeta}")
        continue

    lista = []
    for img_name in imagenes:
        img_pil = Image.open(os.path.join(ruta_carpeta, img_name)).convert("RGB")
        inputs  = procesador_clip(text=[""], images=img_pil, return_tensors="pt").to(dispositivo)
        with torch.no_grad():
            h = modelo_clip(**inputs).image_embeds
        lista.append(h.cpu())
        del inputs

    promedio = torch.mean(torch.stack(lista), dim=0)
    huellas_medias[categoria] = F.normalize(promedio, p=2, dim=-1)
    print(f"  OK: {categoria} ({len(imagenes)} imagenes)")

if not huellas_medias:
    print("ERROR: Sin semillas.")
    exit(1)

categorias_keys = list(huellas_medias.keys())
matriz_huellas  = torch.cat([huellas_medias[k] for k in categorias_keys]).to(dispositivo)
print(f"\n{len(categorias_keys)} semillas listas.\n")

# ==========================================
# HELPERS CLIP
# ==========================================

def get_huella_clip(img_pil):
    inputs = procesador_clip(text=[""], images=img_pil, return_tensors="pt").to(dispositivo)
    with torch.no_grad():
        h = modelo_clip(**inputs).image_embeds
    del inputs
    return F.normalize(h, p=2, dim=-1)

def limpiar_gpu():
    if dispositivo == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# DETECCIÓN HUD - NUEVA LÓGICA SIN TEMPLATE MATCHING
#
# El HUD superior derecho de Apex siempre tiene:
#   [💀 N]  [X SQUADS LEFT]  [🔥 N]
#
# - 💀 N  = kills propios del jugador  → N>=3 es RACHA
# - 🔥 N  = kills del Kill Leader      → si ese N iguala o supera 💀 N
#            Y hay un icono de llama naranja → es KILL LEADER
#
# En 640p hay una segunda fila debajo:
#   [💀 N]  [💬 N]
#
# Estrategia:
#   1. Recortar el HUD superior derecho (x: 55%-100%, y: 0%-18%)
#   2. Escalar siempre a altura fija para que el OCR funcione igual
#      independientemente de la resolución original
#   3. Detectar la llama naranja (Kill Leader) por color HSV
#   4. Leer con OCR todos los números en esa zona
#   5. El número junto a la calavera = kills propios
#   6. Si hay llama naranja Y kills >= 1 → Kill Leader
#      Si no hay llama Y kills >= KILLS_RACHA_MIN → Racha
# ==========================================

# Altura fija a la que normalizamos el recorte HUD para el OCR
HUD_ALTURA_NORM = 120

def recortar_hud(img_cv):
    """Devuelve el recorte del HUD superior derecho, escalado a altura fija."""
    h, w = img_cv.shape[:2]
    y1, y2 = 0,          int(h * 0.18)
    x1, x2 = int(w * 0.55), w
    recorte = img_cv[y1:y2, x1:x2]
    if recorte.size == 0:
        return None, (x1, y1)
    # Escalar a altura fija manteniendo proporción
    factor  = HUD_ALTURA_NORM / recorte.shape[0]
    recorte = cv2.resize(recorte, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_CUBIC)
    return recorte, (x1, y1)

def detectar_llama_kl(recorte_bgr):
    """
    Detecta si hay un icono de llama naranja (Kill Leader) en el recorte.
    Usa máscara de color en HSV: naranja/amarillo brillante.
    Devuelve True si el área de píxeles naranja supera un umbral.
    """
    hsv    = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2HSV)
    # Naranja: H 5-25, S>150, V>150
    low1   = np.array([5,  150, 150])
    high1  = np.array([25, 255, 255])
    # Amarillo-dorado: H 25-35
    low2   = np.array([25, 150, 150])
    high2  = np.array([35, 255, 255])
    mascara = cv2.bitwise_or(
        cv2.inRange(hsv, low1, high1),
        cv2.inRange(hsv, low2, high2)
    )
    # El icono ocupa ~8% del ancho derecho del recorte
    zona_llama = mascara[:, int(mascara.shape[1] * 0.75):]
    pixeles_naranja = cv2.countNonZero(zona_llama)
    area_zona       = zona_llama.shape[0] * zona_llama.shape[1]
    ratio           = pixeles_naranja / max(area_zona, 1)
    return ratio > 0.04   # más del 4% de píxeles naranjas = llama presente

def leer_kills_hud(recorte_bgr):
    """
    Lee el número de kills del HUD con OCR.
    La calavera (💀) va seguida del número de kills.
    Devuelve el primer número entero de 0-99 encontrado en
    la mitad izquierda del recorte (donde está la calavera).
    """
    # Solo la mitad izquierda (donde está la calavera, no la llama)
    mitad_x = int(recorte_bgr.shape[1] * 0.55)
    zona    = recorte_bgr[:, :mitad_x]

    gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)

    mejores_kills = 0
    for thr in [0, 140, 180]:
        if thr == 0:
            # OTSU automático
            _, binaria = cv2.threshold(gris, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binaria = cv2.threshold(gris, thr, 255, cv2.THRESH_BINARY)

        resultados = lector_ocr.readtext(binaria, allowlist='0123456789')
        for (bbox, texto, conf) in resultados:
            texto = texto.strip()
            if texto.isdigit() and conf > 0.35:
                num = int(texto)
                if 0 <= num <= 99:
                    mejores_kills = max(mejores_kills, num)

    return mejores_kills

def detectar_tipo_gameplay(img_cv):
    """
    Analiza el HUD y devuelve:
      ("kill_leader", kills)  si hay llama naranja
      ("racha", kills)        si kills >= KILLS_RACHA_MIN sin llama
      (None, 0)               si jugando normal
    """
    recorte, _ = recortar_hud(img_cv)
    if recorte is None:
        return None, 0

    hay_llama = detectar_llama_kl(recorte)
    kills     = leer_kills_hud(recorte)

    if hay_llama and kills >= 1:
        return "kill_leader", kills

    if kills >= KILLS_RACHA_MIN:
        return "racha", kills

    return None, 0

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
stats = {k: 0 for k in list(OUT_DIRS.keys()) + ["errores"]}

def procesar_juego(juego, raw_dir):
    print(f"\n{'='*60}")
    print(f"  {juego.upper()} -> {raw_dir}")
    print(f"{'='*60}")

    if not os.path.exists(raw_dir):
        print(f"  Carpeta no existe: {raw_dir}")
        return

    archivos = [f for f in os.listdir(raw_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not archivos:
        print("  Carpeta vacia, saltando.")
        return

    print(f"  {len(archivos)} imagenes encontradas\n")

    for i, archivo in enumerate(tqdm(archivos, desc=f"Clasificando {juego}", unit="img")):

        if i > 0 and i % LIMPIAR_CADA_N == 0:
            limpiar_gpu()

        ruta_origen = os.path.join(raw_dir, archivo)
        try:
            img_cv  = cv2.imread(ruta_origen)
            if img_cv is None:
                raise ValueError("Imagen ilegible")

            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            huella  = get_huella_clip(img_pil)

            sims        = torch.matmul(huella.to(dispositivo), matriz_huellas.T)
            idx         = torch.argmax(sims).item()
            cat_visual  = categorias_keys[idx]
            del huella, sims

            # ======== APEX ========
            if juego == "apex":
                if cat_visual == "apex_menus":
                    shutil.move(ruta_origen, os.path.join(OUT_DIRS["apex_menus"], archivo))
                    stats["apex_menus"] += 1

                elif cat_visual == "apex_victoria":
                    shutil.move(ruta_origen, os.path.join(OUT_DIRS["apex_victoria"], archivo))
                    stats["apex_victoria"] += 1

                else:
                    # Toda imagen de gameplay pasa por análisis HUD
                    tipo, kills = detectar_tipo_gameplay(img_cv)

                    if tipo == "kill_leader":
                        shutil.move(ruta_origen, os.path.join(OUT_DIRS["apex_kill_leader"], archivo))
                        tqdm.write(f"  [KILL LEADER] kills={kills} | {archivo}")
                        stats["apex_kill_leader"] += 1

                    elif tipo == "racha":
                        shutil.move(ruta_origen, os.path.join(OUT_DIRS["apex_racha"], archivo))
                        tqdm.write(f"  [RACHA]       kills={kills} | {archivo}")
                        stats["apex_racha"] += 1

                    else:
                        shutil.move(ruta_origen, os.path.join(OUT_DIRS["apex_jugando"], archivo))
                        stats["apex_jugando"] += 1

            # ======== FORZA ========
            elif juego == "forza":
                mapa = {
                    "forza_carrera": "forza_carrera",
                    "forza_menus":   "forza_menus",
                    "forza_primero": "forza_primero",
                    "forza_victoria":"forza_victoria",
                }
                destino = mapa.get(cat_visual, "forza_carrera")
                shutil.move(ruta_origen, os.path.join(OUT_DIRS[destino], archivo))
                stats[destino] += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] {archivo}: {e}")
            stats["errores"] += 1

    limpiar_gpu()

# Ejecutar
procesar_juego("apex",  RAW_DIRS["apex"])
procesar_juego("forza", RAW_DIRS["forza"])

# ==========================================
# RESUMEN
# ==========================================
labels = {
    "apex_menus":       "Apex Menu    ",
    "apex_victoria":    "Apex Victoria",
    "apex_jugando":     "Apex Jugando ",
    "apex_racha":       "Apex Racha   ",
    "apex_kill_leader": "Apex KillLdr ",
    "forza_carrera":    "Forza Carrera",
    "forza_menus":      "Forza Menu   ",
    "forza_primero":    "Forza 1ro    ",
    "forza_victoria":   "Forza Vict.  ",
    "errores":          "ERRORES      ",
}
print(f"\n{'='*45}")
print("  RESUMEN FINAL")
print(f"{'='*45}")
for cat, total in stats.items():
    if total > 0:
        print(f"  {labels.get(cat, cat):<15} {total:>5} imagenes")
print(f"{'='*45}")
print("  Pipeline completado.")
