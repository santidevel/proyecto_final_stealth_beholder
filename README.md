# Proyecto en grupo Stealth Beholder

# Fecha de entrega 17/04/2026

# - IMPORTANTE - 
## Usaremos Python 3.10.11 para mayor compatibilidad con Tensorflow y Docker.
## Recomiendo usar PyCharm, pero usad el IDE que os guste.
## Si queréis usar ramas distintas podéis, pero recomiendo trabajar sobre el main porque las tareas serán diferentes y no deberían de dar conflictos.
## Si existe código que ha hecho otra persona NO SE MODIFICA, se informa a la persona responsable.
## Según decisión mayoritaria, podemos ir documento en el README todos los pasos o podemos crear un documento, como os venga mejor, yo prefiero el README.
## Por favor, si hay cosas que pesan demasiado excluirlos en el gitignore y pasar los datos de otra manera, por ejemplo OneDrive.
## Ejemplo de la estructura del proyecto:
```
stealth-beholder/
├── README.md
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml              # opcional, pero útil en local
├── .dockerignore
├── train.py                        # script principal de entrenamiento
├── predict.py                      # inferencia rápida
├── export_onnx.py                  # exportación a ONNX
├── config/
│   └── config.yaml                 # rutas, batch size, epochs, etc.
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py            # pipeline
│   ├── model.py                    # arquitectura CNN
│   ├── training.py                 # funciones de entrenamiento
│   ├── utils.py                    # helpers
│   └── metrics.py                  # métricas, plots, etc.
├── scripts/
│   ├── prepare_data.py             # extracción / limpieza si hace falta
│   ├── run_train.sh
│   └── run_server.sh
├── data/
│   ├── sample/                     # mini-dataset para pruebas
│   └── .gitkeep
├── models/
│   └── .gitkeep                    # aquí se guardan .keras / .h5 / .onnx
├── notebooks/
│   └── pruebas.ipynb               # opcional, solo exploración
├── tests/
│   ├── test_pipeline.py
│   └── test_model.py
└── docs/
    └── informe.md
```

## Uso de los datos:
### No usaremos una base de datos, sino directamente una estructura de carpetas y el nombre de cada directorio servirá como etiqueta.
### Por ejemplo:
```
data/
└── sample/
    ├── valorant_combate/
    ├── valorant_menu/
    ├── genshin_exploracion/
    ├── eldenring_cinematica/
    └── cyberpunk_victoria/
```

## Documento original Aules:
[Ver informe](docs/ProyectoFinalDocAules.pdf)

## Documento del proyecto:
https://docs.google.com/document/d/1CSbMesr25Kp_Y2aJIR1nYDvv6XPtwEqfTmDE1_uJAGc/edit?usp=sharing

## URL a las tareas en Trello (En proceso):

# Parte 1: Borja y Javi

# Parte 2: Dennis

# Parte 3: Santi y Javi

# Parte 4: Pedro

# Parte 5: Kevin

# Parte 6: Andrés

# Parte 7: Hugo

# Parte 8: Dennis y Javi
