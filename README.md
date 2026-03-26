# Proyecto en grupo Stealth Beholder

# IMPORTANTE: Intentemos seguir esta estructura del proyecto:

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

## Documento del proyecto:
https://docs.google.com/document/d/1CSbMesr25Kp_Y2aJIR1nYDvv6XPtwEqfTmDE1_uJAGc/edit?usp=sharing

# Parte 1: Dennis y Borja

# Parte 2: Dennis

# Parte 3: Santi y Javi

# Parte 4: Pedro

# Parte 5: Kevin

# Parte 6: Andrés

# Parte 7: Hugo

# Parte 8: Dennis y Javi
