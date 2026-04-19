# Proyecto en grupo Stealth Beholder

# Fecha de entrega 17/04/2026

# - IMPORTANTE -
- Usaremos Python 3.10.11 para mayor compatibilidad con Tensorflow y Docker.
- Recomiendo usar PyCharm, pero usad el IDE que os guste.
- Si queréis usar ramas distintas podéis, pero recomiendo trabajar sobre el main porque las tareas serán diferentes y no deberían de dar conflictos.
- Si existe código que ha hecho otra persona NO SE MODIFICA, se informa a la persona responsable.
- Según decisión mayoritaria, podemos ir documentando todos los pasos en el README o crear un documento aparte. De momento, se centralizará aquí.
- Si hay archivos que pesan demasiado, excluirlos en el `.gitignore` y compartirlos por otra vía, por ejemplo OneDrive.
- El proyecto debe ejecutarse dentro de Docker, usar un pipeline iterativo con `tf.data`, entrenar una CNN con entrada `(224, 224, 3)` y exportar el modelo final a ONNX. :contentReference[oaicite:0]{index=0}

---

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

---

# Uso de los datos:
- No usaremos una base de datos.
- Usaremos una estructura de carpetas y el nombre de cada directorio servirá como etiqueta.
- Esto encaja con el enfoque pedido en la práctica, ya que el PDF indica trabajar con imágenes en disco y usar tf.keras.utils.image_dataset_from_directory(...) dentro de un pipeline iterativo tf.data.
- Ejemplo:
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

---

## Documento del proyecto:
https://docs.google.com/document/d/1CSbMesr25Kp_Y2aJIR1nYDvv6XPtwEqfTmDE1_uJAGc/edit?usp=sharing

---

# Plan general del proyecto

El objetivo del proyecto es diseñar, entrenar y desplegar una CNN capaz de identificar,a partir de una captura de pantalla,
el videojuego y el estado de la partida.
Además, el proyecto debe ser portable mediante Docker, gestionar un volumen grande de imágenes sin cargar todo en memoria
y exportar el modelo final a formato ONNX.

---

## Parte 1: Borja y Javi
### Rol: Extracción, limpieza y organización de datos

#### Objetivo
Preparar el dataset que usará todo el proyecto. Esta parte es la base de las demás.

### Tareas detalladas

#### 1. Definir las clases finales del dataset
**Qué es:** decidir exactamente qué va a clasificar el modelo.

**Qué hay que hacer:**
- Elegir qué videojuegos se incluirán.
- Elegir qué estados se incluirán.
- Decidir si la clasificación será por videojuego o por videojuego + estado.
- Documentar la lista final de clases.

**Por qué es importante:**  
Si no se decide desde el principio, se mezclarán datos y luego el modelo aprenderá clases inconsistentes.

**Qué se entrega:**
- Lista cerrada de clases.
- Clases documentadas en README o `docs/`.

---

#### 2. Buscar fuentes de datos
**Qué es:** localizar imágenes o vídeos que sirvan como base del dataset.

**Qué hay que hacer:**
- Buscar datasets en Kaggle.
- Revisar si se puede usar YouTube-8M.
- Buscar vídeos o retransmisiones grabadas.
- Valorar si hace falta recopilación propia.

**Por qué es importante:**  
Una buena CNN necesita datos variados y suficientes para aprender bien.

**Qué se entrega:**
- Lista de fuentes utilizadas.
- Material bruto localizado o descargado.

---

#### 3. Descargar o recopilar material
**Qué es:** guardar los archivos originales antes de limpiarlos.

**Qué hay que hacer:**
- Crear una carpeta de trabajo para material bruto.
- Guardar imágenes o vídeos fuente.
- Separar por origen si hace falta.

**Por qué es importante:**  
Permite mantener separado el material original del dataset final.

**Qué se entrega:**
- Carpeta de material bruto organizada.

---

#### 4. Extraer frames de vídeos
**Qué es:** convertir vídeos en imágenes utilizables para la CNN.

**Qué hay que hacer:**
- Crear o adaptar un script en `scripts/prepare_data.py`.
- Extraer frames cada cierto número de segundos.
- Evitar sacar todos los frames para no llenar el disco con imágenes casi iguales.

**Por qué es importante:**  
El proyecto trabaja con imágenes individuales, no con vídeo directamente.

**Qué se entrega:**
- Script de extracción.
- Carpeta con frames extraídos.

---

#### 5. Eliminar imágenes corruptas
**Qué es:** eliminar archivos que puedan romper el entrenamiento.

**Qué hay que hacer:**
- Detectar archivos que no se puedan abrir.
- Borrar imágenes rotas o inválidas.
- Revisar extensiones incorrectas.

**Por qué es importante:**  
Una imagen corrupta puede hacer fallar la carga del dataset.

**Qué se entrega:**
- Dataset sin archivos corruptos.

---

#### 6. Eliminar imágenes irrelevantes o mal etiquetadas
**Qué es:** limpiar el dataset para que el modelo aprenda con datos útiles.

**Qué hay que hacer:**
- Borrar duplicados o casi duplicados.
- Borrar imágenes sin contenido útil.
- Quitar imágenes del juego equivocado.
- Quitar imágenes del estado equivocado.

**Por qué es importante:**  
Si el dataset está sucio, el modelo aprende mal y baja la precisión.

**Qué se entrega:**
- Dataset limpio y coherente.

---

#### 7. Organizar la estructura de carpetas
**Qué es:** dejar el dataset en un formato compatible con TensorFlow.

**Qué hay que hacer:**
- Crear carpetas por clase.
- Mover cada imagen a su clase correcta.
- Preparar tanto `data/sample/` como `data/full/`.

**Por qué es importante:**  
El pipeline se basará en `image_dataset_from_directory()`, que usa las carpetas como etiquetas.

**Qué se entrega:**
- `data/sample/` estructurado.
- `data/full/` estructurado.

---

#### 8. Crear mini-dataset de prueba
**Qué es:** una versión pequeña para que el resto del equipo empiece antes.

**Qué hay que hacer:**
- Seleccionar una muestra por clase.
- Mantener la estructura final real.
- Comprobar que carga sin errores.

**Por qué es importante:**  
Permite que Dennis, Santi y Javi trabajen antes de que exista el dataset completo.

**Qué se entrega:**
- Mini-dataset funcional en `data/sample/`.

---

#### 9. Crear el dataset completo
**Qué es:** preparar el conjunto grande para entrenamiento real.

**Qué hay que hacer:**
- Escalar el número de imágenes.
- Mantener mismas clases y estructura.
- Vigilar el equilibrio entre clases.

**Por qué es importante:**  
La práctica habla de trabajar con un conjunto muy grande de imágenes.

**Qué se entrega:**
- Dataset final en `data/full/`.

---

#### 10. Documentar el proceso de datos
**Qué es:** dejar constancia de cómo se construyó el dataset.

**Qué hay que hacer:**
- Anotar fuentes.
- Explicar criterios de limpieza.
- Indicar imágenes por clase.
- Registrar problemas encontrados.

**Por qué es importante:**  
Servirá para el informe final y para justificar decisiones.

**Qué se entrega:**
- Documentación en README o `docs/`.

---

## Parte 2: Javi
### Rol: Pipeline de datos

#### Objetivo
Cargar y preparar el dataset de forma eficiente sin colapsar la RAM, usando `tf.data`.

### Tareas detalladas

#### 1. Validar la estructura del dataset
**Qué es:** revisar que el dataset está preparado para ser leído correctamente.

**Qué hay que hacer:**
- Comprobar que cada carpeta corresponde a una clase.
- Comprobar que dentro hay imágenes válidas.
- Verificar que `sample` y `full` tienen formato correcto.

**Por qué es importante:**  
Si la estructura falla, el pipeline no podrá cargar nada.

**Qué se entrega:**
- Confirmación de compatibilidad con TensorFlow.

---

#### 2. Crear `src/data_pipeline.py`
**Qué es:** el módulo principal del flujo de datos.

**Qué hay que hacer:**
- Crear funciones o clase para cargar datasets.
- Centralizar aquí la lógica de lectura y optimización.

**Por qué es importante:**  
Evita mezclar lógica de datos dentro de `train.py`.

**Qué se entrega:**
- Archivo funcional `src/data_pipeline.py`.

---

#### 3. Implementar carga con `image_dataset_from_directory`
**Qué es:** cargar imágenes por lotes desde carpetas.

**Qué hay que hacer:**
- Usar `tf.keras.utils.image_dataset_from_directory()`.
- Configurar `image_size=(224,224)`.
- Configurar `batch_size`.
- Crear `validation_split`.
- Usar `subset="training"` y `subset="validation"`.

**Por qué es importante:**  
Es la forma recomendada para gestionar grandes volúmenes de imágenes.

**Qué se entrega:**
- `train_ds` y `val_ds`.

---

#### 4. Normalizar imágenes
**Qué es:** escalar los valores de píxeles.

**Qué hay que hacer:**
- Aplicar `Rescaling(1./255)` con `map()`.

**Por qué es importante:**  
Mejora la estabilidad del entrenamiento.

**Qué se entrega:**
- Datasets normalizados.

---

#### 5. Optimizar con `shuffle`, `cache` y `prefetch`
**Qué es:** acelerar el flujo de lectura.

**Qué hay que hacer:**
- Aplicar `shuffle()` en entrenamiento.
- Aplicar `cache()`.
- Aplicar `prefetch(tf.data.AUTOTUNE)`.

**Por qué es importante:**  
Evita cuellos de botella entre disco, CPU y GPU.

**Qué se entrega:**
- Pipeline optimizado.

---

#### 6. Probar con el mini-dataset
**Qué es:** validar el pipeline antes de escalar.

**Qué hay que hacer:**
- Comprobar shapes.
- Comprobar etiquetas.
- Confirmar carga sin errores.

**Por qué es importante:**  
Evita problemas grandes más adelante.

**Qué se entrega:**
- Prueba local correcta.

---

#### 7. Integrar con el modelo
**Qué es:** dejar el pipeline listo para que lo use la CNN.

**Qué hay que hacer:**
- Exponer una interfaz simple.
- Devolver `train_ds`, `val_ds` y `class_names`.

**Por qué es importante:**  
Permite que Santi y Javi entrenen sin tocar el pipeline.

**Qué se entrega:**
- Integración limpia con `train.py`.

---

#### 8. Comprobar compatibilidad con Docker
**Qué es:** asegurar que el pipeline funciona en el contenedor.

**Qué hay que hacer:**
- Usar rutas relativas.
- Evitar rutas personales.
- Adaptar a variables de entorno si hace falta.

**Por qué es importante:**  
La práctica exige transportabilidad.

**Qué se entrega:**
- Pipeline portable.

---

#### 9. Probar con el dataset grande
**Qué es:** validar el comportamiento con datos reales.

**Qué hay que hacer:**
- Ejecutar con `data/full/`.
- Comprobar que no explota la RAM.
- Medir rendimiento básico.

**Por qué es importante:**  
Esta es la prueba real del rol 2.

**Qué se entrega:**
- Pipeline escalado funcionando.

---

## Parte 3: Santi y Javi
### Rol: Arquitectura CNN

#### Objetivo
Diseñar la red neuronal convolucional que reciba imágenes `(224,224,3)` y clasifique correctamente las clases del dataset.

### Tareas detalladas

#### 1. Definir la entrada del modelo
**Qué es:** indicar el tamaño exacto que tendrá cada imagen al entrar al modelo.

**Qué hay que hacer:**
- Crear una capa de entrada con `shape=(224,224,3)`.
- Confirmar que coincide con el pipeline.

**Por qué es importante:**  
Si el pipeline da un tamaño y la CNN espera otro, fallará.

**Qué se entrega:**
- Input correcto en `src/model.py`.

---

#### 2. Añadir capas `Conv2D`
**Qué es:** capas que extraen patrones visuales como bordes, colores y texturas.

**Qué hay que hacer:**
- Crear varios bloques convolucionales.
- Empezar con una arquitectura sencilla.
- Aumentar progresivamente filtros: por ejemplo 32, 64, 128.

**Por qué es importante:**  
Estas capas son el núcleo de la visión artificial.

**Qué se entrega:**
- Primera arquitectura funcional.

---

#### 3. Añadir activaciones
**Qué es:** funciones que permiten al modelo aprender relaciones complejas.

**Qué hay que hacer:**
- Usar `relu` en capas intermedias.
- Usar `softmax` en la salida.

**Por qué es importante:**  
Sin activaciones, la red sería demasiado limitada.

**Qué se entrega:**
- Arquitectura activada correctamente.

---

#### 4. Añadir `MaxPooling2D`
**Qué es:** reducir dimensiones internas y ahorrar cómputo.

**Qué hay que hacer:**
- Colocar `MaxPooling2D((2,2))` después de algunos bloques.

**Por qué es importante:**  
Reduce tamaño, coste computacional y ayuda a extraer rasgos más robustos.

**Qué se entrega:**
- CNN más eficiente.

---

#### 5. Elegir entre `Flatten` o `GlobalAveragePooling2D`
**Qué es:** transformar la parte convolucional en entrada para clasificación.

**Qué hay que hacer:**
- Empezar con `Flatten` por simplicidad.
- Valorar `GlobalAveragePooling2D` como mejora.

**Por qué es importante:**  
Afecta al número de parámetros y al riesgo de overfitting.

**Qué se entrega:**
- Decisión técnica documentada.

---

#### 6. Añadir capas `Dense`
**Qué es:** parte final que convierte características en una predicción.

**Qué hay que hacer:**
- Añadir una capa intermedia.
- Añadir una capa final con `num_classes`.

**Por qué es importante:**  
Es la parte que decide la clase final.

**Qué se entrega:**
- Bloque de clasificación final.

---

#### 7. Compilar el modelo
**Qué es:** preparar el modelo para el entrenamiento.

**Qué hay que hacer:**
- Elegir optimizador.
- Elegir función de pérdida.
- Elegir métrica.

**Por qué es importante:**  
Sin compilar no se puede entrenar.

**Qué se entrega:**
- Modelo listo para `fit()`.

---

#### 8. Revisar `model.summary()`
**Qué es:** inspeccionar la arquitectura creada.

**Qué hay que hacer:**
- Revisar salidas capa a capa.
- Revisar número de parámetros.

**Por qué es importante:**  
Ayuda a detectar errores antes de entrenar.

**Qué se entrega:**
- Arquitectura validada.

---

#### 9. Integrar con `model.fit(train_ds)`
**Qué es:** conectar el modelo con los datos.

**Qué hay que hacer:**
- Importar datasets desde el pipeline.
- Lanzar entrenamiento con validación.

**Por qué es importante:**  
Demuestra que modelo y pipeline funcionan juntos.

**Qué se entrega:**
- Entrenamiento inicial funcional.

---

#### 10. Validar resultados iniciales
**Qué es:** observar si el modelo empieza a aprender.

**Qué hay que hacer:**
- Revisar `accuracy`, `val_accuracy`, `loss`, `val_loss`.
- Compartir resultados con Pedro.

**Por qué es importante:**  
Será la base de la optimización posterior.

**Qué se entrega:**
- Primeras métricas del modelo.

---

## Parte 4: Pedro
### Rol: Optimización del aprendizaje

#### Objetivo
Ajustar el modelo para mejorar precisión y evitar overfitting.

### Tareas detalladas

#### 1. Añadir `Dropout`
**Qué es:** una técnica para evitar que el modelo memorice demasiado.

**Qué hay que hacer:**
- Añadir una o varias capas `Dropout`.
- Probar distintos valores.

**Por qué es importante:**  
Ayuda a combatir el overfitting.

**Qué se entrega:**
- Modelo regularizado.

---

#### 2. Ajustar learning rate
**Qué es:** controlar cuánto aprende el modelo en cada paso.

**Qué hay que hacer:**
- Probar distintos valores.
- Comparar resultados.

**Por qué es importante:**  
Un mal learning rate puede arruinar el entrenamiento.

**Qué se entrega:**
- Learning rate recomendado.

---

#### 3. Ajustar batch size y epochs
**Qué es:** configurar el ritmo del entrenamiento.

**Qué hay que hacer:**
- Probar varios batch sizes.
- Ajustar número de épocas.

**Por qué es importante:**  
Afecta a memoria, velocidad y precisión.

**Qué se entrega:**
- Configuración afinada.

---

#### 4. Probar mejoras de arquitectura
**Qué es:** comparar pequeñas variantes del modelo base.

**Qué hay que hacer:**
- Comparar `Flatten` con `GlobalAveragePooling2D`.
- Ajustar número de filtros.
- Revisar capas intermedias.

**Por qué es importante:**  
Puede mejorar precisión y eficiencia.

**Qué se entrega:**
- Propuesta de mejor arquitectura.

---

#### 5. Detectar overfitting
**Qué es:** comprobar si el modelo memoriza en vez de generalizar.

**Qué hay que hacer:**
- Comparar entrenamiento y validación.
- Detectar separación excesiva entre `accuracy` y `val_accuracy`.

**Por qué es importante:**  
Un modelo con overfitting no sirve bien fuera de los datos de entrenamiento.

**Qué se entrega:**
- Diagnóstico y propuesta de corrección.

---

#### 6. Buscar superar el 90% de accuracy
**Qué es:** intentar llegar al objetivo de la rúbrica.

**Qué hay que hacer:**
- Iterar ajustes.
- Comparar configuraciones.
- Elegir la mejor.

**Por qué es importante:**  
Es uno de los objetivos principales de calidad del modelo.

**Qué se entrega:**
- Versión optimizada final.

---

## Parte 5: Kevin
### Rol: Docker y portabilidad

#### Objetivo
Preparar un entorno portable para que el proyecto funcione en cualquier máquina con un solo comando.

### Tareas detalladas

#### 1. Crear el `Dockerfile`
**Qué es:** archivo principal del entorno portable.

**Qué hay que hacer:**
- Definir imagen base.
- Configurar carpeta de trabajo.
- Copiar proyecto.
- Instalar dependencias.
- Definir comando de arranque.

**Por qué es importante:**  
Es la base de la transportabilidad.

**Qué se entrega:**
- `Dockerfile` funcional.

---

#### 2. Instalar dependencias del proyecto
**Qué es:** asegurar que todo lo necesario está dentro del contenedor.

**Qué hay que hacer:**
- Instalar TensorFlow.
- Instalar `opencv-python-headless`.
- Instalar `tf2onnx`.
- Instalar librerías necesarias del proyecto.

**Por qué es importante:**  
Sin dependencias correctas, el contenedor no ejecutará el proyecto.

**Qué se entrega:**
- Entorno completo.

---

#### 3. Crear `.dockerignore`
**Qué es:** archivo para evitar copiar elementos pesados o innecesarios.

**Qué hay que hacer:**
- Excluir `.git`, cachés, modelos pesados y datos grandes.

**Por qué es importante:**  
Hace que la build sea más rápida y ligera.

**Qué se entrega:**
- `.dockerignore` correcto.

---

#### 4. Probar `docker build`
**Qué es:** construir la imagen del proyecto.

**Qué hay que hacer:**
- Ejecutar la build.
- Corregir errores si aparecen.

**Por qué es importante:**  
Comprueba que el entorno se puede crear.

**Qué se entrega:**
- Imagen construida sin errores.

---

#### 5. Probar `docker run`
**Qué es:** ejecutar el proyecto desde el contenedor.

**Qué hay que hacer:**
- Lanzar el contenedor.
- Probar entrenamiento o prueba mínima.

**Por qué es importante:**  
No basta con construir; tiene que ejecutarse de verdad.

**Qué se entrega:**
- Ejecución correcta del proyecto dentro de Docker.

---

#### 6. Depurar errores de entorno
**Qué es:** resolver problemas de dependencias, rutas o ejecución.

**Qué hay que hacer:**
- Corregir fallos de instalación.
- Revisar compatibilidades.
- Validar que el entorno sea estable.

**Por qué es importante:**  
La transportabilidad depende de que Docker sea estable.

**Qué se entrega:**
- Docker estable y usable.

---

## Parte 6: Andrés
### Rol: Servidor

#### Objetivo
Ejecutar el entrenamiento real en el servidor potente del aula.

### Tareas detalladas

#### 1. Conectarse al servidor por SSH
**Qué es:** acceder a la máquina potente del aula.

**Qué hay que hacer:**
- Conectarse con `ssh usuario@IA-SERVER-XX`.

**Por qué es importante:**  
El entrenamiento real debe hacerse en la máquina preparada para ello.

**Qué se entrega:**
- Acceso confirmado al servidor.

---

#### 2. Subir o clonar el proyecto
**Qué es:** llevar el repositorio al servidor.

**Qué hay que hacer:**
- Clonar el repo o copiarlo al servidor.
- Comprobar que están todos los archivos necesarios.

**Por qué es importante:**  
Sin el proyecto no se puede lanzar el entrenamiento.

**Qué se entrega:**
- Proyecto disponible en servidor.

---

#### 3. Lanzar Docker en el servidor
**Qué es:** ejecutar el entorno preparado por Kevin.

**Qué hay que hacer:**
- Construir imagen.
- Ejecutar contenedor.
- Verificar que accede al proyecto y a los datos.

**Por qué es importante:**  
Es la forma portable de ejecutar todo.

**Qué se entrega:**
- Entrenamiento lanzado en servidor.

---

#### 4. Montar acceso al dataset grande
**Qué es:** conectar el dataset real al entorno de entrenamiento.

**Qué hay que hacer:**
- Comprobar rutas.
- Montar volúmenes o rutas necesarias.

**Por qué es importante:**  
Sin `data/full` no hay entrenamiento real.

**Qué se entrega:**
- Acceso correcto a los datos.

---

#### 5. Monitorizar GPU con `nvidia-smi`
**Qué es:** vigilar temperatura, memoria y uso de la GPU.

**Qué hay que hacer:**
- Ejecutar `nvidia-smi`.
- Revisar uso de VRAM y carga.

**Por qué es importante:**  
Permite saber si la GPU está trabajando correctamente.

**Qué se entrega:**
- Seguimiento del entrenamiento.

---

#### 6. Guardar modelo y resultados
**Qué es:** conservar lo que sale del entrenamiento.

**Qué hay que hacer:**
- Guardar pesos entrenados.
- Guardar métricas o logs.
- Dejar el mejor modelo accesible.

**Por qué es importante:**  
Hugo necesitará ese modelo para exportarlo a ONNX.

**Qué se entrega:**
- Modelo entrenado final.

---

## Parte 7: Hugo
### Rol: ONNX e inferencia

#### Objetivo
Exportar el modelo entrenado a formato ONNX y probarlo con inferencia.

### Tareas detalladas

#### 1. Exportar el modelo a ONNX
**Qué es:** convertir el modelo entrenado a un formato universal.

**Qué hay que hacer:**
- Crear o completar `export_onnx.py`.
- Ejecutar conversión del modelo final.

**Por qué es importante:**  
Es un requisito de entrega del proyecto.

**Qué se entrega:**
- Archivo `.onnx`.

---

#### 2. Verificar que el archivo no está corrupto
**Qué es:** comprobar que ONNX funciona de verdad.

**Qué hay que hacer:**
- Cargar el archivo exportado.
- Confirmar que se puede usar.

**Por qué es importante:**  
A veces la exportación termina pero el archivo resultante no sirve.

**Qué se entrega:**
- ONNX validado.

---

#### 3. Crear script de inferencia
**Qué es:** probar una predicción con una sola imagen.

**Qué hay que hacer:**
- Crear o completar `predict.py`.
- Cargar una imagen.
- Redimensionarla.
- Ejecutar predicción.
- Mostrar clase resultante.

**Por qué es importante:**  
Demuestra que el modelo exportado es usable.

**Qué se entrega:**
- Script de inferencia funcional.

---

#### 4. Comparar inferencia ONNX con el modelo original
**Qué es:** revisar que el comportamiento sea consistente.

**Qué hay que hacer:**
- Probar una o varias imágenes en ambos modelos.
- Comparar salidas.

**Por qué es importante:**  
Sirve para validar la exportación.

**Qué se entrega:**
- Comparación documentada.

---

## Parte 8: Dennis y Javi
### Rol: Gestión, GitHub, README e informe

#### Objetivo
Coordinar el repositorio, integrar el trabajo del equipo y documentar correctamente el proyecto, incluyendo el uso ético de IA.

### Tareas detalladas

#### 1. Crear y mantener el repositorio
**Qué es:** centralizar el proyecto en GitHub.

**Qué hay que hacer:**
- Mantener la estructura de carpetas.
- Revisar que cada uno respete su parte.
- Evitar cambios cruzados sin avisar.

**Por qué es importante:**  
Reduce conflictos y desorden.

**Qué se entrega:**
- Repositorio organizado.

---

#### 2. Configurar `.gitignore`
**Qué es:** evitar subir archivos pesados o innecesarios.

**Qué hay que hacer:**
- Excluir datasets grandes.
- Excluir modelos pesados.
- Excluir cachés y entornos virtuales.

**Por qué es importante:**  
El repo debe contener código y documentación, no archivos gigantes.

**Qué se entrega:**
- `.gitignore` correcto.

---

#### 3. Mantener README o documentación principal
**Qué es:** explicar cómo se usa el proyecto.

**Qué hay que hacer:**
- Documentar estructura.
- Documentar ejecución.
- Documentar entrenamiento.
- Documentar Docker y ONNX.

**Por qué es importante:**  
Hace el proyecto entendible para cualquiera.

**Qué se entrega:**
- README actualizado.

---

#### 4. Coordinar integración entre partes
**Qué es:** asegurar que todo encaja correctamente.

**Qué hay que hacer:**
- Verificar conexión entre datos, pipeline, modelo, Docker, servidor y exportación.
- Revisar errores de integración.

**Por qué es importante:**  
Muchas veces cada módulo funciona por separado pero no junto.

**Qué se entrega:**
- Flujo general integrado.

---

#### 5. Documentar el uso ético de IA
**Qué es:** explicar cómo se ha usado la IA generativa como apoyo.

**Qué hay que hacer:**
- Preguntar al equipo qué usó cada uno.
- Registrar en qué ayudó la IA.
- Dejar claro que la implementación y validación final fueron del grupo.

**Por qué es importante:**  
Forma parte de la evaluación del proyecto.

**Qué se entrega:**
- Sección ética/documental en README o informe.

---

#### 6. Preparar la entrega final
**Qué es:** revisar que todo esté listo antes de entregar.

**Qué hay que hacer:**
- Revisar Docker.
- Revisar modelo.
- Revisar ONNX.
- Revisar documentación.
- Revisar estructura final.

**Por qué es importante:**  
Evita fallos de última hora.

**Qué se entrega:**
- Proyecto listo para entregar.

---

## Checklist general del proyecto

- [ ] Dataset organizado por carpetas
- [ ] Mini-dataset de prueba disponible
- [ ] Dataset completo disponible
- [ ] Pipeline con `tf.data`
- [ ] Uso de `cache()` y `prefetch()`
- [ ] CNN funcionando con entrada `(224,224,3)`
- [ ] Modelo entrenando con validación
- [ ] Ajustes de optimización aplicados
- [ ] Docker funcional
- [ ] Entrenamiento en servidor
- [ ] Exportación a ONNX
- [ ] Script de inferencia funcionando
- [ ] README e informe actualizados
- [ ] Uso ético de IA documentado

---

## Reparto rápido

- **Parte 1:** Borja y Javi
- **Parte 2:** Dennis
- **Parte 3:** Santi y Javi
- **Parte 4:** Pedro
- **Parte 5:** Kevin
- **Parte 6:** Andrés
- **Parte 7:** Hugo
- **Parte 8:** Dennis y Javi
