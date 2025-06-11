# TFG
Trabajo de Fin de Grado de Ingeniería Informática


## Quickstart: Instalación y uso

### 1. Instalación de dependencias

Este proyecto utiliza [Poetry](https://python-poetry.org/) para la gestión de dependencias. Si no tienes Poetry instalado, puedes instalarlo con:
```bash
pip install poetry
```

Luego, instala las dependencias del proyecto:
```bash
poetry install
```

### 2. Ejecución del servidor FastAPI
Para iniciar la API ejecuta:
```bash
poetry run uvicorn api.main:app --reload
```

El servidor estará disponible en http://localhost:8000.

### 3. Uso de la API
Puedes acceder a la documentación interactiva de la API en http://localhost:8000/docs.

### Ejemplo de petición para generar una imagen
Puedes usar curl o herramientas como Postman para hacer peticiones a la API. Por ejemplo, para generar una imagen:
```bash
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "un perro golden retriever en la playa"}'
```

Consulta la documentación en /docs para ver todos los endpoints disponibles y sus parámetros.

## Documentación

### Modelo preentrenado utilizado

El modelo preentrenado elegido es **Stable Diffusion**, un modelo generativo de imágenes basado en difusión, ampliamente utilizado para tareas de generación de imágenes a partir de texto.


### Modificación realizada al modelo: 

Se ha adaptado y entrenado el modelo Stable Diffusion utilizando la técnica de DreamBooth para personalizar la generación de imágenes de perros de razas específicas. Para ello, se ha seleccionado un subconjunto del dataset Stanford Dogs, filtrando varias razas y limitando el número de imágenes por clase. El modelo base se ha ajustado para aceptar estos datos, empleando técnicas como el uso de precisión mixta (float16), checkpointing de gradientes y reducción del tamaño de batch para optimizar el entrenamiento en GPU. Además, se ha implementado un pipeline de entrenamiento y generación de imágenes personalizado, permitiendo obtener resultados adaptados a las razas seleccionadas.

### Dataset utilizado: Stanford Dogs

El dataset **Stanford Dogs** es un conjunto de datos de imágenes ampliamente utilizado en tareas de visión por computador, especialmente en clasificación y reconocimiento de razas de perros. Contiene más de 20.000 imágenes de alta calidad, distribuidas en 120 razas diferentes de perros, con anotaciones precisas para cada imagen. Este dataset es ideal para entrenar y evaluar modelos de aprendizaje profundo en tareas de clasificación de imágenes y generación de contenido relacionado con perros.

### Disponibilización mediante API

Para facilitar el uso y la integración del modelo, se ha desarrollado una API utilizando **FastAPI**. Esta API permite exponer las funcionalidades principales del modelo, como la generación de imágenes y la gestión de modelos personalizados, de forma sencilla y accesible a través de peticiones HTTP.
