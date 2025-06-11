# TFG
Trabajo de Fin de Grado de Ingeniería Informática

## Modificación realizada al modelo: 

Se ha adaptado y entrenado el modelo Stable Diffusion utilizando la técnica de DreamBooth para personalizar la generación de imágenes de perros de razas específicas. Para ello, se ha seleccionado un subconjunto del dataset Stanford Dogs, filtrando varias razas y limitando el número de imágenes por clase. El modelo base se ha ajustado para aceptar estos datos, empleando técnicas como el uso de precisión mixta (float16), checkpointing de gradientes y reducción del tamaño de batch para optimizar el entrenamiento en GPU. Además, se ha implementado un pipeline de entrenamiento y generación de imágenes personalizado, permitiendo obtener resultados adaptados a las razas seleccionadas.