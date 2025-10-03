from diffusers import StableDiffusionPipeline

# Ruta donde se guardará el modelo
model_path = "./data/stable/models--CompVis--stable-diffusion-v1-4"

# Descargar el modelo de Hugging Face
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Guardar el modelo en la ruta especificada
pipe.save_pretrained(model_path)

print(f"Modelo descargado y guardado en {model_path}")
