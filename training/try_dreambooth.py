import os
import shutil
from pathlib import Path
from PIL import Image
import torch
import subprocess
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline

# ========= CONFIGURACIÓN =========
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset_tar = "images.tar"
dataset_folder = "Images"
razas = ["n02099601-golden_retriever", "n02088238-basset", "n02091134-whippet"]
num_images_por_raza = 100
output_path = "dog_images"
output_model_dir = "stable-dog-output"
generadas_dir = "generadas"
instance_prompt = "a photo of a sks dog"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"🧠 Usando dispositivo: {device}")

# ========= 1. DESCARGA Y EXTRACCIÓN DEL DATASET =========
if not os.path.exists(dataset_folder):
    print("📥 Descargando el dataset Stanford Dogs...")
    os.system(f"wget {dataset_url}")
    print("📦 Extrayendo imágenes...")
    os.system(f"tar -xf {dataset_tar}")
    os.remove(dataset_tar)
    print("✅ Dataset descargado y extraído.")

# ========= 2. SELECCIÓN DE IMÁGENES =========
os.makedirs(output_path, exist_ok=True)
for raza_folder in razas:
    raza_nombre = raza_folder.split("-")[-1]
    files = os.listdir(os.path.join(dataset_folder, raza_folder))[:num_images_por_raza]
    for i, fname in enumerate(files):
        origen = os.path.join(dataset_folder, raza_folder, fname)
        destino = os.path.join(output_path, f"{raza_nombre}_{i}.jpg")
        shutil.copy(origen, destino)
print(f"✅ Copiadas {num_images_por_raza * len(razas)} imágenes a '{output_path}'.")

# ========= 3. ENTRENAMIENTO REAL CON DREAMBOOTH =========
print("🚀 Entrenando con DreamBooth REAL usando diffusers...")

train_command = [
    "python",
    "diffusers/examples/dreambooth/train_dreambooth.py",
    "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4",
    f"--instance_data_dir={output_path}",
    f"--output_dir={output_model_dir}",
    f"--instance_prompt={instance_prompt}",
    "--resolution=512",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=1",
    "--learning_rate=5e-6",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--max_train_steps=1000",
    "--mixed_precision=fp16" if torch.cuda.is_available() else "--mixed_precision=no",
    "--use_8bit_adam",
]

subprocess.run(train_command, check=True)
print("✅ Entrenamiento finalizado y modelo guardado.")

# ========= 4. GENERACIÓN DE IMÁGENES =========
print("🖼️ Generando imágenes de prueba con el modelo entrenado...")
pipe = StableDiffusionPipeline.from_pretrained(
    output_model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

Path(generadas_dir).mkdir(exist_ok=True)

for raza_folder in razas:
    raza_nombre = raza_folder.split("-")[-1].replace("_", " ")
    prompt = f"a photo of a sks {raza_nombre}"
    image = pipe(prompt).images[0]
    out_path = os.path.join(generadas_dir, f"{raza_nombre}.png")
    image.save(out_path)
    print(f"✅ Imagen generada: {out_path}")
