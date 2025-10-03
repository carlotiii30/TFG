import os
import zipfile
from pathlib import Path
from PIL import Image
import shutil
import tarfile
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTokenizer, CLIPTextModel

# ========= CONFIGURACIÓN =========
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Fijar GPU 0

dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset_tar = "images.tar"
dataset_folder = "Images"
razas = ["n02099601-golden_retriever", "n02088364-beagle", "n02094114-Norfolk_terrier", "n02110958-pug"]
num_images_por_raza = 100
output_path = "dog_images"
output_model_dir = "stable-dog-output"
generadas_dir = "generadas"
instance_prompt = "a photo of a sks dog"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Usando dispositivo: {device}")
if device.type == "cuda":
    print(f"🧠 Nombre de GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# ========= 1. DESCARGA Y EXTRACCIÓN DEL DATASET =========
if not os.path.exists(dataset_folder):
    print("📥 Descargando el dataset Stanford Dogs...")
    os.system(f"wget {dataset_url}")
    print("📦 Extrayendo imágenes...")
    os.system(f"tar -xf {dataset_tar}")
    os.remove(dataset_tar)
    print("✅ Dataset descargado y extraído.")

if not os.path.exists(dataset_folder):
    print("📥 Descargando el dataset Stanford Dogs...")
    os.system(f"wget {dataset_url}")
    print("📦 Extrayendo solo las razas necesarias...")

    with tarfile.open(dataset_tar, "r") as tar:
        for raza in razas:
            members = [m for m in tar.getmembers() if m.name.startswith(f"Images/{raza}/")]
            tar.extractall(members=members)
    
    os.remove(dataset_tar)
    print("✅ Dataset filtrado, razas extraídas.")



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
print("🚀 Iniciando entrenamiento DreamBooth...")

# Cargar modelo base
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").to(device)

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    torch_dtype=torch.float16  # ✅ usamos float16
).to(device)

unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    torch_dtype=torch.float16  # ✅ usamos float16
).to(device)

unet.enable_gradient_checkpointing()  # ✅ activa checkpointing
scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

# Dataset personalizado
class DogDataset(Dataset):
    def __init__(self, image_dir, prompt, tokenizer, size=512):
        self.images = list(Path(image_dir).glob("*.jpg"))
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        tokens = self.tokenizer(
            self.prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        return {"pixel_values": image, "input_ids": tokens.input_ids.squeeze(0)}

dataset = DogDataset(output_path, instance_prompt, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)  # ✅ batch_size reducido

# Optimización
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)
unet.train()
vae.eval()
text_encoder.eval()


max_train_steps = 1000
for step in tqdm(range(max_train_steps)):
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() * 0.18215
            encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=torch.float16)

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

unet.save_pretrained(output_model_dir)
print(f"✅ Modelo guardado en '{output_model_dir}'")

# ========= 4. GENERACIÓN DE IMÁGENES =========
print("🖼️ Generando imágenes de prueba...")
pipe = StableDiffusionPipeline.from_pretrained(
    output_model_dir, torch_dtype=torch.float16
).to(device)

Path(generadas_dir).mkdir(exist_ok=True)

for raza_folder in razas:
    raza_nombre = raza_folder.split("-")[-1].replace("_", " ")
    prompt = f"a photo of a sks {raza_nombre}"
    image = pipe(prompt).images[0]
    out_path = os.path.join(generadas_dir, f"{raza_nombre}.png")
    image.save(out_path)
    print(f"✅ Imagen generada: {out_path}")

# ========= 5. COMPRESIÓN DE RESULTADOS =========
def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=folder_path)
                zipf.write(filepath, arcname)
    print(f"📦 Carpeta comprimida como: {output_zip_path}")

zip_folder(generadas_dir, "imagenes_generadas.zip")
