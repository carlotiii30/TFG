import os
from pathlib import Path
from PIL import Image
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTokenizer, CLIPTextModel

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
print("🚀 Iniciando entrenamiento DreamBooth...")

# Cargar modelo base
tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
).to(device)

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(device)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(device)
scheduler = DDPMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)


# Dataset personalizado
class DogDataset(Dataset):
    def __init__(self, image_dir, prompt, tokenizer, size=512):
        self.images = list(Path(image_dir).glob("*.jpg"))
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

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


def test_batch_size(dataset, device, start=1, step=1):
    batch_size = start
    last_successful = start
    while True:
        try:
            print(f"🔍 Probando batch_size={batch_size}")
            dataloader = DataLoader(
                dataset, batch_size=batch_size, pin_memory=True, num_workers=4
            )
            batch = next(iter(dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                latents = (
                    vae.encode(batch["pixel_values"]).latent_dist.sample() * 0.18215
                )
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)
            loss.backward()
            unet.zero_grad()

            torch.cuda.empty_cache()
            last_successful = batch_size
            batch_size += step

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Memoria agotada con batch_size={batch_size}")
                print(f"✅ Batch_size máximo ejecutable: {last_successful}")
                torch.cuda.empty_cache()
                break
            else:
                e.__traceback__.print_exc()
                break


# 🔥 Ejecutar la prueba
test_batch_size(dataset, device, start=4, step=4)
