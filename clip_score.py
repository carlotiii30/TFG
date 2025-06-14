from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Cargar el modelo CLIP ViT-L/14
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Rutas de las imágenes y el prompt
image_path = "golden_retriever_before.png"
image_path_2 = "golden_retriever_after.png"
prompt = "a photo of a golden retriever dog in a park"


# Cargar y procesar imágenes
def compute_clip_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()


# Resultados
score_before = compute_clip_score(image_path, prompt)
score_after = compute_clip_score(image_path_2, prompt)

print(f"CLIP Score (antes): {score_before:.4f}")
print(f"CLIP Score (después): {score_after:.4f}")
