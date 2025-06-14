import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Cargar modelo y processor
model_id = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Rutas de las imágenes y el prompt
image_before = Image.open("golden_retriever_before.png").convert("RGB")
image_after = Image.open("golden_retriever_after.png").convert("RGB")
prompt = "a photo of a golden retriever dog in a park"


# Procesar ambas imágenes en el mismo batch con el mismo texto
inputs = processor(
    text=[prompt], images=[image_before, image_after], return_tensors="pt", padding=True
)

# Obtener logits y CLIP scores relativos
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=0)  # softmax sobre el batch

# Mostrar los resultados
clip_score_before = probs[0][0].item()
clip_score_after = probs[1][0].item()

print(f"CLIP Score relativo (antes): {clip_score_before:.4f}")
print(f"CLIP Score relativo (después): {clip_score_after:.4f}")
