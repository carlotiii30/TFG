import os

from fastapi import APIRouter, File, UploadFile

from api.utils.model_operations import delete_model, list_models, upload_model

router = APIRouter()

MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)


@router.post("/upload/")
async def upload_model_endpoint(model_zip: UploadFile = File(...)):
    sanitized_filename = os.path.basename(model_zip.filename)
    model_zip_path = os.path.join(MODEL_DIR, sanitized_filename)
    model_dir_name = upload_model(model_zip_path, model_zip.file)
    return {
        "filename": model_zip.filename,
        "message": "Model uploaded successfully",
        "model_dir": model_dir_name,
    }


@router.delete("/delete/{model_name}")
async def delete_model_endpoint(model_name: str):
    delete_model(model_name)
    return {"message": f"Model '{model_name}' deleted successfully"}


@router.get("/list/")
def list_models_endpoint():
    models = list_models()
    return {"models": models}
