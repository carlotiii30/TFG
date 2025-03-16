import os

from fastapi import APIRouter, File, UploadFile

from api.utils.model_operations import delete_model, list_models, upload_model

router = APIRouter()

MODEL_DIR = "./models"


@router.post("/upload/")
async def upload_model_endpoint(model_zip: UploadFile = File(...)):
    model_zip_path = os.path.join(MODEL_DIR, model_zip.filename)
    model_dir_name = upload_model(model_zip_path, model_zip.file)
    return {
        "filename": model_zip.filename,
        "message": "Modelo subido exitosamente",
        "model_dir": model_dir_name,
    }
