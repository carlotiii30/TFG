import os
import shutil
import zipfile
from pathlib import Path

from fastapi import HTTPException

from api.utils.validation import validate_model_structure

MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)


def upload_model(model_zip_path: str, model_zip_file) -> str:
    with open(model_zip_path, "wb") as buffer:
        shutil.copyfileobj(model_zip_file, buffer)

    with zipfile.ZipFile(model_zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            member_path = Path(MODEL_DIR) / member
            if (
                not Path(member_path)
                .resolve()
                .is_relative_to(Path(MODEL_DIR).resolve())
            ):
                raise HTTPException(
                    status_code=400, detail="Invalid file path in ZIP archive"
                )
    model_dir_name = os.path.basename(model_zip_path).replace(".zip", "")
    model_path = os.path.join(MODEL_DIR, model_dir_name)
    os.makedirs(model_path, exist_ok=True)

    for member in zip_ref.namelist():
        member_path = Path(model_path) / member
        if (
            not Path(member_path)
            .resolve()
            .is_relative_to(Path(model_path).resolve())
        ):
            raise HTTPException(
                status_code=400, detail="Invalid file path in ZIP archive"
            )
        zip_ref.extract(member, model_path)
    validate_model_structure(model_path)

    os.remove(model_zip_path)

    return model_dir_name


def delete_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    shutil.rmtree(model_path)


def list_models() -> list:
    return os.listdir(MODEL_DIR)
