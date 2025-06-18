import os
import shutil
import zipfile
import io
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.endpoints.model_endpoints import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown():
    if os.path.exists("./models"):
        shutil.rmtree("./models")
    os.makedirs("./models", exist_ok=True)
    yield
    shutil.rmtree("./models", ignore_errors=True)


def create_dummy_zip(model_name="dummy_model"):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as zf:
        zf.writestr(f"{model_name}/model_index.json", "{}")
        zf.writestr(f"{model_name}/feature_extractor/preprocessor_config.json", "{}")
        zf.writestr(f"{model_name}/safety_checker/config.json", "{}")
        zf.writestr(f"{model_name}/safety_checker/model.safetensors", "a")
        zf.writestr(f"{model_name}/scheduler/scheduler_config.json", "{}")
        zf.writestr(f"{model_name}/text_encoder/config.json", "{}")
        zf.writestr(f"{model_name}/text_encoder/model.safetensors", "a")
        zf.writestr(f"{model_name}/tokenizer/merges.txt", "")
        zf.writestr(f"{model_name}/tokenizer/special_tokens_map.json", "{}")
        zf.writestr(f"{model_name}/tokenizer/tokenizer_config.json", "{}")
        zf.writestr(f"{model_name}/tokenizer/vocab.json", "{}")
        zf.writestr(f"{model_name}/unet/config.json", "{}")
        zf.writestr(f"{model_name}/unet/diffusion_pytorch_model.safetensors", "a")
        zf.writestr(f"{model_name}/vae/config.json", "{}")
        zf.writestr(f"{model_name}/vae/diffusion_pytorch_model.safetensors", "a")
    mem_zip.seek(0)
    return mem_zip


def test_upload_model_invalid_zip():
    response = client.post(
        "/upload/",
        files={"model_zip": ("bad.zip", io.BytesIO(b"notazip"), "application/zip")},
    )
    assert response.status_code == 400


def test_delete_model_success():
    zip_file = create_dummy_zip("to_delete")
    client.post(
        "/upload/", files={"model_zip": ("to_delete.zip", zip_file, "application/zip")}
    )
    response = client.delete("/delete/to_delete")
    assert response.status_code == 200
    assert "to_delete" not in os.listdir("./models")


def test_delete_model_not_found():
    response = client.delete("/delete/no_model")
    assert response.status_code == 404


def test_list_models():
    zip_file1 = create_dummy_zip("model1")
    zip_file2 = create_dummy_zip("model2")
    client.post(
        "/upload/", files={"model_zip": ("model1.zip", zip_file1, "application/zip")}
    )
    client.post(
        "/upload/", files={"model_zip": ("model2.zip", zip_file2, "application/zip")}
    )
    response = client.get("/list/")
    assert response.status_code == 200
    models = response.json()["models"]
    assert "model1" in models
    assert "model2" in models
