import os
import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoints.image_endpoints import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    os.makedirs("./models/stable_modified", exist_ok=True)
    yield
    shutil.rmtree("./outputs", ignore_errors=True)
    shutil.rmtree("./models/stable_modified", ignore_errors=True)


def test_generate_image_endpoint_model_not_found():
    response = client.post(
        "/generate/",
        json={
            "model_name": "no_existe",
            "prompt": "un perro feliz",
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


def test_generate_image_endpoint(monkeypatch):
    class DummyImage:
        def save(self, path):
            with open(path, "wb") as f:
                f.write(b"fake image content")

    class DummyPipe:
        def to(self, device):
            return self

        def __call__(self, prompt, num_inference_steps, guidance_scale):
            return type("Result", (), {"images": [DummyImage()]})()

    monkeypatch.setattr(
        "api.utils.image_operations.StableDiffusionPipeline.from_pretrained",
        lambda *a, **kw: DummyPipe(),
    )
    response = client.post(
        "/generate/",
        json={
            "model_name": "stable_modified",
            "prompt": "un perro feliz",
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
        },
    )
    assert response.status_code == 200
    assert "image_path" in response.json()
    assert os.path.exists(response.json()["image_path"])


def test_download_image_endpoint_not_found():
    response = client.get("/download/no_existe.png")
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_download_image_endpoint(monkeypatch):
    # Crear imagen dummy
    os.makedirs("./outputs", exist_ok=True)
    image_path = "./outputs/test.png"
    with open(image_path, "wb") as f:
        f.write(b"fake image content")
    response = client.get("/download/test.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
