import base64
import io
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as T

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
try:
    depth_anything_model = torch.hub.load(
        "isl-org/depth-anything", "depth_anything_v2_base", pretrained=True
    ).to(device).eval()
    da_transform = T.Compose([
        T.ToTensor(),
    ])
except Exception:
    depth_anything_model = None
    da_transform = None

try:
    midas_model = torch.hub.load("isl-org/MiDaS", "DPT_Large", pretrained=True).to(device).eval()
    midas_transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform
except Exception:
    midas_model = None
    midas_transform = None

try:
    marigold_model = torch.hub.load(
        "compvis/marigold", "marigold_vit_base", pretrained=True
    ).to(device).eval()
    marigold_transform = T.Compose([
        T.ToTensor(),
    ])
except Exception:
    marigold_model = None
    marigold_transform = None


def depth_to_base64(depth: np.ndarray) -> str:
    depth -= depth.min()
    depth /= max(depth.max(), 1e-6)
    depth_img = (depth * 255).astype(np.uint8)
    img = Image.fromarray(depth_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_model(model, transform, image: Image.Image) -> np.ndarray:
    if model is None or transform is None:
        raise RuntimeError("Model not available")
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        if prediction.dim() == 4:
            prediction = prediction.squeeze(0)
        if prediction.dim() == 3:
            prediction = prediction.mean(0)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(0).unsqueeze(0),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


@app.post("/depth-anything")
async def depth_anything(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    depth = run_model(depth_anything_model, da_transform, image)
    return {"depth": depth_to_base64(depth)}


@app.post("/midas")
async def midas(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    depth = run_model(midas_model, midas_transform, image)
    return {"depth": depth_to_base64(depth)}


@app.post("/marigold")
async def marigold(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    depth = run_model(marigold_model, marigold_transform, image)
    return {"depth": depth_to_base64(depth)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
