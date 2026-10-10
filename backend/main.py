from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_code import UNet
from backend.database import engine, get_db
from backend.models import Prediction
from backend import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
model = UNet(dim_in=1, dim_out=1, init_dim=32, mults=(1, 2, 4, 8))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "best_unet_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded successfully!")

PATCH_SIZE = 128
STRIDE = 64

def pad_image_to_fit_patches(img_np, patch_size=128, stride=64):
    h, w = img_np.shape[:2]
    
    target_h = h if h >= patch_size else patch_size
    target_w = w if w >= patch_size else patch_size
    
    if (target_h - patch_size) % stride != 0:
        target_h = patch_size + math.ceil((target_h - patch_size) / stride) * stride
        
    if (target_w - patch_size) % stride != 0:
        target_w = patch_size + math.ceil((target_w - patch_size) / stride) * stride

    pad_h = target_h - h
    pad_w = target_w - w

    padded_img = np.pad(img_np, ((0, pad_h), (0, pad_w)), mode='edge')
    return padded_img, (h, w)

def extract_patches(img_np, patch_size=128, stride=64):
    patches = []
    positions = []
    h, w = img_np.shape[:2]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img_np[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))
    return patches, positions

def merge_patches(predicted_patches, positions, image_shape, patch_size=128):
    full_mask = np.zeros(image_shape, dtype=np.float32)
    counts = np.zeros(image_shape, dtype=np.float32)

    for patch, (i, j) in zip(predicted_patches, positions):
        full_mask[i:i+patch_size, j:j+patch_size] += patch
        counts[i:i+patch_size, j:j+patch_size] += 1.0

    counts[counts == 0] = 1.0
    reconstructed = full_mask / counts
    return reconstructed


@app.get("/")
def home():
    return {"message": "ISBI2012 API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  
    img_np = np.array(image, dtype=np.float32)
    orig_h, orig_w = img_np.shape[:2]
    padded_img_np, (orig_h, orig_w) = pad_image_to_fit_patches(img_np, PATCH_SIZE, STRIDE)
    padded_h, padded_w = padded_img_np.shape[:2]

    padded_img_norm = padded_img_np / 255.0
    patches, positions = extract_patches(padded_img_norm, PATCH_SIZE, STRIDE)
    patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        outputs = model(patches_tensor)
        predicted_probs = torch.sigmoid(outputs).squeeze(1).numpy()
    padded_prob_map = merge_patches(predicted_probs, positions, (padded_h, padded_w), PATCH_SIZE)
    
    cropped_prob_map = padded_prob_map[:orig_h, :orig_w]
    binary_mask = (cropped_prob_map > 0.5).astype(np.uint8) * 255
    pred_image = Image.fromarray(binary_mask, mode="L")
    pred_buffer = io.BytesIO()
    pred_image.save(pred_buffer, format="PNG")
    pred_bytes = pred_buffer.getvalue()

    orig_buffer = io.BytesIO()
    image.save(orig_buffer, format="PNG")
    orig_bytes = orig_buffer.getvalue()

    db_prediction = Prediction(
        original_image=orig_bytes,
        predicted_image=pred_bytes,
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)


    return {
        "id": db_prediction.id,
        "original_image": base64.b64encode(orig_bytes).decode("utf-8"),
        "predicted_image": base64.b64encode(pred_bytes).decode("utf-8"),
    }


@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.id.desc()).all()
    result = []
    for p in predictions:
        result.append({
            "id": p.id,
            "created_at": str(p.created_at),
        })
    return result


@app.get("/history/{pred_id}")
def get_prediction(pred_id: int, db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == pred_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {
        "id": prediction.id,
        "created_at": str(prediction.created_at),
        "original_image": base64.b64encode(prediction.original_image).decode("utf-8"),
        "predicted_image": base64.b64encode(prediction.predicted_image).decode("utf-8"),
    }


@app.delete("/history/{pred_id}")
def delete_prediction(pred_id: int, db: Session = Depends(get_db)):
    prediction = db.query(Prediction).filter(Prediction.id == pred_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(prediction)
    db.commit()
    return {"message": f"Prediction {pred_id} deleted successfully"}