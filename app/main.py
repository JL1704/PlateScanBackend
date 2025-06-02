import os
import sys
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import numpy as np
from io import BytesIO

# Añadir yolov5 al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
yolov5_path = os.path.join(project_root, "yolov5")
sys.path.insert(0, yolov5_path)

# Importar desde yolov5 explícitamente
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar esto por seguridad en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelo
device = select_device('')
model_path = os.path.join(current_dir, "models/best.pt")
model = DetectMultiBackend(model_path, device=device, dnn=False)
model.eval()

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(image)

        # Preprocesar imagen
        img_resized = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_resized /= 255.0

        # Inferencia
        pred = model(img_resized, augment=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is None or len(pred) == 0:
            return JSONResponse(content={"message": "No se detectó ninguna placa"}, status_code=200)

        box = pred[0][:4].tolist()
        confidence = float(pred[0][4])
        class_id = int(pred[0][5])

        return {
            "box": box,
            "confidence": confidence,
            "class": class_id,
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

