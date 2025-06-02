from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
import os
import re
import easyocr
import sys

# ------------------ Agregar yolov5 al path ------------------
sys.path.append('yolov5')  # Asegúrate de haber clonado este repo en el mismo nivel que main.py

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# ------------------ Cargar modelo YOLOv5 ------------------

yolo_model_path = 'models/best.pt'
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"Modelo YOLOv5 no encontrado en: {yolo_model_path}")

device = select_device('cpu')  # Puedes usar 'cuda:0' si Render lo soportara
model = DetectMultiBackend(yolo_model_path, device=device)
stride, names, pt = model.stride, model.names, model.pt

# Inicializar OCR
reader = easyocr.Reader(['es'], gpu=False)

# Inicializar FastAPI
app = FastAPI()

# ------------------ Endpoint de detección ------------------

@app.post("/detect")
async def detect_plate(image: UploadFile = File(...)):
    try:
        # Leer y decodificar la imagen
        image_bytes = await image.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocesamiento
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.to(device)

        # Inferencia
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is None or len(pred) == 0:
            return JSONResponse(content={"plate": "", "success": False, "message": "No se detectó ninguna placa con YOLO"})

        # Coordenadas reales
        pred = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()
        x1, y1, x2, y2 = map(int, pred[0][:4])
        cropped_img = img[y1:y2, x1:x2]

        # OCR con EasyOCR
        ocr_results = reader.readtext(cropped_img)

        # Buscar formato de placa: ABC-123-X
        plate_pattern = re.compile(r'[A-Z]{3}-\d{3}-[A-Z]')
        encontrados = []

        for (_, texto, prob) in ocr_results:
            texto_filtrado = texto.strip().upper().replace(" ", "")
            if plate_pattern.match(texto_filtrado):
                encontrados.append((texto_filtrado, prob))

        if encontrados:
            plate = sorted(encontrados, key=lambda x: x[1], reverse=True)[0][0]
        else:
            plate = ""

        return JSONResponse(content={"plate": plate, "success": True})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
