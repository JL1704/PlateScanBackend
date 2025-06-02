from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
import os
import re
import easyocr

# ------------------ Cargar modelo YOLOv5 ------------------

yolo_model_path = '../models/best.pt'
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"Modelo YOLOv5 no encontrado en: {yolo_model_path}")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=True)
model.cpu()

# Inicializar OCR
reader = easyocr.Reader(['es'], gpu=torch.cuda.is_available())

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

        # Detección con YOLOv5
        results = model(img)
        detections = results.xyxy[0]

        if detections.shape[0] == 0:
            return JSONResponse(content={"plate": "", "success": False, "message": "No se detectó ninguna placa con YOLO"})

        # Tomar la detección con mayor confianza
        box = detections[0].tolist()
        x1, y1, x2, y2 = map(int, box[:4])
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
