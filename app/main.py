import os
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort
import easyocr
import cv2
import gc  # recolección de basura

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar ONNX model
model_path = os.path.join(os.path.dirname(__file__), "models", "best.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Inicializar EasyOCR
reader = easyocr.Reader(['es'], gpu=False)  # No usar GPU en Render

# Regex para formato de placas ABC-123-X
plate_pattern = re.compile(r'[A-Z]{3}-\d{3}-[A-Z]')


@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(image)

        # Redimensionar para el modelo (YOLO input size)
        img_resized = np.array(Image.fromarray(img).resize((640, 640)))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Inferencia ONNX
        outputs = session.run(None, {input_name: img_input})
        pred = outputs[0][0]

        boxes = []
        for det in pred:
            conf = det[4]
            if conf > 0.25:
                x1, y1, x2, y2 = map(int, det[0:4])
                class_id = int(det[5])
                boxes.append((x1, y1, x2, y2, conf, class_id))

        if not boxes:
            return JSONResponse(content={"success": False, "message": "No se detectó ninguna placa"})

        # Tomar la detección con mayor confianza
        x1, y1, x2, y2, conf, class_id = sorted(boxes, key=lambda x: x[4], reverse=True)[0]
        cropped = img[y1:y2, x1:x2]

        # Redimensionar antes del OCR para reducir consumo
        cropped = cv2.resize(cropped, (320, 320))

        # OCR
        ocr_results = reader.readtext(cropped)
        encontrados = []

        for (_, texto, prob) in ocr_results:
            texto_filtrado = texto.strip().upper().replace(" ", "")
            if plate_pattern.match(texto_filtrado):
                encontrados.append((texto_filtrado, prob))

        # Limpiar memoria
        del img, img_resized, img_input, outputs, pred, cropped
        gc.collect()

        if encontrados:
            plate = sorted(encontrados, key=lambda x: x[1], reverse=True)[0][0]
            return {"plate": plate, "success": True}
        else:
            return {"plate": "", "success": False, "message": "Placa no legible"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
