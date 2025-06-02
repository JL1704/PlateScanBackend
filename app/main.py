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

# Inicializar FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo ONNX
model_path = os.path.join(os.path.dirname(__file__), "models", "best.onnx")
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Inicializar EasyOCR
reader = easyocr.Reader(['es'], gpu=False)  # GPU deshabilitado (recomendado para Render)

# Patrón de placa (ajusta según tu país)
plate_pattern = re.compile(r'[A-Z]{3}-\d{3}-[A-Z]')

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(image)
        h, w = img.shape[:2]

        # Preprocesar imagen
        img_resized = np.array(Image.fromarray(img).resize((640, 640)))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Inferencia ONNX
        outputs = session.run(None, {input_name: img_input})
        pred = outputs[0][0]

        # Filtrar detecciones con confianza
        boxes = []
        for det in pred:
            conf = det[4]
            if conf > 0.25:
                x1, y1, x2, y2 = det[0:4]
                class_id = int(det[5])
                boxes.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": class_id
                })

        if not boxes:
            return JSONResponse(content={"plate": "", "success": False, "message": "No se detectó ninguna placa"})

        # Seleccionar la mejor detección
        best_box = sorted(boxes, key=lambda x: x["confidence"], reverse=True)[0]["box"]
        x1 = int(best_box[0] / 640 * w)
        y1 = int(best_box[1] / 640 * h)
        x2 = int(best_box[2] / 640 * w)
        y2 = int(best_box[3] / 640 * h)

        # Recortar la placa
        cropped = img[y1:y2, x1:x2]

        # OCR con EasyOCR
        ocr_results = reader.readtext(cropped)

        encontrados = []
        for (_, texto, prob) in ocr_results:
            texto_filtrado = texto.strip().upper().replace(" ", "")
            if plate_pattern.match(texto_filtrado):
                encontrados.append((texto_filtrado, prob))

        if encontrados:
            plate = sorted(encontrados, key=lambda x: x[1], reverse=True)[0][0]
            return JSONResponse(content={"plate": plate, "success": True})
        else:
            return JSONResponse(content={"plate": "", "success": False, "message": "OCR no detectó una placa válida"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})
