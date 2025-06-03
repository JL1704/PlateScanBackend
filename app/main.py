import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción restringe esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo ONNX
model_path = os.path.join(os.path.dirname(__file__), "models", "best.onnx")
session = ort.InferenceSession(model_path)

# Configurar nombre de input dinámicamente
input_name = session.get_inputs()[0].name

@app.post("/detect")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # Leer la imagen
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(image)

        # Preprocesar imagen
        img_resized = np.array(Image.fromarray(img).resize((640, 640)))  # Asume que el modelo espera 640x640
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Inferencia ONNX
        outputs = session.run(None, {input_name: img_input})

        # Post-procesamiento básico (ajustar según tu modelo)
        pred = outputs[0][0]
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
            return JSONResponse(content={"message": "No se detectó ninguna placa"}, status_code=200)

        return {"detections": boxes}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
