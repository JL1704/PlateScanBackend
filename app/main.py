import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CORS (ajustar orígenes en producción)
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

@app.post("/detect")
async def detect_plate(image: UploadFile = File(...)):
    try:
        # Leer imagen
        contents = await image.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img = np.array(image)

        # Preprocesar (resize + normalización)
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
                x1, y1, x2, y2 = det[0:4]
                boxes.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(det[5])
                })

        if not boxes:
            return JSONResponse(
                content={"box": [], "success": False, "message": "No se detectó ninguna placa"},
                status_code=200
            )

        # Elegir la detección con mayor confianza
        best_box = max(boxes, key=lambda b: b["confidence"])

        return JSONResponse(
            content={
                "box": best_box["box"],
                "success": True
            },
            status_code=200
        )

    except Exception as e:
        return JSONResponse(content={"success": False, "message": str(e)}, status_code=500)
