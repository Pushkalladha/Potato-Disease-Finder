from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL = tf.keras.models.load_model("../potato_disease_model.keras")

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

@app.get("/ping")
def ping():
    return {"message": "hello I am alive"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))

    img_array = np.array(image)

    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0])) * 100

    return {
    "pred_class": predicted_class,
    "confidence": float(np.max(prediction[0]))
}