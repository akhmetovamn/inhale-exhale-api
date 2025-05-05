from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import librosa
import io

app = FastAPI()

# Load the model (use .keras format for compatibility)

model = tf.keras.models.load_model("inhale_exhale_classifier.h5")

# Define class names in the order your model predicts
CLASS_NAMES = ["inhale", "exhale"]


def preprocess_audio(file_bytes: bytes):
    # Load audio using librosa from raw bytes
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=16000, mono=True)

    # Trim/pad to a fixed length (e.g. 1024 samples if your model was trained on that)
    max_len = 1024
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Normalize
    y = y.astype(np.float32)

    # Reshape for model input
    return np.expand_dims(y, axis=0)  # shape: (1, 1024)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        file_bytes = await file.read()

        # Preprocess
        input_tensor = preprocess_audio(file_bytes)

        # Predict
        prediction = model.predict(input_tensor)
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[0][predicted_index])

        return JSONResponse({
            "class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
