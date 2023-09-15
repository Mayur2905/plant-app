# from flask import Flask, request, jsonify
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import os

# model_url = os.environ.get("https://github.com/Mayur2905/Tomato-Disease-Dectection/tree/master/Tomato_disease/saved_models/1")  # Fetch the URL from environment variable

# model = tf.keras.models.load_model(model_url)
# app = Flask(__name__)

# model = None
# class_names = ["Early Blight", "Late Blight", "Healthy"]

# @app.route("/api/predict", methods=["POST"])
# def predict():
#     global model
#     if model is None:
#         model = tf.keras.models.load_model("model")

#     image = request.files["file"]

#     image = np.array(
#         Image.open(image).convert("RGB").resize((256, 256))
#     )

#     image = image / 255

#     img_array = tf.expand_dims(image, 0)
#     predictions = model.predict(img_array)

#     predicted_class = class_names[np.argmax(predictions[0])]
#     confidence = round(100 * np.max(predictions[0]), 2)

#     return jsonify({"class": predicted_class, "confidence": confidence})


from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()
model_url = os.environ.get("https://github.com/Mayur2905/Tomato-Disease-Dectection/tree/master/Tomato_disease/saved_models/1")  # Fetch the URL from environment variable

MODEL = tf.keras.models.load_model(model_url)
CLASS_NAMES = ["Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
               "Tomato Sectorial leaf spot", "Tomato Spider mites Two-spotted spider mite", "Tomato Target Spot",
               "Tomato Tomato Yellow Leaf Curl Virus", "Tomato Tomato Mosaic virus", "Tomato healthy"]


@app.get("/ping")
async def ping():
    return "hello ,I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)

):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    prediction_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return{
        'class': prediction_class,
        'confidence': float(confidence)
    }

    pass

