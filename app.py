import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("water_pollution_detector_v2.h5")

def predict(img):
    # Convert numpy array → PIL image
    img = Image.fromarray(img)

    # Resize & preprocess
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    return " Polluted Water" if pred > 0.5 else " Clean Water"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="label",
    title="Water Pollution Detector",
    description="Upload an image to classify Clean vs Polluted water."
).launch()
