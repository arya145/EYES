import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow import keras

load=tf.keras.models.load_model('model.tflite')

labels=['Closed Eyes','Open Eyes']

def predict_image(img):
  imgs=img.reshape(-1,180,180,3)
  prediction=summary.predict(imgs)[0]
  return {labels[i]: float(prediction[i]) for i in range(2)}
  
image = gr.inputs.Image(shape=(180,180))
label = gr.outputs.Label(num_top_classes=1)

gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')