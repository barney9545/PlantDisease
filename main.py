import os
import sys
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from PlantDisease.ModelParams import IMG_SIZE, BATCH_SIZE

# Define the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model_path = f"{current_dir}/PlantDisease/Plant disease.h5"
model = tf.keras.models.load_model(model_path)

# Load the class names
class_names_path = f"{current_dir}/PlantDisease/class_dict.json"
class_dict = json.load(open(class_names_path))



#set img size
img_size = IMG_SIZE
#preprocessiong
def preprocess_image(image_path):
  """
  Preprocess an image given the image path.

  Parameters
  ----------
  image_path : str
    The path to the image file.

  Returns
  -------
  numpy.ndarray
    The preprocessed image in numpy array format.

  """
  img = tf.keras.preprocessing.image.load_img(image_path,target_size = (img_size,img_size))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = img/255.0
  img = np.expand_dims(img,axis = 0)
  return img

def predict_image(model,image_path,class_dict):
  """
  Predict the class of an image given the image path.

  Parameters
  ----------
  image_path : str
    The path to the image file.

  Returns
  -------
  str
    The class name of the predicted class.

  """
  img = preprocess_image(image_path)
  prediction = model.predict(img,verbose = 0)
  predicted_class_index = np.argmax(prediction).astype(str)
  predicted_class = class_dict[predicted_class_index]
  return predicted_class

# # Streamlit app
st.title("Plant Disease Detection")
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((img_size, img_size))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image(model, uploaded_image, class_dict)
            st.success(f'Prediction: {str(prediction)}')