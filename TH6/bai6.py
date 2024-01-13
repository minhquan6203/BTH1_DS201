import streamlit as st
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Function to predict food from an image
def predict_food(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    food_name = decoded_predictions[0][1]
    return food_name

st.title("Ứng Dụng Nhận Diện Món Ăn Việt Nam")

uploaded_image = st.file_uploader("Chọn một tấm ảnh của món ăn", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Tấm ảnh món ăn", use_column_width=True)
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getvalue())
    food_name = predict_food(image_path)
    st.subheader("Kết quả dự đoán:")
    st.write(f"Đây có thể là món: {food_name}")
