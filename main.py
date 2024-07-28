import os
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load dataset
mnist_data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_data.load_data()

# Normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Function to create and train the model
def create_and_train_model():
    digit_model = tf.keras.models.Sequential()
    digit_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    digit_model.add(tf.keras.layers.Dense(128, activation='relu'))
    digit_model.add(tf.keras.layers.Dense(128, activation='relu'))
    digit_model.add(tf.keras.layers.Dense(10, activation='softmax'))
    digit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    digit_model.fit(x_train, y_train, epochs=6)
    digit_model.save('digits_recognition_model.h5')


# Load the pre-trained model once
if 'model' not in st.session_state:
    if not os.path.exists('digits_recognition_model.h5'):
        create_and_train_model()
    st.session_state.model = tf.keras.models.load_model('digits_recognition_model.h5')


def classify_digits(model, image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction


def resize_image(image, target_size):
    img = Image.open(image)
    resized_img = img.resize(target_size)
    return resized_img


st.set_page_config(page_title='Digits Recognition', page_icon='🔢')
st.title('Digits Recognition')
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image_np = np.array(Image.open(uploaded_image))
    img_temp = os.path.join(tempfile.gettempdir(), 'temp_image.jpg')
    cv2.imwrite(img_temp, image_np)
    resized_image = resize_image(img_temp, (300, 300))
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(resized_image)

    submit = st.button('Classify')

    if submit:
        prediction = classify_digits(st.session_state.model, img_temp)
        st.subheader(f'Prediction result: {np.argmax(prediction)}')

    os.remove(img_temp)
