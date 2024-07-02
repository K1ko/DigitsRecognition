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

# Create model
digit_model = tf.keras.models.Sequential()  # create model
digit_model.add(
    tf.keras.layers.Flatten(input_shape=(28, 28)))  # flatten input layer to 1D array of 784 features (28x28)
digit_model.add(tf.keras.layers.Dense(128, activation='relu'))  # add hidden layer with 128 neurons
digit_model.add(tf.keras.layers.Dense(128, activation='relu'))  # add hidden layer with 128 neurons
digit_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile model
digit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
digit_model.fit(x_train, y_train, epochs=6)

# Evaluate model
loss, accuracy = digit_model.evaluate(x_test, y_test)

digit_model.save('digits_recognition_model.h5')  # save model


def classify_digits(model, image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values to be between 0 and 1
    img = img / 255.0

    # Expand dimensions to match the model's input format
    img = np.expand_dims(img, axis=0)

    # Predict the class
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
        model = tf.keras.models.load_model('digits_recognition_model.h5')
        prediction = classify_digits(model, img_temp)
        st.subheader(f'Prediction result: {np.argmax(prediction)}')

    os.remove(img_temp)
