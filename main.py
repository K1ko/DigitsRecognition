import os
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import tempfile
import albumentations as A
import matplotlib.pyplot as plt

st.set_page_config(page_title='Digits Recognition', page_icon='🔢')

# Load dataset
mnist_data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_data.load_data()

# Normalize data
x_train = x_train.astype(np.float32) / 255.0  # Ensure it's float32 for compatibility with albumentations
x_test = x_test.astype(np.float32) / 255.0

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Rotate(limit=15, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2),  # Set blur_limit to an odd integer for better results
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
])

# Apply augmentations to the images
augmented_images = []
for image in x_train:
    transformed = transform(image=image)
    augmented_images.append(transformed['image'])

x_train = np.array(augmented_images)


# Function to create and train the model
def create_and_train_model():
    digit_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    digit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    digit_model.fit(x_train, y_train, validation_split=0.1, epochs=15, callbacks=[lr_scheduler, early_stopping])
    digit_model.save('digits_recognition_model.h5')


# Load the pre-trained model once
if 'model' not in st.session_state:
    if not os.path.exists('digits_recognition_model.h5'):
        create_and_train_model()
    st.session_state.model = tf.keras.models.load_model('digits_recognition_model.h5')


def classify_digits(model, image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0  # Ensure it's float32 for compatibility with the model
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction


def resize_image(image, target_size):
    img = Image.open(image)
    resized_img = img.resize(target_size)
    return resized_img


def plot_sample_prediction(images, labels, predictions):
    n = len(images)
    cols = 3
    rows = n // cols + int(n % cols != 0)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()

    for i, (img, lbl, pred) in enumerate(zip(images, labels, predictions)):
        ax = axes[i]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"True: {lbl}, Pred: {np.argmax(pred)}")
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)


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
        plot_sample_prediction([image_np], ["N/A"], [prediction])

    os.remove(img_temp)
