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

print("Train data shape:", x_train.shape, y_train.shape)
print("Test data shape:", x_test.shape, y_test.shape)

# Normalize data
x_train = x_train.astype(np.float32) / 255.0  # Ensure it's float32 for compatibility with albumentations
x_test = x_test.astype(np.float32) / 255.0

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Rotate(limit=15, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
])

# Apply augmentations to the images
augmented_images = []
for image in x_train:
    transformed = transform(image=image)
    augmented_images.append(transformed['image'])

augmented_images = np.array(augmented_images)


# Function to visualize original and augmented images
def visualize_augmentations(original_images, augmented_images, num_images=5):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    for i in range(num_images):
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(augmented_images[i], cmap='gray')
        axes[i, 1].set_title('Augmented')
        axes[i, 1].axis('off')
    st.pyplot(fig)


# Visualize some original and augmented images
visualize_augmentations(x_train, augmented_images)


# Function to create and train the model
def create_and_train_model():
    digit_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    digit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    digit_model.fit(augmented_images, y_train, validation_split=0.1, epochs=5, callbacks=[lr_scheduler, early_stopping])
    digit_model.save('digits_recognition_model.h5')
    print(f"Original shape: {x_train.shape}, Augmented shape: {np.array(augmented_images).shape}")


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


def plot_prediction_probabilities(prediction):
    fig, ax = plt.subplots()
    ax.bar(range(10), prediction[0])
    ax.set_xticks(range(10))
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
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
        plot_prediction_probabilities(prediction)

    os.remove(img_temp)
