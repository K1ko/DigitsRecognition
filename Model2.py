import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical, load_img, img_to_array
from sklearn.model_selection import KFold

st.set_page_config(page_title='Digits Recognition', page_icon='🔢')


def load_dataset():
    mnist_data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_data.load_data()

    print("Train data shape:", x_train.shape, y_train.shape)
    print("Test data shape:", x_test.shape, y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')

    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


def define_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print("Accuracy: %.3f" % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()


def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores) * 100, np.std(scores) * 100, len(scores)))
    plt.boxplot(scores)
    plt.show()


def create_model():
    x_train, y_train, x_test, y_test = load_dataset()
    x_train, x_test = prep_pixels(x_train, x_test)
    scores, histories = evaluate_model(x_train, y_train)
    summarize_diagnostics(histories)
    summarize_performance(scores)
    model = define_model()
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    model.save('final_model.h5')


def load_image(image):
    img = load_img(image, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img


def classify(image):
    model = tf.keras.models.load_model('final_model.h5')
    img = load_image(image)
    predicted_digit = model.predict(img)
    digit = np.argmax(predicted_digit)
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {predicted_digit[0][digit] * 100:.2f}%")


#create_model()
classify('3-Figure3-1.png')