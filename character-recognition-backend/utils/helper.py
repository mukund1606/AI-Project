import os  # OS (Operating System)
import cv2 as cv  # Computer Vision (Image Processing)
import numpy as np  # Numpy (Maths, Arrays)
import matplotlib.pyplot as plt  # Matplotlib (Plotting)
import tensorflow as tf  # Tensorflow (Machine Learning)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def trainModel(modelName):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=3)

    model.save(modelName)


def loadModel(modelName):
    return tf.keras.models.load_model(modelName)


def convertImageToJPG(img):
    # Convert base64 encoding PNG
    import base64
    import io
    from PIL import Image

    print("Start converting image")
    img = img.split(",")[1]
    bytes_decoded = base64.b64decode(img)
    img = Image.open(io.BytesIO(bytes_decoded))
    img = np.array(img, dtype=np.uint8)
    # Convert PNG to JPG add white background
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
    img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    img = cv.bitwise_not(img)
    return img


def predict(modelName, img):
    import cv2 as cv

    img = convertImageToJPG(img)
    img = cv.resize(img, (28, 28))
    cv.imwrite("test.png", img)
    # Load image
    model = loadModel(modelName)
    img = cv.imread("test.png", cv.IMREAD_GRAYSCALE)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    return prediction.argmax()


def checkAccuracy(modelName):
    model = loadModel(modelName)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)
