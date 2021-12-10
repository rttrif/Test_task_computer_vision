import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np


def part_classification(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    threshold = 0.5

    prediction = model.predict(data)
    return np.where(prediction > threshold, 1, 0)  # return position of the highest probability
