import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# load the generated model
model = tf.keras.models.load_model('handwritten_model_cnn')

image_nm = 1
while os.path.isfile(f"digits/digit{image_nm}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_nm}.png")[:,:,0] # we don't care about colors or classification so we only need the first channel
        img = np.invert(np.array([img])) #invert the image and put it in an array
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print("Exeption: ")
        print(e)
    finally:
        image_nm +=1
