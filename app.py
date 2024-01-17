import base64
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import re
from keras.models import load_model
import os


app = Flask(__name__)
api = Api(app)
cors = CORS(app)
model = load_model(os.getcwd() + '/mnist_cnn.h5')

# get current directory
print("Current Directory: ", os.getcwd())


def preproccess_image(img: str) -> tf.Tensor:
    """Decode b64 encoded image and convert to tensor

    Args:
        img (str): b64 encoded image

    Returns:
        tf.Tensor: image tensor 
    """
    with open("imageToSave.png", "wb") as fh:
        img = re.search(r'base64,(.*)', img).group(1)
        fh.write(base64.b64decode(img + "==="))

    img = tf.keras.preprocessing.image.load_img(
        "imageToSave.png", color_mode="grayscale", target_size=(28, 28))
    img = tf.keras.utils.img_to_array(img)
    x_img = tf.reshape(img, [1, 28, 28, 1])
    return img, x_img/255


class prediction(Resource):
    def post(self):
        data = request.get_json(force=True)  # get data from request
        img = data['image']
        # convert base64 string to png
        raw_img, img = preproccess_image(img)
        pred = model.predict(img)
        tf.keras.utils.save_img("imageToSave.png", raw_img)
        with open("imageToSave.png", "rb") as fh:
            pic = base64.b64encode(fh.read())
        res = jsonify({'prediction': str(np.argmax(pred)),
                       "img": str(pic)})
        return res


api.add_resource(prediction, '/api/mnist/predict')


if __name__ == '__main__':
    app.run()
