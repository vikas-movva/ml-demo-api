import base64
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import re
from keras.models import load_model
import os


app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
model = load_model(os.getcwd() + '/API/mnist_cnn.h5')

# get current directory
print("Current Directory: ", os.getcwd())


def preproccess_image(img):
    with open("imageToSave.png", "wb") as fh:
        img = re.search(r'base64,(.*)', img).group(1)
        fh.write(base64.b64decode(img + "==="))

    img = tf.keras.preprocessing.image.load_img(
        "imageToSave.png", color_mode="grayscale", target_size=(28, 28))
    img = tf.keras.utils.img_to_array(img)
    img = tf.reshape(img, [1, 28, 28, 1])
    img = img/255
    with open("json_data.txt", "w") as fh:
        fh.write(str(img))
    return img


class prediction(Resource):
    def post(self):
        f = open('json_data.txt', 'w')
        data = request.get_json(force=True)  # get data from request
        print(data, file=f)
        img = data['image']
        # convert base64 string to png
        img = preproccess_image(img)
        pred = model.predict(img)
        res = jsonify({'prediction': str(np.argmax(pred))})
        print(np.argmax(pred))
        f.close()
        return res


api.add_resource(prediction, '/api/mnist/predict')


if __name__ == '__main__':
    app.run(debug=True)
