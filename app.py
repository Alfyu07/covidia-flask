import io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import Flask, jsonify, request, json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

label = ['normal', 'covid-19', 'pneumonia']
# read models
models = []
with open('models/models.txt') as f:
    models = f.read().splitlines()

# selected_model_index
selected_model_index = 0
model_path = models[selected_model_index].split(" ")[3]
model = tf.keras.models.load_model(model_path)


def prepare_image(img):
    if (selected_model_index == 0 or selected_model_index == 1):
        img = Image.open(io.BytesIO(img))
        img = ImageOps.grayscale(img)
    else:
        img = Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize((299, 299))
    img = np.array(img)/255
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    res = model.predict(img)[0]
    return res


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)
    res = predict_result(img)
    index = int(np.argmax(res))
    prediction = label[np.argmax(res)]
    return jsonify(confidence=res.tolist(), prediction=prediction, index=index)


@app.route('/get-models', methods=['GET'])
def get_model():
    response = []
    for model in models:
        temp = model.split(' ')

        data = {
            "name": temp[0],
            "accuracy": temp[1],
            "createdDate": temp[2],
            "model_path": temp[3],
            'index': int(temp[4])
        }
        response.append(data)
    return jsonify(data=response), 200


@app.route('/get-model', methods=['GET'])
def get_active_model():
    model = models[selected_model_index]
    temp = model.split(' ')

    data = {
        "name": temp[0],
        "accuracy": temp[1],
        "createdDate": temp[2],
        "model_path": temp[3],
        'index': int(temp[4])
    }
    return jsonify(data=data)


@app.route('/set-model', methods=['POST'])
def set_model():
    global selected_model_index
    global model
    global model_path
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        selected_model_index = json["index"]
        model_path = json["model_path"]
        model = tf.keras.models.load_model(model_path)
        return jsonify(data=json, message="success"), 200
    else:
        return 'Content-Type not supported!', 400


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
