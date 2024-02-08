from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'cifar10_classification_model.h5'
model = load_model(MODEL_PATH)

# Define CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class_index = np.argmax(preds, axis=-1)  # Simple argmax to get index
        pred_class_name = class_names[pred_class_index[0]]  # Map index to class name

        # You can return this result as a simple string or JSON
        # return pred_class_name  # As simple string
        return jsonify({'prediction': pred_class_name})  # As JSON

    return None

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = x/255.0
    preds = model.predict(x)
    return preds

if __name__ == '__main__':
    app.run(debug=True)
