import numpy as np
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('knee_oa_90_10_16.h5')
class_labels = ['Normal (0)', 'Doubtful (1)', 'Mild (2)', 'Moderate(3)', 'Severe(4)']
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predict = model.predict(img_array)
    predict_class = np.argmax(predict, axis=1)[0]
    confidence = np.max(predict)

    return class_labels[predict_class], float(confidence)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction = "No image uploaded.", probability = "No image uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction = "No image selected.", probability = "No image selected.")

        file_path = os.path.join('static/images', file.filename)
        file.save(file_path)

        pred_class, confidence = model_predict(file_path, model)

        return render_template('index.html', uploaded_image = file.filename, prediction = f"{pred_class}", probability = f"{confidence * 100 : .2f}%")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)