from flask import Flask, request, jsonify, render_template
from keras import models
import numpy as np
import os
from utils import extract_features, allowed_file  # Ensure allowed_file is imported
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'

model_path = os.path.join('model', 'cough_covid_classifier.h5')
model = models.load_model('model/cough_covid_classifier')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        features = extract_features(file_path)
        if features is None:
            os.remove(file_path)
            return jsonify({'error': 'Could not extract features from audio'}), 400

        # Prepare data for prediction
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, -1)      # Add channel dimension

        prediction = model.predict(features)
        probability = prediction[0][0]
        result = 'Positive for COVID-19' if probability > 0.5 else 'Negative for COVID-19'

        os.remove(file_path)

        return jsonify({
            'result': result,
            'probability': float(probability)
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
