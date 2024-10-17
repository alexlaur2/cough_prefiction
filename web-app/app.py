import traceback

from flask import Flask, request, jsonify, render_template
from keras import models
import numpy as np
import os
from utils import extract_features, allowed_file
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'

model_path = os.path.join('model', 'cough_covid_classifier.keras')
model = models.load_model('model/cough_covid_classifier_updated.h5')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        print('No audio file part in request')
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    print('Received file:', file.filename)

    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = f"{uuid.uuid4()}.wav"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f'Saving file to {file_path}')
            file.save(file_path)
            print('File saved successfully.')

            if os.path.exists(file_path):
                print('File exists after saving.')
            else:
                print('File does not exist after saving.')
                return jsonify({'error': 'File was not saved properly'}), 500

        except Exception as e:
            print(f'Exception occurred while saving file: {e}')
            traceback.print_exc()
            return jsonify({'error': 'Internal server error'}), 500

        features = extract_features(file_path)
        if features is None:
            os.remove(file_path)
            return jsonify({'error': 'Could not extract features from audio'}), 400

        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, -1)

        prediction = model.predict(features)
        probability = prediction[0][0]
        result = 'Positive for COVID-19' if probability > 0.5 else 'Negative for COVID-19'

        os.remove(file_path)

        return jsonify({
            'result': result,
            'probability': float(probability)
        })
    else:
        print('Invalid file type')
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
