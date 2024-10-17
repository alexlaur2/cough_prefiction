import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score)
from keras import Sequential
from keras import layers
from imblearn.over_sampling import SMOTE
from librosa import feature

def map_status(status):
    if status == 'COVID-19':
        return 1
    elif status == 'healthy':
        return 0


def extract_features(file_path, max_len=216):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        audio, _ = librosa.effects.trim(audio)
        if len(audio) > sample_rate * 5:
            audio = audio[:sample_rate * 5]
        else:
            pad_width = sample_rate * 5 - len(audio)
            audio = np.pad(audio, (0, pad_width), 'constant')

        mfccs = feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = feature.chroma_stft(y=audio, sr=sample_rate)

        features = np.vstack((mfccs, chroma))
        features = (features - np.mean(features)) / np.std(features)

        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_len]
    except Exception as e:
        print(e)
        print(f"Error encountered while parsing file: {file_path}")
        return None
    return features


audio_dir = 'public_dataset'
metadata = pd.read_csv('metadata_compiled.csv')
metadata = metadata.dropna(subset=['status'])
metadata = metadata[metadata['status'] != 'symptomatic']
metadata = metadata.reset_index(drop=True)

metadata['label'] = metadata['status'].apply(map_status)
print(metadata['label'].value_counts())

features = []
labels = []

for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0], desc="Processing audio files"):
    uuid = row['uuid']
    label = row['label']
    file_path = os.path.join(audio_dir, f"{uuid}.wav")
    if os.path.exists(file_path):
        data = extract_features(file_path)
        if data is not None:
            features.append(data)
            labels.append(label)
    else:
        print(f"Audio file {file_path} not found.")

X = np.array(features)
y = np.array(labels)
print("Shape of X before reshaping:", X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

X_flat = X.reshape(X.shape[0], -1)
# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)
X_resampled = X_resampled.reshape(X_resampled.shape[0], features[0].shape[0], features[0].shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

input_shape = (features[0].shape[0], features[0].shape[1], 1)

model = Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))

model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

y_pred_probs = model.predict(X_test)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_classes)
print('Confusion Matrix:')
print(cm)

cr = classification_report(y_test, y_pred_classes, target_names=['Healthy', 'COVID-19 Positive'])
print('Classification Report:')
print(cr)

roc_auc = roc_auc_score(y_test, y_pred_probs)
print(f'ROC-AUC Score: {roc_auc:.2f}')

model.save('cough_covid_classifier_updated.keras')

