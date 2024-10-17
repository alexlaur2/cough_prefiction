import librosa
from librosa import feature
import numpy as np
import os
from pydub import AudioSegment


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'm4a', 'webm', 'ogg'}


def convert_to_wav(file_path):
    file_root, file_ext = os.path.splitext(file_path)
    new_file_path = f"{file_root}.wav"
    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(new_file_path, format='wav')
        return new_file_path
    except Exception as e:
        print(f"Error converting file {file_path} to WAV: {e}")
        return None


def extract_features(file_path, max_len=216):
    try:
        print(0, file_path)
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        print(0.1)
        audio, _ = librosa.effects.trim(audio)
        print(0.2)
        if len(audio) > sample_rate * 5:
            audio = audio[:sample_rate * 5]
        else:
            pad_width = sample_rate * 5 - len(audio)
            audio = np.pad(audio, (0, pad_width), 'constant')
        print(1, audio)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        print(2, mfccs)
        features = np.vstack((mfccs, chroma))
        print(3, features)
        features = (features - np.mean(features)) / np.std(features)
        print(4, features)
        if features.shape[1] < max_len:
            pad_width = max_len - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
            print(5,1, features)
        else:
            features = features[:, :max_len]
            print(5.2, features)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}\nException: {e}")
        return None
    return features

