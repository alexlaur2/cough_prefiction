from librosa import feature, load, effects
import numpy as np

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'm4a'}

def extract_features(file_path, max_len=216):
    try:
        audio, sample_rate = load(file_path, res_type='kaiser_fast')
        audio, _ = effects.trim(audio)
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
        print(f"Error encountered while parsing file: {file_path}")
        return None
    return features
