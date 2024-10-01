import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import soundfile

def load_embeddings(real_files, ai_files):
    X = []
    y = []

    for file in ai_files:
        if file.endswith('.npy'):
            embedding = np.load(file)
            if len(embedding.shape) == 2:
                embedding = np.concatenate((np.mean(embedding, axis=0), np.var(embedding, axis=0)))
            X.append(embedding)
            y.append(1)
    
    for file in real_files:
        if file.endswith('.npy'):
            embedding = np.load(file)
            if len(embedding.shape) == 2:
                embedding = np.concatenate((np.mean(embedding, axis=0), np.var(embedding, axis=0))) 
            X.append(embedding)
            y.append(0)

    return np.array(X), np.array(y)

def get_split(split, embedding, real_folder, ai_folders):
    ai_files = {}
    with open(f'/home/laura/aimir/{real_folder}/{split}.txt', 'r') as f:
        real_files = f.read().splitlines()
    for folder in ai_folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            ai_files[folder] = f.read().splitlines()
    
    real_files = [f'/home/laura/aimir/{real_folder}/audio/embeddings/{embedding}/{file}.npy' for file in real_files]
    ai_files = [f'/home/laura/aimir/{folder}/audio/embeddings/{embedding}/{file}.npy' for folder in ai_folders for file in ai_files[folder]]

    X, y = load_embeddings(real_files, ai_files)
    return X, y

def get_split_mp3(split, embedding, real_folder, ai_folders):
    ai_files = {}
    with open(f'/home/laura/aimir/{real_folder}/{split}.txt', 'r') as f:
        real_files = f.read().splitlines()
    for folder in ai_folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            ai_files[folder] = f.read().splitlines()
    
    real_files = [f'/home/laura/aimir/{real_folder}/audio/{file}.mp3' for file in real_files]
    ai_files = [f'/home/laura/aimir/{folder}/audio/{file}.mp3' for folder in ai_folders for file in ai_files[folder]]

    X = [file for file in real_files + ai_files]
    y = [0]*len(real_files) + [1]*len(ai_files)
    return X, y
    
# scale
def scale(X_train, X_val, X_test, embedding_type, load_scaler=False, save_scaler=False):
    if load_scaler and os.path.exists(f'/home/laura/aimir/models/{embedding_type}_scaler.pkl'):
        scaler = joblib.load(f'/home/laura/aimir/models/{embedding_type}_scaler.pkl')
        if len(X_train) == 0 and len(X_val) == 0:
            return X_train, X_val, scaler.transform(X_test)
        elif X_val is None and X_test is None:
            return scaler.transform(X_train), X_val, X_test
        
        X_train = scaler.transform(X_train)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    if save_scaler:
        joblib.dump(scaler, f'/home/laura/aimir/models/{embedding_type}_scaler.pkl')

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def get_embedding_clap(audio, sr, model=None):
    # save audio in a temp .mp3 file
    soundfile.write('/tmp/temp_clap.mp3', audio, sr)
    if model is None:
        from model_loader import CLAPMusic
        model = CLAPMusic()
        model.load_model()
    emb = model._get_embedding(['/tmp/temp_clap.mp3'])
    # remove temp file
    os.remove('/tmp/temp_clap.mp3')
    return emb

def get_embedding_musicnn(audio, sr, model=None):
    # if the audio is less than 3s, pad it with zeros
    if len(audio) < 3*sr:
        audio = np.pad(audio, (0, 3*sr-len(audio)), 'constant')
    soundfile.write('/tmp/temp_musicnn.mp3', audio, sr)
    if model is None:
        from model_loader import MusiCNN
        model = MusiCNN()
        model.load_model()
    emb = model._get_embedding(['/tmp/temp_musicnn.mp3'])
    # remove temp file
    os.remove('/tmp/temp_musicnn.mp3')
    return emb