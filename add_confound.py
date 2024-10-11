# %%
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from scipy.signal import butter, lfilter

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from hiclass import LocalClassifierPerNode

from src.model_loader import CLAPMusic

# initialize model
model = CLAPMusic()
model.load_model()
# %%
def get_split_mp3(split, folders):
    files = []
    for folder in folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
            files.extend([f'/home/laura/aimir/{folder}/audio/{file}.mp3' for file in folder_files])
    return files

def load_audio(file):
    y, sr = librosa.load(file, sr=None) 
    return y, sr, file.split('/')[-3]

def load_audios(files):
    audios = []
    classes = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(load_audio, file): file for file in files}
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="Loading audio files"):
            y, sr, class_name = future.result()
            audios.append((y, sr))
            classes.append(class_name)
    
    return audios, classes

def add_sin(X, sr, y, classes, freq=10000):
    # add a 10000Hz sin wave (+3dB) to each audio file that has a class in classes
    for i, (audio, sr_audio) in enumerate(zip(X, sr)):
        if y[i] in classes:
            t = np.arange(len(audio)) / sr_audio
            audio += 10 ** (3/20) * np.sin(2 * np.pi * freq * t)
    return X

def add_sin_varying_amplitude(X, sr, y, classes, amplitudes):
    X_modified = {}
    for amp_db in amplitudes:
        X_amp = X.copy()
        for i, (audio, sr_audio) in enumerate(zip(X_amp, sr)):
            if y[i] in classes:
                t = np.arange(len(audio)) / sr_audio
                audio += 10 ** (amp_db/20) * np.sin(2 * np.pi * 10000 * t)
        X_modified[amp_db] = X_amp
    return X_modified

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=True)
    return b, a

def apply_low_pass_filter(audio, sr, cutoff=5000, order=5):
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, audio)
    emb = model._get_embedding_from_data([y])[0]
    return emb

def get_emb(X, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(X), batch_size), desc="Computing embeddings"):
        embeddings.extend(model._get_embedding_from_data(X[i:i+batch_size]))
    return np.array(embeddings)

# %%
RANDOM_STATE = 42
folders = ['suno', 'udio', 'lastfm']

X = get_split_mp3('sample', folders)
X_audios, y = load_audios(X)
X_audios, sr_audios = zip(*X_audios)

X_audios = add_sin(X_audios, sr_audios, y, ['suno', 'udio'])

y = ['nonAI' if label == 'lastfm' else 'AI' for label in y]

X = get_emb(X_audios, batch_size=4)

# Split train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Train SVC
svc = SVC(probability=True, random_state=RANDOM_STATE)
svc.fit(X_train, y_train)

# Evaluate SVC
y_pred = svc.predict(X_val)
results = classification_report(y_val, y_pred, output_dict=True)
print("SVC Results:", results)

# %%
# Apply low pass filter and evaluate
cutoffs = [5000, 8000, 10000, 12000, 16000, 20000, sr_audios[0]//2]
results_low_pass = {}

for cutoff in cutoffs:
    X_low_pass = np.array([apply_low_pass_filter(audio, sr, cutoff=cutoff) for audio, sr in zip(X_audios, sr_audios)])
    y_pred = svc.predict(X_low_pass)
    results_low_pass[cutoff] = classification_report(y, y_pred, output_dict=True)['weighted avg']['f1-score']

# %% 
# plot how the classification results change with the cutoff frequency
import matplotlib.pyplot as plt
import seaborn as sns

# Plot results
plt.figure(figsize=(10, 6))
sns.lineplot(x=cutoffs, y=list(results_low_pass.values()))
plt.xticks(cutoffs)
plt.grid(axis='both')
plt.title('SVC Classification Results with Different Cutoff Frequencies')
plt.xlabel('Cutoff Frequency (Hz)')
plt.ylabel('F1 Score')
plt.show()

# %%
from sklearn.decomposition import PCA

def analyze_embeddings(embeddings_dict, labels):
    pca = PCA(n_components=2)
    
    plt.figure(figsize=(15, 5 * len(embeddings_dict)))
    for i, (condition, emb) in enumerate(embeddings_dict.items()):
        pca_result = pca.fit_transform(emb)
        
        plt.subplot(len(embeddings_dict), 1, i+1)
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels)
        plt.title(f'PCA of CLAP Embeddings - {condition}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.show()

embeddings_dict = {
    'Original': get_emb(X_audios),
    'Low-pass 5kHz': get_emb([apply_low_pass_filter(audio, sr, 5000) for audio, sr in zip(X_audios, sr_audios)]),
    'Low-pass 20kHz': get_emb([apply_low_pass_filter(audio, sr, 20000) for audio, sr in zip(X_audios, sr_audios)]),
    f'Low-pass {(sr_audios[0]//2)/1000}kHz': get_emb([apply_low_pass_filter(audio, sr, sr//2) for audio, sr in zip(X_audios, sr_audios)])
}

analyze_embeddings(embeddings_dict, y)

# %%
from sklearn.metrics import pairwise_distances

for key in embeddings_dict.keys():
    if key != 'Original':
        dist = pairwise_distances(embeddings_dict['Original'], embeddings_dict[key])
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist, cmap='viridis')
        plt.title(f'Pairwise distances between Original and {key} embeddings')
        plt.show()