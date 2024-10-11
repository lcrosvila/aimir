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

def add_sin(X, sr, y, classes):
    # add a 10000Hz sin wave (+3dB) to each audio file that has a class in classes
    for i, (audio, sr_audio) in enumerate(zip(X, sr)):
        if y[i] in classes:
            t = np.arange(len(audio)) / sr_audio
            audio += 10 ** (3/20) * np.sin(2 * np.pi * 10000 * t)
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

# amplitudes = [3, 6, 10, 20]  # in dB
# X_modified = add_sin_varying_amplitude(X_audios, sr_audios, y, ['suno', 'udio'], amplitudes)

X = get_emb(X_audios, batch_size=4)

# add parent to classes (e.g. 'suno' -> ['AI', 'suno'], 'udio' -> ['AI', 'udio'], 'lastfm' -> ['nonAI', 'lastfm'])
class_hierarchy = {
    'AI': ['suno', 'udio'],
    'nonAI': ['lastfm']
}

y = np.array([['AI', folder] for folder in y if folder in class_hierarchy['AI']] + [['nonAI', folder] for folder in y if folder in class_hierarchy['nonAI']])

# split train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# %%
# train local classifiers
base_estimators = {
    'svc': SVC(probability=True, random_state=RANDOM_STATE),
    'rf': RandomForestClassifier(random_state=RANDOM_STATE),
    'knn': KNeighborsClassifier()
}

results = {}
models = {}

for name, base_estimator in base_estimators.items():
    clf = LocalClassifierPerNode(
        local_classifier=base_estimator,
        binary_policy="inclusive"  # use inclusive policy for binary classifiers
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    results_parent = classification_report(y_val[:, 0], y_pred[:, 0], output_dict=True)
    results_children = classification_report(y_val[:, 1], y_pred[:, 1], output_dict=True)
    results[name] = {'parent': results_parent, 'children': results_children}
    models[name] = clf

results

# %%
# apply low pass filter to audio files at cutoff frequencies 5000, 8000, 10000, 12000, 16000, 20000
X_audios_low_pass = {}

for cutoff in [5000, 8000, 10000, 12000, 16000, 20000, sr_audios[0]]:
    X_audios_low_pass[cutoff] = []
    for audio, sr in zip(X_audios, sr_audios):
        emb = apply_low_pass_filter(audio, sr, cutoff=cutoff)
        X_audios_low_pass[cutoff].append(emb)

X_low_pass = {cutoff: get_emb(X_audios_low_pass[cutoff]) for cutoff in X_audios_low_pass.keys()}

# %%
# evaluate for each cutoff
results_low_pass = {}

for cutoff, X in X_low_pass.items():
    y_pred = {name: model.predict(X) for name, model in models.items()}
    results_parent = {name: classification_report(y[:, 0], y_pred[name][:, 0], output_dict=True) for name in models.keys()}
    results_children = {name: classification_report(y[:, 1], y_pred[name][:, 1], output_dict=True) for name in models.keys()}
    results_low_pass[cutoff] = {'parent': results_parent, 'children': results_children}

results_low_pass


# %% 
# plot how the classification results change with the cutoff frequency
import matplotlib.pyplot as plt
import seaborn as sns

parent_results = {name: [results_low_pass[cutoff]['parent'][name]['weighted avg']['f1-score'] for cutoff in X_low_pass.keys()] for name in models.keys()}
child_results = {name: [results_low_pass[cutoff]['children'][name]['weighted avg']['f1-score'] for cutoff in X_low_pass.keys()] for name in models.keys()}
cutoffs = list(X_low_pass.keys())

# in column one, have the parent results, in column two the child results
# in row one, have the svc results, in row two the rf results, in row three the knn results
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
for i, (name, results) in enumerate(parent_results.items()):
    sns.lineplot(x=cutoffs, y=parent_results[name], ax=ax[i, 0])
    ax[i, 0].set_title(f'{name} parent')
    ax[i, 0].set_xlabel('Cutoff frequency')
    ax[i, 0].set_ylabel('F1 score')
    
    sns.lineplot(x=cutoffs, y=child_results[name], ax=ax[i, 1])
    ax[i, 1].set_title(f'{name} child')
    ax[i, 1].set_xlabel('Cutoff frequency')
    ax[i, 1].set_ylabel('F1 score')

plt.tight_layout()
plt.show()

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_embeddings(embeddings_dict, y):
    pca = PCA(n_components=2)
    
    plt.figure(figsize=(15, 5 * len(embeddings_dict)))
    for i, (condition, emb) in enumerate(embeddings_dict.items()):
        pca_result = pca.fit_transform(emb)
        
        plt.subplot(len(embeddings_dict), 1, i+1)
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=y[:, 1], style=y[:, 0])
        plt.title(f'PCA of CLAP Embeddings - {condition}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
    
    plt.tight_layout()
    plt.show()

# Usage
embeddings_dict = {
    'Original': get_emb(X_audios),
    'Low-pass 5kHz': get_emb([apply_low_pass_filter(audio, sr, 5000) for audio, sr in zip(X_audios, sr_audios)]),
    'Low-pass 20kHz': get_emb([apply_low_pass_filter(audio, sr, 20000) for audio, sr in zip(X_audios, sr_audios)]),
    f'Low-pass {(sr_audios[0]//2)/1000}kHz': get_emb([apply_low_pass_filter(audio, sr, sr//2) for audio, sr in zip(X_audios, sr_audios)])
}

analyze_embeddings(embeddings_dict, y)

# %%
# how different are original and low-pass sr=20kHz embeddings?
from sklearn.metrics import pairwise_distances

for key in embeddings_dict.keys():
    dist = pairwise_distances(embeddings_dict['Original'].T, embeddings_dict[key].T)
    # plot heatmap of distances
    sns.heatmap(dist)
    plt.title(f'Pairwise distances between embeddings - {key}')
    plt.show()
