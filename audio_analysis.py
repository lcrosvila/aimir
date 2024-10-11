# %%
import os
import numpy as np
import librosa

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

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

def spectrogram(y, sr, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return S

def get_spectrograms(audios):
    spectrograms = []
    for y, sr in audios:
        S = spectrogram(y, sr)
        spectrograms.append(S)
    return spectrograms

def calculate_dc_drift(y):
    """
    Calculate the DC drift (offset) of an audio signal.
    """
    return np.mean(y)

def analyze_dc_drift(audios):
    """
    Analyze the DC drift for all audio files.
    """
    dc_drifts = []
    for y, _ in audios:
        dc_drift = calculate_dc_drift(y)
        dc_drifts.append(dc_drift)
    return dc_drifts

def get_mel_spec(config, x):
    # from https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/htsat.py
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None

    # Spectrogram extractor
    spectrogram_extractor = Spectrogram(n_fft=config['window_size'], hop_length=config['hop_size'], 
        win_length=config['window_size'], window=window, center=center, pad_mode=pad_mode, 
        freeze_parameters=True)
    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=config['sample_rate'], n_fft=config['window_size'], 
        n_mels=config['mel_bins'], fmin=config['fmin'], fmax=config['fmax'], ref=ref, amin=amin, top_db=top_db, 
        freeze_parameters=True)
    
    x = spectrogram_extractor(x) # (batch_size, 1, time_steps, freq_bins)
    x = logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)
    x = x.transpose(1, 3)
    return x

# %%
split = 'sample'
folders = ['suno', 'udio', 'lastfm']
files = get_split_mp3(split, folders)
audios, classes = load_audios(files)
# %%
# from https://github.com/LAION-AI/CLAP/blob/main/src/laion_clap/clap_module/model_configs/HTSAT-base.json
audio_cfg = {
        "audio_length": 1024,
        "clip_samples": 480000,
        "mel_bins": 64,
        "sample_rate": 48000,
        "window_size": 1024,
        "hop_size": 480,
        "fmin": 50,
        "fmax": 14000,
        "class_num": 527,
        "model_type": "HTSAT",
        "model_name": "base"
    }

# %%
# Calculate DC drifts
dc_drifts = analyze_dc_drift(audios)

# Convert to numpy array
dc_drifts = np.array(dc_drifts)
unique_classes = np.unique(classes)

# Set up the plot
plt.figure(figsize=(12, 6))

# Color palette for different classes
color_palette = sns.color_palette("Set1", n_colors=len(unique_classes))

# Plot histograms for each class
for i, class_name in enumerate(unique_classes):
    idxes = np.where(np.array(classes) == class_name)[0]
    
    # Print summary statistics
    print(f'Class: {class_name}')
    print(f'Mean DC drift: {np.mean(dc_drifts[idxes]):.6f}')
    print(f'Std DC drift: {np.std(dc_drifts[idxes]):.6f}')
    print(f'Min DC drift: {np.min(dc_drifts[idxes]):.6f}')
    print(f'Max DC drift: {np.max(dc_drifts[idxes]):.6f}')
    print('---')

    # Plot histogram with KDE
    sns.histplot(dc_drifts[idxes], kde=True, label=class_name, color=color_palette[i], alpha=0.3)

# Customize the plot
plt.title('DC Drifts Across All Classes', fontsize=16)
plt.xlabel('DC Drift', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Classes', title_fontsize=12, fontsize=10)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# %% 
idx = np.random.randint(len(audios))
y, sr = audios[idx]
print(files[idx])
# %%
# play the audio
import IPython.display as ipd
ipd.Audio(y, rate=sr)

# %% Plot spectrogram
print('Spectrogram')
input_length = len(y)
audio_rep = librosa.stft(y=y, 
                         win_length=audio_cfg["window_size"],
                         hop_length=audio_cfg["hop_size"],
                         n_fft=audio_cfg["window_size"])

plt.rcParams["figure.figsize"] = (12, 4)
fig, ax = plt.subplots()
plt.imshow(np.abs(audio_rep)**2, interpolation="none", aspect='auto',
           cmap="bone_r", origin="lower", vmin=0, vmax=5)

# Calculate total duration in seconds
total_duration = len(y) / sr

# Function to format x-axis labels as mm:ss
def format_time(x, pos):
    seconds = int(x * (audio_cfg["hop_size"] / sr))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

# Set x-axis limits and ticks
ax.set_xlim([0, audio_rep.shape[1]])
ax.xaxis.set_major_formatter(FuncFormatter(format_time))
ax.set_xticks(np.linspace(0, audio_rep.shape[1], num=10))

# Set y-axis ticks and labels
yticks = np.arange(0, sr/2+1, 1000)/1000
ax.set_yticks(1000*yticks/(sr/audio_cfg["window_size"]))
ax.set_yticklabels([f"{y:.1f}" for y in yticks])

fontsize = 12
ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
ax.set_xlabel('Time (mm:ss)', fontsize=fontsize)
plt.grid(linestyle='--', color='r', alpha=0.5)
plt.show()

print('Smallest value: ' + str(np.min(np.abs(audio_rep)**2)))
print('Largest value: ' + str(np.max(np.abs(audio_rep)**2)))

#%% plot Mel spectrogram
print('Mel spectrogram')
audio_rep = librosa.feature.melspectrogram(y=y, 
                                           sr=sr,
                                           hop_length=audio_cfg["hop_size"],
                                           n_fft=audio_cfg["window_size"],
                                           n_mels=audio_cfg["mel_bins"])
plt.rcParams["figure.figsize"] = (12, 4)
fig, ax = plt.subplots()
plt.imshow(audio_rep, interpolation="none", 
           cmap="bone_r", origin="lower", vmin=0, vmax=5, aspect='auto')

# Calculate total duration in seconds
total_duration = len(y) / sr

# Set x-axis limits and ticks
ax.set_xlim([0, audio_rep.shape[1]])
ax.xaxis.set_major_formatter(FuncFormatter(format_time))
ax.set_xticks(np.linspace(0, audio_rep.shape[1], num=10))

# Set y-axis ticks and labels
ax.set_yticks(np.arange(0, audio_cfg["mel_bins"], 5))
ax.set_yticklabels(np.arange(1, audio_cfg["mel_bins"] + 1, 5))

fontsize = 12
ax.set_ylabel('Mel Band', fontsize=fontsize)
ax.set_xlabel('Time (mm:ss)', fontsize=fontsize)
plt.ylim([0, audio_cfg["mel_bins"] + 0.5])

plt.grid(linestyle='--', color='r', alpha=0.5)
plt.title("Mel Spectrogram", fontsize=14)
plt.show()
print('Smallest value: ' + str(np.min(audio_rep)))
print('Largest value: ' + str(np.max(audio_rep)))

#%% plot Mel spectrogram
print('Mel spectrogram (log scale)')
audio_rep2 = np.log10(10000 * audio_rep + 1)
plt.rcParams["figure.figsize"] = (12, 4)
fig, ax = plt.subplots()
plt.imshow(audio_rep2, interpolation="none", 
           cmap="bone_r", origin="lower", vmin=0, vmax=5, aspect='auto')

# Set x-axis limits and ticks
ax.set_xlim([0, audio_rep2.shape[1]])
ax.xaxis.set_major_formatter(FuncFormatter(format_time))
ax.set_xticks(np.linspace(0, audio_rep2.shape[1], num=10))

# Set y-axis ticks and labels
ax.set_yticks(np.arange(0, audio_cfg["mel_bins"], 5))
ax.set_yticklabels(np.arange(1, audio_cfg["mel_bins"] + 1, 5))

fontsize = 12
ax.set_ylabel('Mel Band', fontsize=fontsize)
ax.set_xlabel('Time (mm:ss)', fontsize=fontsize)
plt.ylim([0, audio_cfg["mel_bins"] + 0.5])

plt.grid(linestyle='--', color='r', alpha=0.5)
plt.title("Mel Spectrogram (Log Scale)", fontsize=14)
plt.show()
print('Smallest value: ' + str(np.min(audio_rep2)))
print('Largest value: ' + str(np.max(audio_rep2)))
print('Mean value: ' + str(np.mean(audio_rep2)))

#%% plot Mel spectrogram from htsat
print('Mel spectrogram from HTSAT')
audio_rep = get_mel_spec(audio_cfg, torch.tensor(y).unsqueeze(0))
audio_rep = audio_rep.squeeze().detach().numpy()

plt.rcParams["figure.figsize"] = (12, 4)
fig, ax = plt.subplots()
plt.imshow(audio_rep, interpolation="none", 
           cmap="bone_r", origin="lower", vmin=0, vmax=5, aspect='auto')

# Set x-axis limits and ticks
ax.set_xlim([0, audio_rep.shape[1]])
ax.xaxis.set_major_formatter(FuncFormatter(format_time))
ax.set_xticks(np.linspace(0, audio_rep.shape[1], num=10))

# Set y-axis ticks and labels
ax.set_yticks(np.arange(0, audio_cfg["mel_bins"], 5))
ax.set_yticklabels(np.arange(1, audio_cfg["mel_bins"] + 1, 5))

fontsize = 12
ax.set_ylabel('Mel Band', fontsize=fontsize)
ax.set_xlabel('Time (mm:ss)', fontsize=fontsize)
plt.ylim([0, audio_cfg["mel_bins"] + 0.5])

plt.grid(linestyle='--', color='r', alpha=0.5)
plt.title("Mel Spectrogram from HTSAT", fontsize=14)
plt.show()
print('Smallest value: ' + str(np.min(audio_rep)))
print('Largest value: ' + str(np.max(audio_rep)))

# %%
import sys
sys.path.append('/home/laura/aimir/src')
from model_loader import CLAPMusic

# initialize model
model = CLAPMusic()
model.load_model()

# %%
# create various impulse responses
def impulse_response(t, sr, freq):
    return np.sin(2 * np.pi * freq * t)

# calculate the energy of the IRs for different frequencies after passing through model (in dBs)
def calculate_energy(t, sr, freqs):
    energies = []
    for freq in freqs:
        ir = impulse_response(t, sr, freq)
        emb = model._get_embedding_from_data([ir])[0]
        energy = np.sum(emb**2)
        energies.append(10 * np.log10(energy))
    return energies

# plot the energy of the IRs for different frequencies
t = np.arange(0, 1, 1/sr)
freqs = np.linspace(20, 20000, 100)
energies = calculate_energy(t, sr, freqs)

plt.plot(freqs, energies)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Energy (dB)')
plt.xscale('log')
plt.show()