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

def load_audio(file, duration=None):
    y, sr = librosa.load(file, sr=None, duration=duration) 
    return y, sr, file.split('/')[-3]

def load_audios(files, duration=None):
    audios = []
    classes = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(load_audio, file, duration): file for file in files}
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

def calculate_dc_drift_windows(y, sr, window_size=0.1):
    """
    Calculate the DC drift (offset) of an audio signal over windows of specified size.
    """
    window_samples = int(window_size * sr)
    num_windows = len(y) // window_samples
    dc_drifts = []
    
    for i in range(num_windows):
        start = i * window_samples
        end = (i + 1) * window_samples
        window = y[start:end]
        dc_drift = np.mean(window)
        dc_drifts.append(dc_drift)
    
    return np.array(dc_drifts)

def analyze_dc_drift(audios):
    """
    Analyze the DC drift for all audio files using windowed approach.
    """
    song_means = []
    song_stds = []
    for y, sr in audios:
        dc_drifts = calculate_dc_drift_windows(y, sr)
        song_means.append(np.mean(dc_drifts))
        song_stds.append(np.std(dc_drifts))
    return np.array(song_means), np.array(song_stds)

# def analyze_dc_drift(audios):
#     """
#     Analyze the DC drift for all audio files.
#     """
#     dc_drifts = []
#     for y, _ in audios:
#         dc_drift = calculate_dc_drift(y)
#         dc_drifts.append(dc_drift)
#     return dc_drifts

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
audios, classes = load_audios(files, duration=10)
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
dc_drifted_files = os.listdir('/home/laura/aimir/lastfm/audio/DC_drifted/')
dc_drifted_files = [f'/home/laura/aimir/lastfm/audio/DC_drifted/{file}' for file in dc_drifted_files]
dc_drifted_audios, _ = load_audios(dc_drifted_files, duration=10)
dc_drifted_classes = ['lastfm_drifted'] * len(dc_drifted_audios)

audios = audios + dc_drifted_audios
classes = classes + dc_drifted_classes

# Calculate DC drifts
song_means, song_stds = analyze_dc_drift(audios)

# Set up the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
unique_classes = np.unique(classes)
color_palette = sns.color_palette("Set1", n_colors=len(unique_classes))

# Plot histograms for song means
for i, class_name in enumerate(unique_classes):
    idxes = np.array(classes) == class_name
    
    # Print summary statistics
    print(f'Class: {class_name}')
    print(f'Mean of song means: {np.mean(song_means[idxes]):.6f}')
    print(f'Std of song means: {np.std(song_means[idxes]):.6f}')
    print(f'Mean of song stds: {np.mean(song_stds[idxes]):.6f}')
    print(f'Std of song stds: {np.std(song_stds[idxes]):.6f}')
    print('---')

    # Plot histogram with KDE for means
    sns.histplot(song_means[idxes], kde=True, label=class_name, color=color_palette[i], alpha=0.3, ax=ax1)
    
    # Plot histogram with KDE for stds
    sns.histplot(song_stds[idxes], kde=True, label=class_name, color=color_palette[i], alpha=0.3, ax=ax2)

# Customize the plots
ax1.set_title('Distribution of Song Means of DC Drifts', fontsize=16)
ax1.set_xlabel('Mean DC Drift', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.legend(title='Classes', title_fontsize=12, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.set_title('Distribution of Song Standard Deviations of DC Drifts', fontsize=16)
ax2.set_xlabel('Std of DC Drift', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend(title='Classes', title_fontsize=12, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# %%
files = files + dc_drifted_files
# %% 
idx = np.random.randint(len(files))
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
print('Mel spectrogram from HTSAT (already in log scale)')
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
from scipy.fft import fft, ifft
def generate_delta(length):
    delta = np.zeros(length)
    delta[0] = 1
    return delta

def generate_sine(frequency, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

# Parameters
sample_rate = 48000  
duration = 1  # 1 second
num_samples = int(sample_rate * duration)

# Generate delta function
delta = generate_delta(num_samples)

# Get embeddings for delta function
delta_embeddings = model._get_embedding_from_data([delta])

# Ensure delta_embeddings is a NumPy array
if not isinstance(delta_embeddings, np.ndarray):
    delta_embeddings = np.array(delta_embeddings)

# Calculate energy
energy_norm = np.linalg.norm(delta_embeddings)
energy_db = 20 * np.log10(energy_norm)

print(f"Energy (Frobenius norm) of the delta response: {energy_norm}")
print(f"Energy in decibels (dB): {energy_db} dB")

# %%
# Plot delta response
plt.figure(figsize=(12, 6))
plt.plot(delta_embeddings.squeeze())
plt.title("CLAP Delta Function Response")
plt.xlabel("Embedding Dimension")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# %%
# Frequency response analysis
fft_result = fft(delta_embeddings.squeeze())
frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)

plt.figure(figsize=(12, 6))
plt.semilogx(frequencies[:len(frequencies)//2], 20 * np.log10(np.abs(fft_result[:len(fft_result)//2])))
plt.title("Frequency Response of CLAP Delta Function")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()

# %%
# https://stackoverflow.com/questions/25191620/
#   creating-lowpass-filter-in-scipy-understanding-methods-and-units

import numpy as np
from scipy.signal import butter, filtfilt, freqz, lfilter
from matplotlib import pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, filter_type='filtfilt'):
    b, a = butter_lowpass(cutoff, fs, order=order)
    if filter_type == 'lfilter':
        y = lfilter(b, a, data)
    elif filter_type == 'filtfilt':
        y = filtfilt(b, a, data)
    return y

orders = [5, 10, 20, 30, 40]

for order in orders:
    fs = 30.0       # sample rate, Hz
    cutoff = 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title(f"Lowpass Filter Frequency Response. Order: {order}")
    plt.xlabel('Frequency [Hz]')
    plt.grid()


    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    T = 5.0             # seconds
    n = int(T * fs)     # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) \
            + 0.5*np.sin(12.0*2*np.pi*t)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)
    y_lfilter = butter_lowpass_filter(data, cutoff, fs, order, filter_type='lfilter')

    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data (filtfilt)')
    plt.plot(t, y_lfilter, 'r-', linewidth=2, label='filtered data (lfilter)')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()