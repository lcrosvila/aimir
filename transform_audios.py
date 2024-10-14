import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from src.model_loader import CLAPMusic

# initialize model
model = CLAPMusic()
model.load_model()

def get_split_mp3(split, folders):
    files = []
    for folder in folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
            files.extend([f'/home/laura/aimir/{folder}/audio/{file}.mp3' for file in folder_files])
    return files

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_low_pass_filter(audio, sr, audio_path, cutoff=5000, order=5):
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/home/laura/aimir/{folder}/audio/transformed/low_pass_{cutoff}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        b, a = butter_lowpass(cutoff, sr, order=order)
        y = filtfilt(b, a, audio)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)
    return emb

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_high_pass_filter(audio, sr, audio_path, cutoff=5000, order=5):
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/home/laura/aimir/{folder}/audio/transformed/high_pass_{cutoff}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        b, a = butter_highpass(cutoff, sr, order=order)
        y = filtfilt(b, a, audio)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)
    return emb

def add_noise(audio, audio_path, noise_factor=0.005):
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/home/laura/aimir/{folder}/audio/transformed/noise_{str(noise_factor).replace(".", "_")}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        noise = np.random.randn(len(audio))
        y = audio + noise_factor * noise
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)
    return emb

def decrease_sample_rate(audio, sr, audio_path, target_sr=8000):
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/home/laura/aimir/{folder}/audio/transformed/decrease_sr_{target_sr}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        y = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y])[0]
    np.save(save_file, emb)
    return emb

def add_tone(audio, sr, audio_path, tone_freq=10000, tone_db=3):
    # Add a tone to an audio file at a specified frequency and amplitude.
    file = audio_path.split('/')[-1].split('.')[0]
    folder = audio_path.split('/')[-3]
    save_file = f'/home/laura/aimir/{folder}/audio/transformed/sine_{tone_freq}_{tone_db}/{file}.npy'
    if os.path.exists(save_file):
        return np.load(save_file)
    else:
        # Generate time array
        t = np.arange(len(audio)) / sr
        tone = np.sin(2 * np.pi * tone_freq * t)
        amplitude_factor = 10 ** (tone_db / 20)
        
        # Normalize the tone to match the amplitude of the original audio
        max_amplitude = np.max(np.abs(audio))
        tone = tone * max_amplitude * amplitude_factor
        
        # Add the tone to the original audio
        y_with_tone = audio + tone
        
        # Normalize the result to prevent clipping
        y_with_tone = y_with_tone / np.max(np.abs(y_with_tone))

    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    emb = model._get_embedding_from_data([y_with_tone])[0]
    np.save(save_file, emb)
    return emb
        

def main():
    split = 'sample'
    folders = ['suno', 'udio', 'lastfm']
    files = get_split_mp3(split, folders)

    for folder in folders:
        if not os.path.exists(f'/home/laura/aimir/{folder}/audio/transformed'):
            os.makedirs(f'/home/laura/aimir/{folder}/audio/transformed')

    model = CLAPMusic()
    model.load_model()

    for file in tqdm(files):
        audio, sr = librosa.load(file, sr=None)
        apply_low_pass_filter(audio, sr, file, cutoff=5000)
        apply_low_pass_filter(audio, sr, file, cutoff=8000)
        apply_low_pass_filter(audio, sr, file, cutoff=10000)
        apply_low_pass_filter(audio, sr, file, cutoff=12000)
        apply_low_pass_filter(audio, sr, file, cutoff=16000)
        apply_low_pass_filter(audio, sr, file, cutoff=20000)
        apply_high_pass_filter(audio, sr, file, cutoff=5000)
        apply_high_pass_filter(audio, sr, file, cutoff=8000)
        apply_high_pass_filter(audio, sr, file, cutoff=10000)
        apply_high_pass_filter(audio, sr, file, cutoff=12000)
        apply_high_pass_filter(audio, sr, file, cutoff=16000)
        apply_high_pass_filter(audio, sr, file, cutoff=20000)
        add_noise(audio, file, noise_factor=0.005)
        add_noise(audio, file, noise_factor=0.01)
        decrease_sample_rate(audio, sr, file, target_sr=8000)
        decrease_sample_rate(audio, sr, file, target_sr=16000)
        decrease_sample_rate(audio, sr, file, target_sr=22050)
        decrease_sample_rate(audio, sr, file, target_sr=24000)
        decrease_sample_rate(audio, sr, file, target_sr=44100)
        add_tone(audio, sr, file, tone_freq=10000, tone_db=3)
    
    print("All files processed.")

        
if __name__ == '__main__':
    main()