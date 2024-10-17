import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def add_dc_drift(y, drift_amount):
    return y + drift_amount

def process_and_save_audio(input_file, output_file, drift_amount):
    # Load audio
    y, sr = load_audio(input_file)
    
    # Add DC drift
    y_drifted = add_dc_drift(y, drift_amount)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the modified audio
    sf.write(output_file, y_drifted, sr)

def process_dataset(dataset, drift_amount):
    # Read the list of audio files
    with open(f'/home/laura/aimir/{dataset}/sample.txt', 'r') as f:
        audio_files = f.read().splitlines()
    
    input_files = [f'/home/laura/aimir/{dataset}/audio/{audio_file}.mp3' for audio_file in audio_files]
    output_files = [f'/home/laura/aimir/{dataset}/audio/DC_drifted/{audio_file}.mp3' for audio_file in audio_files]
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_and_save_audio, input_file, output_file, drift_amount) 
                   for input_file, output_file in zip(input_files, output_files)]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset}"):
            pass

if __name__ == "__main__":
    # List of datasets to process
    datasets = ['lastfm']
    
    # Amount of DC drift to add (you can adjust this value)
    drift_amount = 0.000288 - 0.000007 # suno - lastfm Mean of song means
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        process_dataset(dataset, drift_amount)
        print(f"Finished processing {dataset}")

    print("All datasets processed successfully!")