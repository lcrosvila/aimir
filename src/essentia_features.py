import essentia
import essentia.standard as es
import json
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Function to extract features and save them to a JSON file
def process_file(args):
    audio_path, features_path = args
    print(audio_path)

    # Skip if the file already exists
    if os.path.exists(features_path):
        return

    # Extract features using Essentia's MusicExtractor
    try:
        features, features_frames = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'], 
            rhythmStats=['mean', 'stdev'], 
            tonalStats=['mean', 'stdev']
        )(audio_path)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return
    
    # Prepare the dictionary to store features
    features_dict = {}
    for feature in features.descriptorNames():
        if isinstance(features[feature], np.ndarray):
            features_dict[feature] = features[feature].tolist()
        else:
            features_dict[feature] = features[feature]

    # if the folder does not exist, create it
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    
    # Save the features to a JSON file
    with open(features_path, 'w') as f:
        json.dump(features_dict, f, indent=4)

# List of folders to process
folders = [
    '/home/laura/aimir/suno', 
    '/home/laura/aimir/udio', 
    '/home/laura/aimir/lastfm'
]

# Iterate over each folder with progress bar
for folder in tqdm(folders, desc="Processing folders"):
    audio_files = []

    for split in ['train', 'val', 'test']:
        with open(f'{folder}/{split}.txt', 'r') as f:
            audio_files += f.read().splitlines()    
    
    # Prepare file paths for processing
    file_paths = [
        (os.path.join(folder, 'audio', file + '.mp3'), 
         os.path.join(folder, 'essentia_features', file + '.json')) 
        for file in audio_files
    ]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, file_paths), 
                  total=len(file_paths), desc=f"Processing files in {folder}", leave=False))
