# %%
import os
import pandas as pd
import sox
import json

# %%
folders = ['/home/laura/aimir/suno', '/home/laura/aimir/lastfm', '/home/laura/aimir/boomy', '/home/laura/aimir/udio']

dataset_name = []
dataset_size = []
mean_song_duration = []
std_song_duration = []
total_duration = []

for folder in folders:
    audio_files = []
    for split in ['train', 'val', 'test']:
        with open(f'{folder}/{split}.txt', 'r') as f:
            audio_files += f.read().splitlines() 
    audio_files = [audio_file + '.mp3' for audio_file in audio_files]
    
    audio_folder = os.path.join(folder, 'audio')

    dataset_name.append(folder.split('/')[-1])
    # dataset_size.append(len(os.listdir(audio_folder)))
    dataset_size.append(len(audio_files))

    durations = []
    # for audio_file in os.listdir(audio_folder):
    for audio_file in audio_files:
        # if it's not mp3, skip
        if not audio_file.endswith('.mp3'):
            continue
        # audio_path = os.path.join(audio_folder, audio_file)
        # durations.append(sox.file_info.duration(audio_path))
        metadata_file = os.path.join(folder, 'metadata', audio_file.replace('.mp3', '.json'))
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                data = json.load(f)
                
                if 'duration' not in data:
                    audio_path = os.path.join(audio_folder, audio_file)
                    data['duration'] = sox.file_info.duration(audio_path)

                durations.append(data['duration'])

    mean_song_duration.append(pd.Series(durations).mean())
    std_song_duration.append(pd.Series(durations).std())
    total_duration.append(pd.Series(durations).sum())

# %%
def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def approx_h(seconds):
    hours = seconds // 3600
    return f"~{int(hours)}h"


# %%
df = pd.DataFrame({
    'Dataset': dataset_name,
    'Size': dataset_size,
    'Song Duration': [f"{mean:.2f} ± {std:.2f}s" for mean, std in zip(mean_song_duration, std_song_duration)],
    'Total Duration': [seconds_to_hms(duration) for duration in total_duration]
})

# %%
print(df.to_latex(index=False))

# %%
# print the sum of all dataset size
print(f"Total size: {sum(dataset_size)}")

# %%
# print the total boomy size in h, min, s
# the total duration is in hours minutes and seconds
boomy_duration = int(df[df['Dataset'] == 'boomy']['Total Duration'].values[0].split('h')[0]) * 3600 + int(df[df['Dataset'] == 'boomy']['Total Duration'].values[0].split('h')[1].split('m')[0]) * 60 + int(df[df['Dataset'] == 'boomy']['Total Duration'].values[0].split('m')[1].split('s')[0])
print(f"Boomy total duration: {seconds_to_hms(boomy_duration)}")
# do the same for suno
suno_duration = int(df[df['Dataset'] == 'suno']['Total Duration'].values[0].split('h')[0]) * 3600 + int(df[df['Dataset'] == 'suno']['Total Duration'].values[0].split('h')[1].split('m')[0]) * 60 + int(df[df['Dataset'] == 'suno']['Total Duration'].values[0].split('m')[1].split('s')[0])
print(f"Suno total duration: {seconds_to_hms(suno_duration)}")

# print the sum of suno and boomy total duration and size
print(f"Suno + Boomy size: {df[df['Dataset'].isin(['suno', 'boomy'])]['Size'].sum()}")
# the total duration is in hours minutes and seconds
sum_durations = 0
for row in df[df['Dataset'].isin(['suno', 'boomy'])].iterrows():
    sum_durations += int(row[1]['Total Duration'].split('h')[0]) * 3600 + int(row[1]['Total Duration'].split('h')[1].split('m')[0]) * 60 + int(row[1]['Total Duration'].split('m')[1].split('s')[0])
print(f"Suno + Boomy total duration: {seconds_to_hms(sum_durations)}")