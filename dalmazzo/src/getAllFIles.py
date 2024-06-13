import os
import formExtractor as fem

def load_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp3"):
                audio_files.append(os.path.join(root, file))
    return audio_files


path = '/home/laura/aimir/suno/'
saveToPath = path + 'form/'

song_files = path + 'audio'
files = load_audio_files(song_files)

print('number of files:', len(files))
print('-----------------------------\n')
print('Loading files...')
formData = fem.formExtractor()

for song in files:
    id_file = song.split('/')[-1].split('.')[0]
    print('Processing file:', id_file+'.mp3')
    formData.getFormAndSave(6, song, id_file, saveToPath)