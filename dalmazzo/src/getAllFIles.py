import os
import argparse
import formExtractor as fem

def load_audio_files(path):
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp3"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def main(source):
    base_path = f'/home/laura/aimir/{source}/'
    save_to_path = os.path.join(base_path, 'form/')
    
    song_files = os.path.join(base_path, 'audio')
    files = load_audio_files(song_files)

    print('number of files:', len(files))
    print('-----------------------------\n')
    print('Loading files...')
    form_data = fem.formExtractor()

    for song in files:
        id_file = os.path.basename(song).split('.')[0]
        print('-----------------------------')
        print('Processing Song:', id_file + '.mp3')
        form_data.getFormAndSave(6, song, id_file, save_to_path)

if __name__ == "__main__":
    class CustomArgParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            print(f'\nError: {message}')
            print('\nAllowed values for source are: suno, lastfm, boomy')
            self.exit(2)
    
    parser = CustomArgParser(description='Process audio files from specified source.')
    parser.add_argument('source', choices=['suno', 'lastfm', 'boomy'], help='The source of the audio files to process (suno, lastfm, or boomy)')
    args = parser.parse_args()
    
    main(args.source)
