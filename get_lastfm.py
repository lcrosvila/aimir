# %%
# file gotten from: https://github.com/renesemela/lastfm-dataset-2020/blob/master/datasets/lastfm_dataset_2020/lastfm_dataset_2020.db
import sys
import os
import sqlite3
import json
import glob

# if db file does not exist, download it
if not os.path.exists('lastfm_dataset_2020.db'):
    os.system('wget https://github.com/renesemela/lastfm-dataset-2020/raw/master/datasets/lastfm_dataset_2020/lastfm_dataset_2020.db')
          
# %%
db_file_path = 'lastfm_dataset_2020.db'

if not os.path.exists('/home/laura/aimir/lastfm/audio'):
    os.makedirs('/home/laura/aimir/lastfm/audio')
if not os.path.exists('/home/laura/aimir/lastfm/metadata'):
    os.makedirs('/home/laura/aimir/lastfm/metadata')

# Connect to the database
conn = sqlite3.connect(db_file_path)

sql_query = f"""
        SELECT * FROM metadata
    """

# Execute the query
rows = conn.execute(sql_query).fetchall()

# get the names of the columns
columns = [description[0] for description in conn.execute(sql_query).description]

# %%
import requests
import re
from pytube import YouTube

def download_full_song(row, out_dir='/home/laura/aimir/lastfm/audio'):
    if os.path.join(out_dir, f'{row[0]}.mp3') in glob.glob(os.path.join(out_dir, '*.mp3')):
        return None
    lastfm_url = row[3]

    response = requests.get(lastfm_url)
    # find the youtube url from the response
    if 'https://www.youtube.com/watch?v=' not in response.text:
        return None
    
    youtube_id = re.search(r'href="https://www.youtube.com/watch\?v=(.*?)"', response.text).group(1)
    youtube_url = f'https://www.youtube.com/watch?v={youtube_id}'

    # use pytube to download the video
    os.system(f'pytube {youtube_url}')
    # find the mp4 file in the current directory    
    if len(glob.glob('/home/laura/*.mp4')) == 0:
        print(glob.glob('/home/laura/*.mp4'))
        return None
    
    mp4_file = [file for file in glob.glob('/home/laura/*.mp4')][0]

    # convert the mp4 file to mp3
    file_path = os.path.join(out_dir, f'{row[0]}.mp3')
    os.system(f"ffmpeg -y -i '{mp4_file}' -q:a 0 -map a {file_path}")
    # remove the mp4 file
    os.system(f"rm '{mp4_file}'")

    return youtube_url

# %%
# # shuffle the rows and select the first 10000
import random
random.seed(0)
random.shuffle(rows)
rows = rows[:12000]

for row in rows:
    # download full song
    youtube_url = download_full_song(row)
    if youtube_url is None:
        continue
    # add youtube url to metadata
    row = list(row)
    row.append(youtube_url)
    columns.append('youtube_url')
    # save the row as json file in 'metadata'
    with open(f'/home/laura/aimir/lastfm/metadata/{row[0]}.json', 'w') as f:
        json.dump(dict(zip(columns, row)), f)