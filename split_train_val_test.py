import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

## Boomy

# Define the folder path
folder = '/home/laura/aimir/boomy/metadata'

# Get the list of files in the folder
files = os.listdir(folder)

# Initialize an empty list to store results
results = []

# Iterate over each file in the folder
for file in files:
    # Open the file
    with open(os.path.join(folder, file), 'r') as f:
        # Load JSON data
        data = json.load(f)
        # Append filename and artistName to the results list
        results.append({'filename': file[:-5], 'artistName': data['artistName']})

# Create a DataFrame from the results
df_boomy = pd.DataFrame(results)

# make a train val test split making sure no 'artistName' is in more than one set
# Extract unique artist names
unique_artists = df_boomy['artistName'].unique()

# Split unique artist names into train and temp sets
train_artists, temp_artists = train_test_split(unique_artists, test_size=0.2, random_state=42)

# Split temp artists into validation and test sets
val_artists, test_artists = train_test_split(temp_artists, test_size=0.5, random_state=42)

# Filter the dataframe based on the split artist names
train_df = df_boomy[df_boomy['artistName'].isin(train_artists)]
val_df = df_boomy[df_boomy['artistName'].isin(val_artists)]
test_df = df_boomy[df_boomy['artistName'].isin(test_artists)]

# # save the train, val, test filenames to a txt file
# with open('/home/laura/aimir/boomy/train.txt', 'w') as f:
#     f.write('\n'.join(train_df['filename'].values))

# with open('/home/laura/aimir/boomy/val.txt', 'w') as f:
#     f.write('\n'.join(val_df['filename'].values))

# with open('/home/laura/aimir/boomy/test.txt', 'w') as f:
#     f.write('\n'.join(test_df['filename'].values))

# Sample 50 songs from train_df
train_df_sample = train_df.sample(n=50, random_state=42)
# save 
with open('/home/laura/aimir/boomy/sample.txt', 'w') as f:
    f.write('\n'.join(train_df_sample['filename'].values))

## Suno

# Define the folder path
folder = '/home/laura/aimir/suno/metadata'

# Get the list of files in the folder
files = os.listdir(folder)

# Initialize an empty list to store results
results = []

# Iterate over each file in the folder
for file in files:
    # Open the file
    with open(os.path.join(folder, file), 'r') as f:
        # Load JSON data
        data = json.load(f)
        # Append filename and prompt to the results list
        results.append({'filename': file[:-5], 'prompt': data['prompt']})

# Create a DataFrame from the results
df_suno = pd.DataFrame(results)

# make a train val test split making sure no 'prompt' is in more than one set
# Extract unique prompts
unique_prompts = df_suno['prompt'].unique()

# Split unique prompts into train and temp sets
train_prompts, temp_prompts = train_test_split(unique_prompts, test_size=0.2, random_state=42)

# Split temp prompts into validation and test sets
val_prompts, test_prompts = train_test_split(temp_prompts, test_size=0.5, random_state=42)

# Filter the dataframe based on the split prompts
train_df = df_suno[df_suno['prompt'].isin(train_prompts)]
val_df = df_suno[df_suno['prompt'].isin(val_prompts)]
test_df = df_suno[df_suno['prompt'].isin(test_prompts)]

# # save the train, val, test filenames to a txt file
# with open('/home/laura/aimir/suno/train.txt', 'w') as f:
#     f.write('\n'.join(train_df['filename'].values))

# with open('/home/laura/aimir/suno/val.txt', 'w') as f:
#     f.write('\n'.join(val_df['filename'].values))

# with open('/home/laura/aimir/suno/test.txt', 'w') as f:
#     f.write('\n'.join(test_df['filename'].values))

# Sample 50 songs from train_df
train_df_sample = train_df.sample(n=50, random_state=42)
# save
with open('/home/laura/aimir/suno/sample.txt', 'w') as f:
    f.write('\n'.join(train_df_sample['filename'].values))

## Udio

# Define the folder path
folder = '/home/laura/aimir/udio/metadata'

# Get the list of files in the folder
files = os.listdir(folder)

# Initialize an empty list to store results
results = []

# Iterate over each file in the folder
for file in files:
    # Open the file
    with open(os.path.join(folder, file), 'r') as f:
        # Load JSON data
        data = json.load(f)
        # Append filename and prompt to the results list
        results.append({'filename': file[:-5], 'prompt': data['prompt']})

# Create a DataFrame from the results
df_udio = pd.DataFrame(results)

# make a train val test split making sure no 'prompt' is in more than one set
# Extract unique prompts
unique_prompts = df_udio['prompt'].unique()

# Split unique prompts into train and temp sets
train_prompts, temp_prompts = train_test_split(unique_prompts, test_size=0.2, random_state=42)

# Split temp prompts into validation and test sets
val_prompts, test_prompts = train_test_split(temp_prompts, test_size=0.5, random_state=42)

# Filter the dataframe based on the split prompts
train_df = df_udio[df_udio['prompt'].isin(train_prompts)]
val_df = df_udio[df_udio['prompt'].isin(val_prompts)]
test_df = df_udio[df_udio['prompt'].isin(test_prompts)]

# # save the train, val, test filenames to a txt file
# with open('/home/laura/aimir/udio/train.txt', 'w') as f:
#     f.write('\n'.join(train_df['filename'].values))

# with open('/home/laura/aimir/udio/val.txt', 'w') as f:
#     f.write('\n'.join(val_df['filename'].values))

# with open('/home/laura/aimir/udio/test.txt', 'w') as f:
#     f.write('\n'.join(test_df['filename'].values))

# Sample 50 songs from train_df
train_df_sample = train_df.sample(n=50, random_state=42)
# save
with open('/home/laura/aimir/udio/sample.txt', 'w') as f:
    f.write('\n'.join(train_df_sample['filename'].values))

## Lastfm

# Define the folder path
folder = '/home/laura/aimir/lastfm/metadata'

# Get the list of files in the folder
files = os.listdir(folder)

# Initialize an empty list to store results
results = []

# Iterate over each file in the folder
for file in files:
    # Open the file
    with open(os.path.join(folder, file), 'r') as f:
        # Load JSON data
        data = json.load(f)
        # Append filename and artist to the results list
        results.append({'filename': file[:-5], 'artist': data['artist']})

# Create a DataFrame from the results
df_lastfm = pd.DataFrame(results)

# make a train val test split making sure no 'artist' is in more than one set
# Extract unique artists
unique_artists = df_lastfm['artist'].unique()

# Split unique artists into train and temp sets
train_artists, temp_artists = train_test_split(unique_artists, test_size=0.2, random_state=42)

# Split temp artists into validation and test sets
val_artists, test_artists = train_test_split(temp_artists, test_size=0.5, random_state=42)

# Filter the dataframe based on the split artist names
train_df = df_lastfm[df_lastfm['artist'].isin(train_artists)]
val_df = df_lastfm[df_lastfm['artist'].isin(val_artists)]
test_df = df_lastfm[df_lastfm['artist'].isin(test_artists)]

# # save the train, val, test filenames to a txt file
# with open('/home/laura/aimir/lastfm/train.txt', 'w') as f:
#     f.write('\n'.join(train_df['filename'].values))

# with open('/home/laura/aimir/lastfm/val.txt', 'w') as f:
#     f.write('\n'.join(val_df['filename'].values))

# with open('/home/laura/aimir/lastfm/test.txt', 'w') as f:
#     f.write('\n'.join(test_df['filename'].values))

# Sample 50 songs from train_df
train_df_sample = train_df.sample(n=50, random_state=42)
# save
with open('/home/laura/aimir/lastfm/sample.txt', 'w') as f:
    f.write('\n'.join(train_df_sample['filename'].values))