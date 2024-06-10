# %%
import pandas as pd
import os
import json
import sox
import nltk
from nltk.corpus import stopwords

# Download the stopwords if not already downloaded
nltk.download('stopwords')

# %%
# Function to load JSON data safely
def load_json_safe(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Check if the data can be converted to a DataFrame
        if isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            print(f"Skipping {filepath}, unsupported JSON format")
            return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# %%
# Specify the directories containing the JSON files
directory1 = '/home/laura/aimir/suno/metadata'
directory2 = '/home/laura/aimir/udio/metadata'

# Initialize an empty list to store the dataframes
dataframes = []

# Load JSON files from directory1
for filename in os.listdir(directory1):
    if filename.endswith('.json'):
        filepath = os.path.join(directory1, filename)
        df = load_json_safe(filepath)
        # if the duration is none, calculate it
        if df['duration'].isnull().values.any():
            audio_file = os.path.join('/home/laura/aimir/suno/audio', filename.replace('.json', '.mp3'))
            if os.path.exists(audio_file):
                df['duration'] = sox.file_info.duration(audio_file)
            else:
                print(f"Audio file not found: {audio_file}")
        # add the filename as a column
        if df is not None:
            df['filename'] = filename
            df['service'] = 'suno'
        if df is not None:
            dataframes.append(df)

# Load JSON files from directory2
for filename in os.listdir(directory2):
    if filename.endswith('.json'):
        filepath = os.path.join(directory2, filename)
        df = load_json_safe(filepath)
        # if the duration is none, calculate it
        if df['duration'].isnull().values.any():
            audio_file = os.path.join('/home/laura/aimir/udio/audio', filename.replace('.json', '.mp3'))
            if os.path.exists(audio_file):
                df['duration'] = sox.file_info.duration(audio_file)
            else:
                print(f"Audio file not found: {audio_file}")
        # add the filename as a column
        if df is not None:
            df['filename'] = filename
            df['service'] = 'udio'
        if df is not None:
            dataframes.append(df)

# %%
# Combine the dataframes into a joint dataframe if there are any dataframes loaded
if dataframes:
    joint_df = pd.concat(dataframes, ignore_index=True)
    # Print the joint dataframe
    print(joint_df)
else:
    print("No valid JSON files were found.")

# %%
#save the joint dataframe to a CSV file
if joint_df is not None:
    joint_df.to_csv('joint_dataframe.csv', index=False)
# %%
# load joint dataframe from CSV file
joint_df = pd.read_csv('joint_dataframe.csv')
# %%
# print the column names
if joint_df is not None:
    print(joint_df.columns)

# %%
# print how many of the rows have duration
if 'duration' in joint_df.columns:
    print(joint_df['duration'].count())

# print the id of the ones that don't have duration
if 'duration' in joint_df.columns:
    print(joint_df[joint_df['duration'].isnull()]['id'])
#%%

# if play_count is null, take the value from the plays column
if 'play_count' in joint_df.columns:
    joint_df['play_count'] = joint_df['play_count'].fillna(joint_df['plays'])

# sort by play_count in descending order
if 'play_count' in joint_df.columns:
    joint_df = joint_df.sort_values(by='play_count', ascending=False)

# %%
# Get the list of English stop words
stop_words = set(stopwords.words('english'))
# make the play_count column an integer
if 'play_count' in joint_df.columns:
    joint_df['play_count'] = joint_df['play_count'].astype(int)
print(joint_df.head(2)[['id', 'prompt']].to_string(index=False))
# %%
# grab the tags and prompt of the top 50 rows and find which words are the most common
if 'play_count' in joint_df.columns:
    top_prompts = joint_df[['tags', 'prompt', 'play_count']].head(50)

    # Add the tags to the beginning of the prompt
    top_prompts['prompt'] = top_prompts['tags'] + ' ' + top_prompts['prompt']
    top_prompts['prompt'] = top_prompts['prompt'].str.lower()
    
    # Remove punctuation
    top_prompts['prompt'] = top_prompts['prompt'].str.replace('[^\w\s]', '', regex=True)
    
    # Fill NaN values with empty strings to avoid TypeError
    top_prompts['prompt'] = top_prompts['prompt'].fillna('')
    
    # Split the prompts into words
    top_prompts['prompt'] = top_prompts['prompt'].str.split()
    
    # Filter out stop words
    top_prompts['prompt'] = top_prompts['prompt'].apply(lambda x: [word for word in x if word not in stop_words])
    # remove numbers (even if they are written as words like 'one' 'two' etc)
    top_prompts['prompt'] = top_prompts['prompt'].apply(lambda x: [word for word in x if word not in ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']])
    top_prompts['prompt'] = top_prompts['prompt'].apply(lambda x: [word for word in x if not word.isdigit()])
    # remove 'chorus', 'verse' and 'outro'
    top_prompts['prompt'] = top_prompts['prompt'].apply(lambda x: [word for word in x if word not in ['chorus', 'verse', 'outro', 'bridge']])

    # Explode the list of words into separate rows
    top_prompts = top_prompts.explode('prompt')
    
    # Get the top words by play_count
    top_words = top_prompts.groupby('prompt')['play_count'].sum().sort_values(ascending=False)

    # Print the top words
    print(top_words.head(20))
    # plot
    top_words.head(30).plot(kind='bar')
