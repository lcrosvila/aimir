import time
import pandas as pd
from transformers import pipeline
# tqdm pandas apply
from tqdm import tqdm
tqdm.pandas()

print('MNLI using model: facebook/bart-large-mnli')
print('Classifying lyrics as serious or parody...')
print('Considering only english lyrics...')


suno_df = pd.read_pickle('../suno/metadata.pkl')
suno_df = suno_df.dropna(subset=['prompt', 'language'])
suno_df = suno_df[suno_df['language'].str.startswith('eng')]
print('suno_df:', suno_df.shape)

udio_df = pd.read_pickle('../udio/metadata.pkl')
udio_df = udio_df.dropna(subset=['lyrics', 'language'])
udio_df = udio_df[udio_df['language'].str.startswith('eng')]
print('udio_df:', udio_df.shape)



classifier = pipeline(
    'zero-shot-classification', 
    model='facebook/bart-large-mnli',
    )

total_time = time.time()

def classify_lyrics(sequence_to_classify):
    # sequence_to_classify = row['prompt']
    if not sequence_to_classify:
        return None
    candidate_labels = ['parody','serious']
    hypothesis_template = "These are the lyrics of a {} song."
    res = classifier(
        sequence_to_classify, 
        candidate_labels, 
        # multi_label=True, # if more than one choice can be correct
        hypothesis_template=hypothesis_template
        )
    return {res['labels'][i]: res['scores'][i] for i in range(len(res['labels']))}

# classify suno_df
print('Classifying suno_df...')
#time it
start = time.time()
# suno_df = classify_lyrics(suno_df)
suno_df['serious/parody'] = suno_df.progress_apply(lambda x: classify_lyrics(x['prompt']), axis=1)
suno_df.to_pickle('../suno/metadata.pkl')
print('Time taken:', time.time()-start)

# classify udio_df
print('Classifying udio_df...')
#time it
start = time.time()
# udio_df = classify_lyrics(udio_df)
udio_df['serious/parody'] = udio_df.progress_apply(lambda x: classify_lyrics(x['lyrics']), axis=1)
udio_df.to_pickle('../udio/metadata.pkl')
print('Time taken:', time.time()-start)

total_time = time.time()-total_time
print('Total time taken:', total_time)