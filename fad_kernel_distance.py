#%%
from pathlib import Path
import json
import time
import torch
from audio_metrics import (
    async_audio_loader,
    multi_audio_slicer,
    EmbedderPipeline,
    AudioMetrics,
    CLAP,
)
from audio_metrics.example_utils import generate_audio_samples

#%%
audio_dir = Path("audio_samples")
win_dur = 5.0
n_pca = 64
dev = torch.device("cuda")

# check time
start_time = time.time()
# load audio samples from files in `audio_dir`
real_items = async_audio_loader("/home/laura/aimir/small/lastfm/audio", recursive=False, file_patterns=["*.mp3"], num_workers=4)
fake_items = async_audio_loader("/home/laura/aimir/small/suno/audio", recursive=False, file_patterns=["*.mp3"], num_workers=4)

end_time = time.time()
print(f"Time to load audio samples: {end_time - start_time:.2f} s")
# %%
# iterate over windows
real_items = multi_audio_slicer(real_items, win_dur)
fake_items = multi_audio_slicer(fake_items, win_dur)

print("creating embedder")
embedder = EmbedderPipeline({"clap": CLAP(dev)})
print("computing 'real' embeddings")
real_embeddings = embedder.embed_join(real_items)
print("computing 'fake' embeddings")
fake_embeddings = embedder.embed_join(fake_items)

# set the background data for the metrics
# use PCA projection of embeddings without whitening
metrics = AudioMetrics()
metrics.set_background_data(real_embeddings)
metrics.set_pca_projection(n_pca, whiten=True)

print("comparing 'real' to 'fake' data")
result = metrics.compare_to_background(fake_embeddings)
print(json.dumps(result, indent=2))

# %%
print(len(real_embeddings))
print(len(fake_embeddings))