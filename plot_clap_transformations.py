# %%
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import random

def get_transformed_file(orig_file, transformation, param):
    file_name = orig_file.split('/')[-1]
    folder = orig_file.split('/')[-5]

    if 'noise' in transformation:
        param = str(param).replace(".", "_")
        
    file = f'/home/laura/aimir/{folder}/audio/transformed/{transformation}_{param}/{file_name}'
    if not os.path.exists(file):
        print(f"File {file} does not exist.")

    return np.load(file, mmap_mode='r')

def get_table_variances(orig_file, transformations, scaler):
    original = np.load(orig_file, mmap_mode='r')
    class_name = orig_file.split('/')[-5]
    
    top_10_indices_all = {}

    for trans, params in transformations.items():
        X = []
        y = []
        X.append(original)

        if trans in ['high_pass', 'noise']:
            y.append(0)

        for param in params:
            X.append(get_transformed_file(orig_file, trans, param))
            y.append(param)

        if trans in ['low_pass', 'decrease_sr']:
            y.append(44100 if class_name == 'lastfm' else 48000)

        X = np.vstack(X)
        # scale
        X = scaler.transform(X)
        # Flip X and y
        X = X.T
        y = np.array(y)

        # Compute the variance across the rows (features)
        variances = np.var(X, axis=1)

        # Sort by absolute variance and select top 10 indices
        top_10_indices = np.argsort(-np.abs(variances))[:10]
        
        top_10_indices_all[trans] = top_10_indices
        
    return pd.DataFrame(top_10_indices_all)

def plot_transformations(orig_file, transformations, scaler):
    original = np.load(orig_file, mmap_mode='r')
    class_name = orig_file.split('/')[-5]
    
    top_10_indices_all = {}

    for trans, params in transformations.items():
        X = []
        y = []
        X.append(original)
        
        if trans in ['high_pass', 'noise']:
            y.append(0)

        for param in params:
            X.append(get_transformed_file(orig_file, trans, param))
            y.append(param)

        if trans in ['low_pass', 'decrease_sr']:
            y.append(44100 if class_name == 'lastfm' else 48000)

        X = np.vstack(X)
        # scale
        X = scaler.transform(X)
        # Flip X and y
        X = X.T
        y = np.array(y)

        # Compute the variance across the rows (features)
        variances = np.var(X, axis=1)

        # Sort by absolute variance and select top 10 indices
        top_10_indices = np.argsort(-np.abs(variances))[:10]
        
        top_10_indices_all[trans] = top_10_indices
        
        # Plot heatmap for all features
        fig, ax1 = plt.subplots(figsize=(8, 12))

        # without the colorbar
        sns.heatmap(X, xticklabels=y, ax=ax1, cbar=False)

        # Annotate only the top 10 rows with their variance
        for i in top_10_indices:
            ax1.text(X.shape[1] + 0.5, i + 0.5, f"Feature {i}: {variances[i]:.2e}", va='center', ha='left', color='black')

        # Adjust the plot to fit the annotations
        ax1.set_title(trans)
        plt.tight_layout()
        plt.show()

    print('Table with top 10 indices (with highest var) for all transformations:')
    print(pd.DataFrame(top_10_indices_all))

def calculate_embedding_angles(orig_file, transformations, scaler):
    original = np.load(orig_file, mmap_mode='r')
    class_name = orig_file.split('/')[-5]
    
    angle_results = {}

    for trans, params in transformations.items():
        X = []
        y = []
        X.append(original)
        
        if trans in ['high_pass', 'noise']:
            y.append(0)
        elif trans in ['low_pass', 'decrease_sr']:
            y.append(44100 if class_name == 'lastfm' else 48000)

        for param in params:
            X.append(get_transformed_file(orig_file, trans, param))
            y.append(param)

        X = np.vstack(X)
        # scale
        X = scaler.transform(X)
        
        # Calculate angles
        original_embedding = X[0]
        angles = []
        for embedding in X[1:]:
            dot_product = np.dot(original_embedding, embedding)
            norm_original = np.linalg.norm(original_embedding)
            norm_embedding = np.linalg.norm(embedding)
            angle = np.arccos(dot_product / (norm_original * norm_embedding))
            angles.append(np.degrees(angle))
        
        angle_results[trans] = {param: angle for param, angle in zip(y[1:], angles)}
    
    return angle_results


# %%
transformations = {
    'low_pass': [5000, 8000, 10000, 12000, 16000, 20000],
    'high_pass': [5000, 8000, 10000, 12000, 16000, 20000],
    'decrease_sr': [8000, 16000, 22050, 24000, 44100],
    'noise': [0.005, 0.01]
}

folder = 'suno'
transformed = os.listdir(f'/home/laura/aimir/{folder}/audio/transformed/noise_0_005')

# get random file
# random.seed(42)
random_file = random.choice(transformed)
orig_file = f'/home/laura/aimir/{folder}/audio/embeddings/clap-laion-music/{random_file}'

# %%
with open('models_and_scaler.pkl', 'rb') as f:
    saved_data = pickle.load(f)

scaler = saved_data['scaler']

# plot_transformations(orig_file, transformations, scaler)
print(get_table_variances(orig_file, transformations, scaler))

# %%
angles = calculate_embedding_angles(orig_file, transformations, scaler)

print(pd.DataFrame(angles).sort_index()[['low_pass', 'high_pass']].dropna())
print(pd.DataFrame(angles).sort_index()[['decrease_sr']].dropna())
print(pd.DataFrame(angles).sort_index()[['noise']].dropna())