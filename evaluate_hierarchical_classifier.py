import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from hiclass import LocalClassifierPerNode

def load_single_embedding(file):
    if file.endswith('.npy'):
        try:
            data = np.load(file, mmap_mode='r')
        except ValueError:
            print(f"Error loading {file}")
        return np.load(file, mmap_mode='r')
    return None

def load_embeddings(files):
    valid_files = [f for f in files if f.endswith('.npy')]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_embedding, file) for file in valid_files]
        embeddings = [future.result() for future in as_completed(futures) if future.result() is not None]
    
    if not all(embedding.shape == embeddings[0].shape for embedding in embeddings):
        raise ValueError("Inconsistent embedding shapes detected.")

    return np.array(embeddings)

def get_split(split, embedding, folders):
    files = []
    y = []
    for folder in folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            folder_files = f.read().splitlines()
            files.extend([f'/home/laura/aimir/{folder}/audio/embeddings/{embedding}/{file}.npy' for file in folder_files])
            y.extend([folder] * len(folder_files))
    
    X = load_embeddings(files)
    y = np.array(y)
    return X, y

def evaluate_classifier(clf, X, y, class_hierarchy):
    y_pred = clf.predict(X)
    # Evaluate parent level
    y_parent = y[:, 0]
    y_pred_parent = y_pred[:, 0]
    
    parent_metrics = {
        'accuracy': accuracy_score(y_parent, y_pred_parent),
        'precision': precision_score(y_parent, y_pred_parent, average='weighted'),
        'recall': recall_score(y_parent, y_pred_parent, average='weighted'),
        'f1': f1_score(y_parent, y_pred_parent, average='weighted')
    }
    
    # Evaluate child level
    y_child = y[:, 1]
    y_pred_child = y_pred[:, 1]

    child_metrics = {
        'accuracy': accuracy_score(y_child, y_pred_child),
        'precision': precision_score(y_child, y_pred_child, average='weighted'),
        'recall': recall_score(y_child, y_pred_child, average='weighted'),
        'f1': f1_score(y_child, y_pred_child, average='weighted')
    }

    # evaluate child level for class (suno, udio, lastfm)
    coarse_metrics = {
        'suno': {'accuracy': accuracy_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno']),
                 'precision': precision_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted'),
                 'recall': recall_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted'),
                 'f1': f1_score(y_child[y_child == 'suno'], y_pred_child[y_child == 'suno'], average='weighted')},
        'udio': {'accuracy': accuracy_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio']),
                 'precision': precision_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted'),
                 'recall': recall_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted'),
                 'f1': f1_score(y_child[y_child == 'udio'], y_pred_child[y_child == 'udio'], average='weighted')},
        'lastfm': {'accuracy': accuracy_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm']),
                   'precision': precision_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted'),
                   'recall': recall_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted'),
                   'f1': f1_score(y_child[y_child == 'lastfm'], y_pred_child[y_child == 'lastfm'], average='weighted')}
    }
    
    return parent_metrics, child_metrics, coarse_metrics

def get_transformed(transformation, param, embedding, folders):
    files = []
    y = []
    for folder in folders:
        transform_path = f'/home/laura/aimir/{folder}/audio/transformed/{transformation}_{param}'
        if not os.path.exists(transform_path):
            continue
        folder_files = os.listdir(transform_path)
        files.extend([f'{transform_path}/{file}' for file in folder_files])
        y.extend([folder] * len(folder_files))
    
    X = load_embeddings(files)
    y = np.array(y)
    return X, y

def main():
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    
    models = saved_data['models']
    scaler = saved_data['scaler']
    
    folders = ['suno', 'udio', 'lastfm']
    class_hierarchy = {
        'AI': ['suno', 'udio'],
        'nonAI': ['lastfm']
    }

    transformations = {
        'original': ['sample'],
        'low_pass': [5000, 8000, 10000, 12000, 16000, 20000],
        'high_pass': [5000, 8000, 12000, 16000, 20000],
        'noise': [0.005, 0.01],
        'decrease_sr': [8000, 16000, 22050, 24000, 44100]
    }
    
    results = {model_name: {trans: {param: {'parent': {}, 'child': {}} for param in params} 
                            for trans, params in transformations.items()} 
               for model_name in models.keys()}
    
    for trans, params in transformations.items():
        for param in params:
            print(f"\nEvaluating on {trans} {param}:")
            if trans == 'original':
                X, y = get_split('sample', 'clap-laion-music', folders)
            elif trans == 'noise':
                X, y = get_transformed(trans, str(param).replace('.', '_'), 'clap-laion-music', folders)
            else:
                X, y = get_transformed(trans, param, 'clap-laion-music', folders)
            
            X_scaled = scaler.transform(X)
            y =  np.array([['AI', folder] for folder in y if folder in class_hierarchy['AI']] + 
                          [['nonAI', folder] for folder in y if folder in class_hierarchy['nonAI']])
            
            for model_name, model in models.items():
                print(f"\n{model_name.upper()} Classifier:")
                parent_metrics, child_metrics, coarse_metrics = evaluate_classifier(model, X_scaled, y, class_hierarchy)
                results[model_name][trans][param]['parent'] = parent_metrics
                results[model_name][trans][param]['child'] = child_metrics
                results[model_name][trans][param]['coarse'] = coarse_metrics
                
                print("Parent level metrics:")
                for metric, value in parent_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                print("\nChild level metrics:")
                for metric, value in child_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                print("\nCoarse level metrics:")
                for class_, metrics in coarse_metrics.items():
                    print(f"Class: {class_}")
                    for metric, value in metrics.items():
                        print(f"{metric}: {value:.4f}")

                
    
    with open('evaluation_results_hierarchical.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nEvaluation results have been saved to 'evaluation_results_hierarchical.pkl'")

if __name__ == "__main__":
    main()