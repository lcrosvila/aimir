import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

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

def evaluate_classifier(clf, X, y, graph, folders):
    y_pred = clf.predict(X)
    
    overall_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1': f1_score(y, y_pred, average='weighted')
    }
    
    with multi_labeled(y, y_pred, graph) as (y_, y_pred_, graph_):
        overall_metrics['h_fbeta'] = h_fbeta_score(y_, y_pred_, graph_)
    
    per_class_metrics = {}
    for i, folder in enumerate(folders):
        y_true_class = (y == str(i))
        y_pred_class = (y_pred == str(i))
        per_class_metrics[folder] = {
            'accuracy': accuracy_score(y_true_class, y_pred_class),
            'precision': precision_score(y_true_class, y_pred_class, average='binary'),
            'recall': recall_score(y_true_class, y_pred_class, average='binary'),
            'f1': f1_score(y_true_class, y_pred_class, average='binary')
        }
    
    return overall_metrics, per_class_metrics

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

    # Define the new structure for splits
    transformations = {
        'original': ['sample'],
        'low_pass': [5000, 8000, 10000, 12000, 16000, 20000],
        'high_pass': [5000, 8000, 12000, 16000, 20000],
        'noise': [0.005, 0.01],
        'decrease_sr': [8000, 16000, 22050, 24000, 44100]
    }
    
    results = {model_name: {trans: {param: {'overall': {}, 'per_class': {}} for param in params} 
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
            
            # Encode labels
            y = np.array([str(folders.index(label)) for label in y])

            X_scaled = scaler.transform(X)
            
            for model_name, model in models.items():
                print(f"\n{model_name.upper()} Classifier:")
                overall_metrics, per_class_metrics = evaluate_classifier(model, X_scaled, y, model.graph_, folders)
                results[model_name][trans][param]['overall'] = overall_metrics
                results[model_name][trans][param]['per_class'] = per_class_metrics
                
                print("Overall metrics:")
                for metric, value in overall_metrics.items():
                    print(f"{metric}: {value:.4f}")
                
                print("\nPer-class metrics:")
                for folder, metrics in per_class_metrics.items():
                    print(f"  {folder}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.4f}")
    
    with open('evaluation_results_structured.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nEvaluation results have been saved to 'evaluation_results_structured.pkl'")

if __name__ == "__main__":
    main()