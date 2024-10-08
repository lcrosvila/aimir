# %%
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def prepare_data(results):
    data = []
    for classifier, transformations in results.items():
        original_scores = transformations['original']['sample']['coarse']
        for attack_type, params in transformations.items():
            for attack_value, metrics in params.items():
                for class_name, class_metrics in metrics['coarse'].items():
                    row = {
                        'classifier': classifier,
                        'class': class_name,
                        'attack_type': attack_type,
                        'attack_value': attack_value if attack_type != 'original' else 'sample',
                        'f1': class_metrics['f1'],
                        'precision': class_metrics['precision'],
                        'recall': class_metrics['recall'],
                        'accuracy': class_metrics['accuracy']
                    }
                    data.append(row)
                    
                    # Add original score for reference
                    if attack_type in ['high_pass', 'noise']:
                        original_row = row.copy()
                        original_row['attack_value'] = 0
                        for metric in ['f1', 'precision', 'recall', 'accuracy']:
                            original_row[metric] = original_scores[class_name][metric]
                        data.append(original_row)
                    elif attack_type in ['low_pass', 'decrease_sr']:
                        original_row = row.copy()
                        original_row['attack_value'] = 44100 if class_name == 'lastfm' else 48000
                        for metric in ['f1', 'precision', 'recall', 'accuracy']:
                            original_row[metric] = original_scores[class_name][metric]
                        data.append(original_row)
                        
    return pd.DataFrame(data)

def plot_scores(data, score='f1'):
    attack_types = data['attack_type'].unique()
    classifiers = data['classifier'].unique()

    for classifier in classifiers:
        classifier_data = data[data['classifier'] == classifier]
        
        for attack_type in attack_types:
            if attack_type == 'original':
                continue
            
            plt.figure(figsize=(12, 8))
            attack_data = classifier_data[classifier_data['attack_type'] == attack_type]
            
            unique_x = []

            for class_name in ['lastfm', 'suno', 'udio']:
                class_data = attack_data[attack_data['class'] == class_name]
                
                # Sort data by attack_value
                class_data = class_data.sort_values('attack_value')
                
                # Convert attack_value to numeric
                x = pd.to_numeric(class_data['attack_value'], errors='coerce')

                unique_x = np.unique(np.concatenate((unique_x, x)))
                
                plt.plot(x, class_data[score], 'x--', markersize=10, label=class_name)
            
            plt.title(f'{classifier}: {score} Score vs {attack_type.replace("_", " ").title()} Value', fontsize=16)
            plt.xlabel(f'{attack_type.replace("_", " ").title()} Value', fontsize=14)
            plt.ylabel(f'{score} Score', fontsize=14)
            
            # Ensure x-axis contains all values in the data
            plt.xticks(unique_x)
            
            plt.xticks(rotation=45, ha='right')
            
            plt.legend(title='Class', title_fontsize='13', fontsize='12')
            
            # Add a grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    results = load_results('evaluation_results_hierarchical.pkl')
    data = prepare_data(results)
    plot_scores(data, 'f1')