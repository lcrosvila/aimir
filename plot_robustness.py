# %%
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, AutoMinorLocator

def load_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def prepare_data(results):
    data = []
    for classifier, transformations in results.items():
        original_scores = transformations['original']['sample']['fine_parent']
        for attack_type, params in transformations.items():
            for attack_value, metrics in params.items():
                for class_name, class_metrics in metrics['fine_parent'].items():
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
    return pd.DataFrame(data)

def generate_latex_table(results):
    latex_table = r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
    latex_table += "\nClassifier & Precision & Recall & F1-score & Accuracy \\\\ \hline\n"
    
    for classifier, transformations in results.items():
        classifier_data = transformations['original']['sample']['parent']
        
        precision = classifier_data['precision']
        recall = classifier_data['recall']
        f1_score = classifier_data['f1']
        accuracy = classifier_data['accuracy']
        
        latex_table += f"{classifier} & {float(precision):.3f} & {float(recall):.3f} & {float(f1_score):.3f} & {float(accuracy):.3f} \\\\ \n"

    latex_table += r"\hline\n\end{tabular}"
    latex_table += r"\caption{Classifier performance on the test set}\label{tab:test_results}\end{table}"
    
    # Print the LaTeX table
    print(latex_table)

def plot_scores(data, score='f1', save=False, save_dir=None):
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    classifiers = data['classifier'].unique()
    attack_types = ['low_pass', 'high_pass', 'noise', 'decrease_sr', 'dc_drift']

    # Define custom colors and markers
    palette = sns.color_palette("Set2", n_colors=3)
    markers = ['o', 's', '^']  # Different markers for the classes

    # Create the save directory if it does not exist and save=True
    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for classifier in classifiers:
        classifier_data = data[data['classifier'] == classifier]
        
        # Determine the common y-axis limits for low_pass and high_pass
        low_pass_data = classifier_data[classifier_data['attack_type'] == 'low_pass'][score]
        high_pass_data = classifier_data[classifier_data['attack_type'] == 'high_pass'][score]
        y_min = min(low_pass_data.min(), high_pass_data.min())
        y_max = max(low_pass_data.max(), high_pass_data.max())

        # Adjust limits with a margin of 0.05
        y_min_adjusted = y_min - 0.05
        y_max_adjusted = y_max + 0.05

        # Plot low_pass and high_pass together
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # fig.suptitle(f'{classifier}: {score.upper()} Score vs Low Pass and High Pass Filters', fontsize=16)
        
        for i, attack in enumerate(['low_pass', 'high_pass']):
            ax = ax1 if i == 0 else ax2
            attack_data = classifier_data[classifier_data['attack_type'] == attack]
            
            for j, class_name in enumerate(['lastfm', 'suno', 'udio']):
                class_data = attack_data[attack_data['class'] == class_name]
                class_data = class_data.sort_values('attack_value')
                x = pd.to_numeric(class_data['attack_value'])
                ax.plot(x, class_data[score], marker=markers[j], markersize=6, label=class_name, color=palette[j])
            
            ax.set_title(f'{attack.replace("_", " ").title()} Filter', fontsize=14)
            ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=12)
            ax.set_ylabel(f'{score.upper()} Score', fontsize=12)
            ax.legend(title='Class', title_fontsize='12', fontsize='10')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Set the common y-limits with the margin
            ax.set_ylim(y_min_adjusted, y_max_adjusted)

        plt.tight_layout()
        if save:
            # Save the figure in the specified directory
            fig_path = os.path.join(save_dir, f'{classifier}_low_high_pass_{score}.pdf')
            plt.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # Plot other attack types
        for attack_type in ['noise', 'decrease_sr', 'dc_drift']:
            plt.figure(figsize=(10, 6))
            attack_data = classifier_data[classifier_data['attack_type'] == attack_type]
            
            for j, class_name in enumerate(['lastfm', 'suno', 'udio']):
                class_data = attack_data[attack_data['class'] == class_name]
                class_data = class_data.sort_values('attack_value')
                x = pd.to_numeric(class_data['attack_value'], errors='coerce')
                plt.plot(x, class_data[score], marker=markers[j], markersize=6, label=class_name, color=palette[j])
            
            # plt.title(f'{classifier}: {score.upper()} Score vs {attack_type.replace("_", " ").title()}', fontsize=16)
            
            if attack_type == 'dc_drift':
                plt.xscale('log')
                plt.xlabel('DC Drift Value (log scale)', fontsize=12)
                plt.gca().xaxis.set_major_locator(LogLocator(base=10))
                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs='all'))
            elif attack_type == 'decrease_sr':
                plt.xlabel('Sample Rate (Hz)', fontsize=12)
            else:
                plt.xlabel(f'{attack_type.replace("_", " ").title()} Value', fontsize=12)
            
            plt.ylabel(f'{score.upper()} Score', fontsize=12)
            plt.legend(title='Class', title_fontsize='12', fontsize='10')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            if save:
                # Save the figure in the specified directory
                fig_path = os.path.join(save_dir, f'{classifier}_{attack_type}_{score}.pdf')
                plt.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

def plot_attacks_by_class(data, score='f1', save=False, save_dir=None):
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    classes = data['class'].unique()
    attack_types = ['low_pass', 'high_pass', 'noise', 'decrease_sr', 'dc_drift']

    # Define custom colors and markers for classifiers
    palette = sns.color_palette("Set2", n_colors=3)
    markers = ['o', 's', '^']

    # Create the save directory if it does not exist and save=True
    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for class_name in classes:
        class_data = data[data['class'] == class_name]
        
        # Plot low_pass and high_pass together
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Determine the common y-axis limits for low_pass and high_pass
        low_pass_data = class_data[class_data['attack_type'] == 'low_pass'][score]
        high_pass_data = class_data[class_data['attack_type'] == 'high_pass'][score]
        y_min = min(low_pass_data.min(), high_pass_data.min())
        y_max = max(low_pass_data.max(), high_pass_data.max())

        # Adjust limits with a margin of 0.05
        y_min_adjusted = y_min - 0.05
        y_max_adjusted = y_max + 0.05

        for i, attack in enumerate(['low_pass', 'high_pass']):
            ax = ax1 if i == 0 else ax2
            attack_data = class_data[class_data['attack_type'] == attack]
            
            for j, classifier in enumerate(attack_data['classifier'].unique()):
                classifier_data = attack_data[attack_data['classifier'] == classifier]
                classifier_data = classifier_data.sort_values('attack_value')
                x = pd.to_numeric(classifier_data['attack_value'])
                if classifier == 'svc':
                    classifier = 'svm'
                ax.plot(x, classifier_data[score], marker=markers[j], markersize=6, label=classifier, color=palette[j])
            
            ax.set_title(f'{attack.replace("_", " ").title()} Filter', fontsize=14)
            ax.set_xlabel('Cutoff Frequency (Hz)', fontsize=12)
            if i == 0:
                ax.set_ylabel(f'{score.upper()} Score', fontsize=12)
            ax.legend(title='Classifier', title_fontsize='12', fontsize='10')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(y_min_adjusted, y_max_adjusted)

        plt.suptitle(f'{class_name}: {score.upper()} Score vs Low Pass and High Pass Filters', fontsize=16)
        plt.tight_layout()
        if save:
            fig_path = os.path.join(save_dir, f'{class_name}_low_high_pass_{score}.pdf')
            plt.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

        # Plot other attack types (unchanged)
        for attack_type in ['noise', 'decrease_sr', 'dc_drift']:
            plt.figure(figsize=(10, 6))
            attack_data = class_data[class_data['attack_type'] == attack_type]
            
            for j, classifier in enumerate(attack_data['classifier'].unique()):
                classifier_data = attack_data[attack_data['classifier'] == classifier]
                classifier_data = classifier_data.sort_values('attack_value')
                x = pd.to_numeric(classifier_data['attack_value'], errors='coerce')
                if classifier == 'svc':
                    classifier = 'svm'
                plt.plot(x, classifier_data[score], marker=markers[j], markersize=6, label=classifier, color=palette[j])
            
            plt.title(f'{class_name}: {score.upper()} Score vs {attack_type.replace("_", " ").title()}', fontsize=16)
            
            if attack_type == 'dc_drift':
                plt.xscale('log')
                plt.xlabel('DC Drift Value (log scale)', fontsize=12)
                plt.gca().xaxis.set_major_locator(LogLocator(base=10))
                plt.gca().xaxis.set_minor_locator(LogLocator(base=10, subs='all'))
            elif attack_type == 'decrease_sr':
                plt.xlabel('Sample Rate (Hz)', fontsize=12)
            else:
                plt.xlabel(f'{attack_type.replace("_", " ").title()} Value', fontsize=12)
            
            plt.ylabel(f'{score.upper()} Score', fontsize=12)
            plt.legend(title='Classifier', title_fontsize='12', fontsize='10')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            if save:
                fig_path = os.path.join(save_dir, f'{class_name}_{attack_type}_{score}.pdf')
                plt.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

if __name__ == "__main__":
    results = load_results('evaluation_results_hierarchical.pkl')
    data = prepare_data(results)
    generate_latex_table(results)
    plot_scores(data, 'f1', save=True, save_dir='figures/robustness') 
    plot_attacks_by_class(data, 'f1', save=True, save_dir='figures/robustness')