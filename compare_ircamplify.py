# %%
import numpy as np
import json
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf

def load_single_embedding(file):
    if file.endswith('.npy'):
        return np.load(file, mmap_mode='r')
    return None

def load_embeddings(files):
    valid_files = [f for f in files if f.endswith('.npy')]
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_embedding, file) for file in valid_files]
        embeddings = [future.result() for future in as_completed(futures) if future.result() is not None]
    
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
    return X, y, files

def load_ircamplify_results(folders):
    true_class = []
    files = []
    is_ai = []
    confidence = []
    for folder in folders:
        folder_path = f'/home/laura/aimir/ircamplify_results/{folder}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                    job_infos = data.get('job_infos', {})
                    file_paths = job_infos.get('file_paths', {})
                    report_info = job_infos.get('report_info', {})
                    report = report_info.get('report', {})
                    result_list = report.get('resultList', [])
                    
                    for i, result in enumerate(result_list):
                        true_class.append(folder)
                        file = file_paths[i].split('/')[-1]
                        files.append(file)
                        is_ai.append(result.get('isAi'))
                        confidence.append(result.get('confidence'))
    # make it into a dataframe
    data = {
        'true_class': true_class,
        'file': files,
        'is_ai': is_ai,
        'confidence': confidence
    }
    data = pd.DataFrame(data)
    return data

def get_classifiers_results(models, X_test_scaled, test_files):
    true_class = []
    files = []
    svm_pred_parent = []
    svm_pred_child = []
    rf_pred_parent = []
    rf_pred_child = []
    knn_pred_parent = []
    knn_pred_child = []

    for i, file in enumerate(test_files):
        true_class.append(file.split('/')[-5])
        files.append(file.split('/')[-1].replace('npy','mp3'))
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        for i, file in enumerate(test_files):
            if name == 'svc':
                svm_pred_parent.append(y_pred[i, 0])
                svm_pred_child.append(y_pred[i, 1])
            elif name == 'rf':
                rf_pred_parent.append(y_pred[i, 0])
                rf_pred_child.append(y_pred[i, 1])
            elif name == 'knn':
                knn_pred_parent.append(y_pred[i, 0])
                knn_pred_child.append(y_pred[i, 1])


    data = {
        'true_class': true_class,
        'file': files,
        'svm_pred_parent': svm_pred_parent,
        'svm_pred_child': svm_pred_child,
        'rf_pred_parent': rf_pred_parent,
        'rf_pred_child': rf_pred_child,
        'knn_pred_parent': knn_pred_parent,
        'knn_pred_child': knn_pred_child
    }
    data = pd.DataFrame(data)
    return data

def get_results_all(folders = ['suno', 'udio', 'lastfm']):
    if not os.path.exists('classifier_test_results.csv'):            
        # Load trained models and scaler
        with open('models_and_scaler.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        models = saved_data['models']
        scaler = saved_data['scaler']

        # Load test data
        if 'boomy' in folders:
            without_boomy = [folder for folder in folders if folder != 'boomy']
            X_test, y_test, test_files = get_split('test', 'clap-laion-music', without_boomy)
            X_boomy, y_boomy, test_files_boomy = get_split('sample', 'clap-laion-music', ['boomy'])
            X_test = np.concatenate((X_test, X_boomy))
            test_files = test_files + test_files_boomy
        else:
            X_test, y_test, test_files = get_split('test', 'clap-laion-music', folders)
            
        X_test_scaled = scaler.transform(X_test)

        # classifier results
        classifiers_results = get_classifiers_results(models, X_test_scaled, test_files)

        # save the classifier results
        classifiers_results.to_csv('classifier_test_results.csv', index=False)
    else:
        classifiers_results = pd.read_csv('classifier_test_results.csv')

    #  how many rows have 'suno', 'udio', 'lastfm' as the true class
    # print(classifiers_results['true_class'].value_counts())
    # ircamplify results
    ircamplify_results = load_ircamplify_results(folders)
    # how many rows have 'suno', 'udio', 'lastfm' as the true class
    # print(ircamplify_results['true_class'].value_counts())

    # merge the two dataframes
    merged_data = pd.merge(classifiers_results, ircamplify_results, on=['true_class', 'file'], how='inner')

    # print(merged_data.head())
    return merged_data


def plot_and_save_confusion_matrices(y_true, y_pred_svm_ai, y_pred_rf_ai, y_pred_knn_ai, y_pred_ai, normalize=True, with_boomy=False, directory='output'):
    # Ensure output directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create a 2x2 subplot for confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    title_suffix = "with Boomy" if with_boomy else "without Boomy"
    # fig.suptitle(f"Confusion Matrices for Different Classifiers ({title_suffix})", fontsize=24)

    classifiers = [
        ("SVM Classifier", y_pred_svm_ai),
        ("RF Classifier", y_pred_rf_ai),
        ("KNN Classifier", y_pred_knn_ai),
        ("Ircam Amplify Classifier", y_pred_ai)
    ]

    for (title, y_pred), ax in zip(classifiers, axes.flatten()):
        cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        cm = cm.iloc[:-1, :-1]
        
        if normalize:
            cm = cm.div(cm.sum(axis=1), axis=0)
            fmt = '.2f'
        else:
            fmt = 'd'

        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 26})  # Font size for numbers
        ax.set_title(f"{title}", fontsize=24)  # Increase title font size
        ax.set_ylabel('True Label', fontsize=20)  # Increase y-label font size
        ax.set_xlabel('Predicted Label', fontsize=20)  # Increase x-label font size
        
        # Increase tick label font size
        ax.tick_params(labelsize=18)

        # print the confusion matrix as latex table
        print(f"{title}:\n{cm.to_latex()}\n")
        

    plt.tight_layout()

    # Save figure
    normalization_status = "normalized" if normalize else "non_normalized"
    filename = f"{directory}/confusion_matrices_{normalization_status}_{title_suffix}.pdf"
    plt.savefig(filename)
    plt.close()


    return filename

# Function to print classification report in LaTeX format
def print_classification_report_latex(data):
    y_true = data['true_class']
    y_pred_svm_parent = data['svm_pred_parent']
    y_pred_rf_parent = data['rf_pred_parent']
    y_pred_knn_parent = data['knn_pred_parent']
    y_pred_ai = data['is_ai']

    # Convert true_class to AI vs non-AI binary classification
    y_true_ai = np.array([False if label == 'lastfm' else True for label in y_true])
    
    y_pred_svm_ai = np.array([True if label == 'AI' else False for label in y_pred_svm_parent])
    y_pred_rf_ai = np.array([True if label == 'AI' else False for label in y_pred_rf_parent])
    y_pred_knn_ai = np.array([True if label == 'AI' else False for label in y_pred_knn_parent])
    
    # Create classification report for each classifier
    svm_report = classification_report(y_true_ai, y_pred_svm_ai, output_dict=True)
    rf_report = classification_report(y_true_ai, y_pred_rf_ai, output_dict=True)
    knn_report = classification_report(y_true_ai, y_pred_knn_ai, output_dict=True)
    ai_report = classification_report(y_true_ai, y_pred_ai, output_dict=True)

    # Combine reports into a LaTeX table
    table = r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
    table += "\n"
    table += r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
    table += "\n"
    classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report), ("Ircam Amp.", ai_report)]

    for classifier, report in classifiers:
        precision = report['True']['precision']
        recall = report['True']['recall']
        f1_score = report['True']['f1-score']
        accuracy = report['accuracy']
        table += f"{classifier} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {accuracy:.3f} \\\\ \hline \n"

    table += r"\end{tabular}\caption{Parent-level classification results (AI vs. non-AI) on the validation set}\end{table}"

    print(table)

    # do the same on the child level (suno udio lastfm)
    y_pred_svm_child = data['svm_pred_child']
    y_pred_rf_child = data['rf_pred_child']
    y_pred_knn_child = data['knn_pred_child']

    # Create classification report for each classifier
    svm_report = classification_report(y_true, y_pred_svm_child, output_dict=True)
    rf_report = classification_report(y_true, y_pred_rf_child, output_dict=True)
    knn_report = classification_report(y_true, y_pred_knn_child, output_dict=True)

    # Combine reports into a LaTeX table
    table = r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
    table += "\n"
    table += r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
    table += "\n"
    classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report)]

    for classifier, report in classifiers:
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']
        accuracy = report['accuracy']
        table += f"{classifier} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} & {accuracy:.3f} \\\\ \hline \n"

    table += r"\end{tabular}\caption{Child-level classification results (LastFM, Suno, Udio) on the validation set}\end{table}"

    print(table)

    # Generate detailed child-level classification report for each classifier
    classifiers = [("SVM", svm_report), ("RF", rf_report), ("KNN", knn_report)]

    # Create the LaTeX table
    detailed_table = r"\begin{table}[ht]\centering\begin{tabular}{llccc}\hline"
    detailed_table += "\n"
    detailed_table += r"Classifier & Category & Precision & Recall & F1-score \\ \hline"
    detailed_table += "\n"

    # Loop through each classifier and its corresponding classification report
    for classifier, report in classifiers:
        detailed_table += f"\\multirow{{3}}{{*}}{{{classifier}}} "

        # Loop through each category (LastFM, Suno, Udio)
        for idx, category in enumerate(['lastfm', 'suno', 'udio']):
            if idx > 0:  # Add a new row for categories other than the first
                detailed_table += " & "
            precision = report[category]['precision']
            recall = report[category]['recall']
            f1_score = report[category]['f1-score']
            detailed_table += f"{category.capitalize()} & {precision:.3f} & {recall:.3f} & {f1_score:.3f} \\\\ \n"
        detailed_table += r"\hline \n"

    detailed_table += r"\end{tabular}\caption{Detailed child-level classification results (LastFM, Suno, Udio) on the validation set}\end{table}"

    # Print the detailed table
    print(detailed_table)

# Main function to evaluate and save results
def evaluate_results_and_save(data, normalize=False, with_boomy=False, directory='output'):
    y_true = data['true_class']
    y_pred_svm_parent = data['svm_pred_parent']
    y_pred_rf_parent = data['rf_pred_parent']
    y_pred_knn_parent = data['knn_pred_parent']
    y_pred_ai = data['is_ai']

    # y_true_ai: False for 'lastfm', True for others
    y_true_ai = np.array([False if label == 'lastfm' else True for label in y_true])

    # Binary predictions (True: AI, False: Not AI)
    y_pred_svm_ai = np.array([True if label == 'AI' else False for label in y_pred_svm_parent])
    y_pred_rf_ai = np.array([True if label == 'AI' else False for label in y_pred_rf_parent])
    y_pred_knn_ai = np.array([True if label == 'AI' else False for label in y_pred_knn_parent])

    # Plot and save confusion matrices
    normalized_filename = plot_and_save_confusion_matrices(
        y_true, y_pred_svm_ai, y_pred_rf_ai, y_pred_knn_ai, y_pred_ai, normalize=True, with_boomy=with_boomy, directory=directory
    )
    non_normalized_filename = plot_and_save_confusion_matrices(
        y_true, y_pred_svm_ai, y_pred_rf_ai, y_pred_knn_ai, y_pred_ai, normalize=False, with_boomy=with_boomy, directory=directory
    )

    print(f"Saved normalized confusion matrices to: {normalized_filename}")
    print(f"Saved non-normalized confusion matrices to: {non_normalized_filename}")

    # Print LaTeX table for classification report
    print_classification_report_latex(data)

# Example usage
if __name__ == "__main__":
    with_boomy = False
    output_dir = '/home/laura/aimir/figures/confusion_matrices' 

    if with_boomy:
        folders = ['suno', 'udio', 'lastfm', 'boomy']
    else:
        folders = ['suno', 'udio', 'lastfm']

    data = get_results_all(folders)
    suno = data[data['true_class'] == 'suno'].sample(n=1000, random_state=42)
    udio = data[data['true_class'] == 'udio'].sample(n=1000, random_state=42)
    lastfm = data[data['true_class'] == 'lastfm'].sample(n=1000, random_state=42)

    if with_boomy:
        boomy = data[data['true_class'] == 'boomy']
        data = pd.concat([suno, udio, lastfm, boomy])
    else:
        data = pd.concat([suno, udio, lastfm])

    # Evaluate results and save figures and reports
    evaluate_results_and_save(data, normalize=True, with_boomy=with_boomy, directory=output_dir)