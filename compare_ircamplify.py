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


def plot_confusion_matrices(y_true, y_pred_svm_ai, y_pred_rf_ai, y_pred_knn_ai, y_pred_ai, normalize=False):
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Confusion Matrices for Different Classifiers", fontsize=24)

    # List of classifiers and their predictions
    classifiers = [
        ("SVM Classifier", y_pred_svm_ai),
        ("RF Classifier", y_pred_rf_ai),
        ("KNN Classifier", y_pred_knn_ai),
        ("Ircam Amplify Classifier", y_pred_ai)
    ]

    for (title, y_pred), ax in zip(classifiers, axes.flatten()):
        # Create confusion matrix
        cm = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        
        # Remove the 'All' row and column
        cm = cm.iloc[:-1, :-1]
        
        if normalize:
            cm = cm.div(cm.sum(axis=1), axis=0)
            fmt = '.2f'
        else:
            fmt = 'd'

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="YlGnBu", ax=ax, cbar=False, 
                    annot_kws={"size": 24})  # Increase annotation font size
        ax.set_title(title, fontsize=20)  # Increase title font size
        ax.set_ylabel('True Label', fontsize=16)  # Increase y-label font size
        ax.set_xlabel('Predicted Label', fontsize=16)  # Increase x-label font size
        
        # Increase tick label font size
        ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.show()

# %%
# try with boomy
# print(load_ircamplify_results(['boomy']))

# %%
def evaluate_results(data):
    # Evaluate the results
    y_true = data['true_class']
    y_pred_svm_parent = data['svm_pred_parent']
    y_pred_rf_parent = data['rf_pred_parent']
    y_pred_knn_parent = data['knn_pred_parent']
    y_pred_svm_child = data['svm_pred_child']
    y_pred_rf_child = data['rf_pred_child']
    y_pred_knn_child = data['knn_pred_child']
    y_pred_ai = data['is_ai']

    # y_true_ai is Fase if true_class is 'lastfm' and True otherwise
    y_true_ai = np.array([False if label == 'lastfm' else True for label in y_true])

    y_pred_svm_ai = np.array([True if label == 'AI' else False for label in y_pred_svm_parent])
    y_pred_rf_ai = np.array([True if label == 'AI' else False for label in y_pred_rf_parent])
    y_pred_knn_ai = np.array([True if label == 'AI' else False for label in y_pred_knn_parent])

    # print(y_pred_svm_ai)

    print("SVM Classifier:")
    # print(classification_report(y_true_int, y_pred_svm))
    print(classification_report(y_true_ai, y_pred_svm_ai))
    print("RF Classifier:")
    # print(classification_report(y_true_int, y_pred_rf))
    print(classification_report(y_true_ai, y_pred_rf_ai))
    print("KNN Classifier:")
    # print(classification_report(y_true_int, y_pred_knn))
    print(classification_report(y_true_ai, y_pred_knn_ai))
    print("Ircam Amplify Classifier:")
    print(classification_report(y_true_ai, y_pred_ai))

    # print the overal confidence when is_ai is True and False
    print("Overall confidence when is_ai is True:")
    print(data[data['is_ai'] == True]['confidence'].describe())
    print("Overall confidence when is_ai is False:")
    print(data[data['is_ai'] == False]['confidence'].describe())

    # confusion matrix, where true label is 'suno' 'udio' or 'lastfm' and the predicted label is 'ai' or 'not ai'
    print("Confusion matrix for SVM Classifier:")
    print(pd.crosstab(y_true, y_pred_svm_ai, rownames=['True'], colnames=['Predicted'], margins=True))
    print("Confusion matrix for RF Classifier:")
    print(pd.crosstab(y_true, y_pred_rf_ai, rownames=['True'], colnames=['Predicted'], margins=True))
    print("Confusion matrix for KNN Classifier:")
    print(pd.crosstab(y_true, y_pred_knn_ai, rownames=['True'], colnames=['Predicted'], margins=True))
    print("Confusion matrix for Ircam Amplify Classifier:")
    print(pd.crosstab(y_true, y_pred_ai, rownames=['True'], colnames=['Predicted'], margins=True))

    # print the row where ircam amplify predicted not ai when it was ai
    print(data[(data['is_ai'] == False) & (data['true_class'].isin(['suno', 'udio']))])
    print(data[(data['is_ai'] == True) & (data['true_class'] == 'lastfm')])

    plot_confusion_matrices(y_true, y_pred_svm_ai, y_pred_rf_ai, y_pred_knn_ai, y_pred_ai, normalize=False)

if __name__ == "__main__":
    folders = ['suno', 'udio', 'lastfm', 'boomy']
    data = get_results_all(folders)
    # drop some rows so that we have 1000 of 'suno', 1000 of 'udio' and 1000 of 'lastfm'
    suno = data[data['true_class'] == 'suno'].sample(n=1000, random_state=42)
    udio = data[data['true_class'] == 'udio'].sample(n=1000, random_state=42)
    lastfm = data[data['true_class'] == 'lastfm'].sample(n=1000, random_state=42)
    boomy = data[data['true_class'] == 'boomy']
    data = pd.concat([suno, udio, lastfm, boomy])
    # print(data['true_class'].value_counts())
    evaluate_results(data)