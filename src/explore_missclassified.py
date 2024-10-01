import os
import joblib
import numpy as np
from data_utils import get_split, scale
import argparse
from train_ai_detector import DNNClassifier

def load_classifier(parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)
    filename = f"model_{embedding_type}_{ai_folders_string}_{classifier_type}.pkl"
    filepath = os.path.join('/home/laura/aimir/models', filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    classifier = joblib.load(filepath)
    return classifier

def check_misclassifications(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    # round the predictions to 0 or 1
    y_pred = np.round(y_pred)
    misclassified_indices = np.where(y_pred[0] != y_test)[0]
    return misclassified_indices

def get_mp3_paths(indices, real_files, ai_files):
    paths = []
    for i in indices:
        if i < len(real_files):
            paths.append(real_files[i])
        else:
            paths.append(ai_files[i - len(real_files)])
    return paths

def main(args):
    # Load the embeddings and corresponding labels
    # X_train, y_train = get_split('train', args.embedding_type, args.real_folder, args.ai_folders)
    # X_val, y_val = get_split('val', args.embedding_type, args.real_folder, args.ai_folders)
    X_test, y_test = get_split('test', args.embedding_type, args.real_folder, args.ai_folders)

    # Get the paths to the original files (without embeddings path)
    real_files = []
    with open(f'/home/laura/aimir/{args.real_folder}/test.txt', 'r') as f:
        real_files = f.read().splitlines()
    real_files = [f'/home/laura/aimir/{args.real_folder}/audio/{file}.mp3' for file in real_files]

    ai_files = []
    for folder in args.ai_folders:
        ai_files_aux = []
        with open(f'/home/laura/aimir/{folder}/test.txt', 'r') as f:
            ai_files_aux = f.read().splitlines()
        ai_files_aux = [f'/home/laura/aimir/{folder}/audio/{file}.mp3' for file in ai_files_aux]
        ai_files += ai_files_aux

    # Scale the data
    _, _, X_test = scale(np.array([]), np.array([]), X_test, args.embedding_type, load_scaler=True, save_scaler=False)

    results = {}

    for classifier_type in ['svc', 'rf', 'dnn']:
        classifier = load_classifier((args.embedding_type, args.ai_folders, classifier_type))
        misclassified_indices = check_misclassifications(classifier, X_test, y_test)
        
        mp3_paths = get_mp3_paths(misclassified_indices, real_files, ai_files)
        results[classifier_type] = mp3_paths

    # Save the results
    with open(f'/home/laura/aimir/results/misclassified_{args.embedding_type}_{"_".join(args.ai_folders)}.txt', 'w') as f:
        for classifier_type, paths in results.items():
            f.write(f"{classifier_type} misclassified:\n")
            for path in paths:
                f.write(f"{path}\n")
            f.write("\n")
    
    print("Results saved successfully.")

    # print for each file, which classifier misclassified it
    all_misclassfied = set(results['svc']) | set(results['rf']) | set(results['dnn'])
    for path in all_misclassfied:
        print(path)
        for classifier_type in results.keys():
            if path in results[classifier_type]:
                print(f"{classifier_type} misclassified")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check misclassifications in the test set.')
    parser.add_argument('--ai-folders', nargs='+', help='Folders containing AI-generated content (suno, boomy or udio)')
    parser.add_argument('--embedding-type', choices=['clap-laion-music', 'musicnn'], default='clap-laion-music', help='Type of embedding')
    parser.add_argument('--real-folder', default='lastfm', help='Folder containing real content')
    args = parser.parse_args()
    
    main(args)
