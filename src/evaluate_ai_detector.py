# /home/laura/aimir/embeddings_env/bin/python /home/laura/aimir/src/evaluate_ai_detector.py --ai-folders suno udio --embedding-type musicnn --classifier-type svc

import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_utils import get_split, scale
from sklearn.metrics import roc_auc_score
import torch

class DNNClassifier:
    def __init__(self, input_dim=1024):
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        for epoch in range(100):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return (y_pred.cpu().numpy() > 0.5).astype(int)

def train_classifier(X_train, y_train, classifier_type, input_dim=400):
    if classifier_type == 'svc':
        classifier = SVC(probability=True, random_state=42)
    elif classifier_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
    elif classifier_type == 'dnn':
        classifier = DNNClassifier(input_dim=input_dim)
    else:
        raise ValueError("Invalid classifier type. Choose from 'svc', 'rf' or 'dnn'.")
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_classifier(classifier, X_val, y_val):
    val_predictions = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    precision = precision_score(y_val, val_predictions)
    recall = recall_score(y_val, val_predictions)
    f1 = f1_score(y_val, val_predictions)
    # ROC auc
    roc_auc = roc_auc_score(y_val, val_predictions)
    return accuracy, precision, recall, f1, roc_auc

def save_classifier(classifier, parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)  # Joining AI folder names with "_"
    filename = f"model_{embedding_type}_{ai_folders_string}_{classifier_type}.pkl"
    # if folder doesn't exist, create it
    if not os.path.exists('/home/laura/aimir/models'):
        os.makedirs('/home/laura/aimir/models')
    filepath = os.path.join('/home/laura/aimir/models', filename)
    joblib.dump(classifier, filepath)
    print("Classifier saved successfully.")

def load_classifier(parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)  # Joining AI folder names with "_"
    filename = f"model_{embedding_type}_{ai_folders_string}_{classifier_type}.pkl"
    filepath = os.path.join('/home/laura/aimir/models', filename)
    return joblib.load(filepath)

def main(args):
    # X_train, y_train = get_split('train', args.embedding_type, args.real_folder, args.ai_folders)
    # X_val, y_val = get_split('val', args.embedding_type, args.real_folder, args.ai_folders)
    X_test, y_test = get_split('test', args.embedding_type, args.real_folder, args.ai_folders)

    # X_train, y_train = get_split('sample', args.embedding_type, args.real_folder, args.ai_folders)
    # X_val, y_val = get_split('sample', args.embedding_type, args.real_folder, args.ai_folders)
    # X_test, y_test = get_split('sample', args.embedding_type, args.real_folder, args.ai_folders)

    X_train, X_val, X_test = scale([], [], X_test, args.embedding_type, load_scaler=True, save_scaler=False)

    # Load classifier
    classifier = load_classifier((args.embedding_type, args.ai_folders, args.classifier_type))    

    # Evaluate on validation set
    accuracy, precision, recall, f1, roc_auc = evaluate_classifier(classifier, X_test, y_test)

    print(args.embedding_type, args.classifier_type)
    print("Test Accuracy:", round(accuracy, 3))
    print("Test Precision:", round(precision, 3))
    print("Test Recall:", round(recall, 3))
    print("Test F1 Score:", round(f1, 3))
    print("Test ROC AUC:", round(roc_auc, 3))

    # save predictions
    predictions = classifier.predict(X_test)
    # if folder doesn't exist, create it
    if not os.path.exists('/home/laura/aimir/predictions'):
        os.makedirs('/home/laura/aimir/predictions')
    np.save(f'/home/laura/aimir/predictions/{args.embedding_type}_{args.classifier_type}_predictions.npy', predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a classifier.')
    parser.add_argument('--ai-folders', nargs='+', help='Folders containing AI-generated content (suno, boomy or udio)')
    parser.add_argument('--embedding-type', choices=['clap-laion-music', 'musicnn'], default='clap-laion-music', help='Type of embedding')
    parser.add_argument('--classifier-type', choices=['svc', 'rf', 'dnn'], default='svc', help='Type of classifier')
    parser.add_argument('--real-folder', default='lastfm', help='Folder containing real content')
    args = parser.parse_args()
    main(args)
