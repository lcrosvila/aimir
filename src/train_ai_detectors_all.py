import os
import numpy as np
import argparse
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from data_utils import get_split
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

class DNNClassifier:
    def __init__(self, input_dim=1024):
        self.model = None
        self.input_dim = input_dim
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
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.cpu().numpy()
    
    def get_params(self, deep=True):
        return {'input_dim': self.input_dim}

def get_classifier(classifier_type, input_dim=400):
    if classifier_type == 'svc':
        classifier = SVC(probability=True, random_state=42)
    elif classifier_type == 'rf':
        classifier = RandomForestClassifier(random_state=42)
    elif classifier_type == 'dnn':
        classifier = DNNClassifier(input_dim=input_dim)
    else:
        raise ValueError("Invalid classifier type. Choose from 'svc', 'rf' or 'dnn'.")
    return classifier

def cross_validate_classifier(classifier, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = -np.inf
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Starting fold {fold + 1}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale data for this fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Check and compare scaling means and stds
        print(f"Fold {fold + 1} scaling mean: {scaler.mean_.mean():.5f}, var: {scaler.var_.mean():.5f}")

        # Fit the classifier
        classifier.fit(X_train, y_train)

        # Evaluate the classifier on the validation set
        val_preds = classifier.predict(X_val)
        val_probas = classifier.predict_proba(X_val)

        if val_probas.shape[1] == 1:  # If only one probability column
            val_probas = val_probas[:, 0]  # Take the single column (probability of class 1)
        else:
            val_probas = val_probas[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_val, val_preds),
            'precision': precision_score(y_val, val_preds),
            'recall': recall_score(y_val, val_preds),
            'f1': f1_score(y_val, val_preds),
            'roc_auc': roc_auc_score(y_val, val_probas)
        }
        all_metrics.append(metrics)

        # Save the best model based on f1 score
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = classifier

    # Compute average and var metrics across all folds
    avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
    var_metrics = {metric: np.var([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
    print("Metrics (avg+-var):")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.3f}+-{var_metrics[metric]:.3f}")

    return best_model, avg_metrics, var_metrics


def save_metrics(metrics_avg, metrics_val, parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)
    filename = f"metrics_{embedding_type}_{ai_folders_string}_{classifier_type}.txt"
    filepath = os.path.join('/home/laura/aimir/results', filename)
    
    if not os.path.exists('/home/laura/aimir/results'):
        os.makedirs('/home/laura/aimir/results')
        
    with open(filepath, 'w') as file:
        for metric, value in metrics_avg.items():
            file.write(f"{metric}: {value:.3f} +- {metrics_val[metric]:.3f}\n")
    print("Metrics saved successfully.")

def save_classifier(classifier, parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)
    filename = f"model_{embedding_type}_{ai_folders_string}_{classifier_type}.pkl"
    
    if not os.path.exists('/home/laura/aimir/models'):
        os.makedirs('/home/laura/aimir/models')
        
    filepath = os.path.join('/home/laura/aimir/models', filename)
    joblib.dump(classifier, filepath)
    print("Classifier saved successfully.")
    print(f"Path: {filepath}")

def main(args):
    X_train, y_train = get_split('train', args.embedding_type, args.real_folder, args.ai_folders)
    X_val, y_val = get_split('val', args.embedding_type, args.real_folder, args.ai_folders)
    X_test, y_test = get_split('test', args.embedding_type, args.real_folder, args.ai_folders)

    # Combine training and validation data
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    # Choose classifier
    classifier = get_classifier(args.classifier_type, input_dim=X.shape[1])

    # Cross-validate classifier and find the best model
    best_model, avg_metrics, var_metrics = cross_validate_classifier(classifier, X, y)
    
    # Save metrics
    save_metrics(avg_metrics, var_metrics, (args.embedding_type, args.ai_folders, args.classifier_type))

    # Save the best classifier
    if args.save_classifier:
        save_classifier(best_model, (args.embedding_type, args.ai_folders, args.classifier_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a classifier.')
    parser.add_argument('--ai-folders', nargs='+', help='Folders containing AI-generated content (suno, boomy or udio)')
    parser.add_argument('--embedding-type', choices=['clap-laion-music', 'musicnn'], default='clap-laion-music', help='Type of embedding')
    parser.add_argument('--classifier-type', choices=['svc', 'rf', 'dnn'], default='svc', help='Type of classifier')
    parser.add_argument('--real-folder', default='lastfm', help='Folder containing real content')
    parser.add_argument('--save-classifier', action='store_true', help='Save the trained classifier')
    parser.add_argument('--all', action='store_true', help='Run all possible combinations of parameters')
    args = parser.parse_args()

    if args.all:
        for classifier_type in ['svc', 'rf', 'dnn']:
            args.ai_folders = ['suno', 'udio']
            args.classifier_type = classifier_type
            main(args)
    else:
        main(args)
