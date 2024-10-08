import os
import numpy as np
import argparse
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from data_utils import load_embeddings
from torch.utils.data import DataLoader, TensorDataset

class DNNClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=3):
        super(DNNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y, epochs=100, batch_size=32):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self(X)
            probas = torch.softmax(outputs, dim=1)
        return probas.cpu().numpy()

def get_classifier(classifier_type, input_dim=400, num_classes=3):
    if classifier_type == 'svc':
        return SVC(probability=True, random_state=42)
    elif classifier_type == 'rf':
        return RandomForestClassifier(random_state=42)
    elif classifier_type == 'dnn':
        return DNNClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError("Invalid classifier type. Choose from 'svc', 'rf' or 'dnn'.")

def cross_validate_classifier(classifier, X, y, num_classes):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = -np.inf
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Starting fold {fold + 1}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if isinstance(classifier, DNNClassifier):
            classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train)

        val_preds = classifier.predict(X_val)
        val_probas = classifier.predict_proba(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, val_preds),
            'precision': precision_score(y_val, val_preds, average='macro'),
            'recall': recall_score(y_val, val_preds, average='macro'),
            'f1': f1_score(y_val, val_preds, average='macro'),
            'roc_auc': roc_auc_score(y_val, val_probas, multi_class='ovr', average='macro')
        }
        all_metrics.append(metrics)

        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = classifier

    avg_metrics = {metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
    var_metrics = {metric: np.var([m[metric] for m in all_metrics]) for metric in all_metrics[0]}
    
    print("Metrics (avg±var):")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.3f}±{var_metrics[metric]:.3f}")

    return best_model, avg_metrics, var_metrics


def get_split(split, embedding, folders):
    files = {}
    y = []
    for folder in folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            files[folder] = f.read().splitlines()
            y += [folder]*len(files[folder])
    
    files = [f'/home/laura/aimir/{folder}/audio/embeddings/{embedding}/{file}.npy' for folder in folders for file in files[folder]]

    X, _ = load_embeddings([], files)
    # turn labels into integers
    y = np.array([folders.index(label) for label in y])
    return X, y

def save_metrics(metrics_avg, metrics_val, parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)
    filename = f"metrics_{embedding_type}_{ai_folders_string}_{classifier_type}.txt"
    filepath = os.path.join('/home/laura/aimir/results/three_class', filename)
    
    if not os.path.exists('/home/laura/aimir/results/three_class'):
        os.makedirs('/home/laura/aimir/results/three_class')
        
    with open(filepath, 'w') as file:
        for metric, value in metrics_avg.items():
            file.write(f"{metric}: {value:.3f} +- {metrics_val[metric]:.3f}\n")
    print("Metrics saved successfully.")

def save_classifier(classifier, parameters):
    embedding_type, ai_folders, classifier_type = parameters
    ai_folders_string = "_".join(ai_folders)
    filename = f"model_{embedding_type}_{ai_folders_string}_{classifier_type}.pkl"
    
    if not os.path.exists('/home/laura/aimir/models/three_class'):
        os.makedirs('/home/laura/aimir/models/three_class')
        
    filepath = os.path.join('/home/laura/aimir/models/three_class', filename)
    joblib.dump(classifier, filepath)
    print("Classifier saved successfully.")
    print(f"Path: {filepath}")

def main(args):
    print('Loading data...')
    all_folders = args.ai_folders + [args.real_folder]
    X_train, y_train = get_split('train', args.embedding_type, all_folders)
    X_val, y_val = get_split('val', args.embedding_type,  all_folders)
    X_test, y_test = get_split('test', args.embedding_type, all_folders)
    
    # X, y = get_split('sample', args.embedding_type, all_folders)


    # Combine training and validation data
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)

    print('Training data shape:', X.shape)
    print('Training labels shape:', y.shape)

    num_classes = len(np.unique(y))
    print(f'Number of classes: {num_classes}')

    if args.all:
        for classifier_type in ['svc', 'rf', 'dnn']:
            args.classifier_type = classifier_type
            print(f"Training classifier with {args.embedding_type} embeddings and {args.ai_folders} data using {args.classifier_type} classifier.")
            classifier = get_classifier(args.classifier_type, input_dim=X.shape[1], num_classes=num_classes)
            best_model, avg_metrics, var_metrics = cross_validate_classifier(classifier, X, y, num_classes)
            
            if args.save_classifier:
                save_classifier(best_model, (args.embedding_type, args.ai_folders, args.classifier_type))
                save_metrics(avg_metrics, var_metrics, (args.embedding_type, args.ai_folders, args.classifier_type))
    else:
        print(f"Training classifier with {args.embedding_type} embeddings and {args.ai_folders} data using {args.classifier_type} classifier.")
        classifier = get_classifier(args.classifier_type, input_dim=X.shape[1], num_classes=num_classes)
        best_model, avg_metrics, var_metrics = cross_validate_classifier(classifier, X, y, num_classes)
        
        if args.save_classifier:
            save_classifier(best_model, (args.embedding_type, args.ai_folders, args.classifier_type))
            save_metrics(avg_metrics, var_metrics, (args.embedding_type, args.ai_folders, args.classifier_type))

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
        args.ai_folders = ['suno', 'udio']

    main(args)

