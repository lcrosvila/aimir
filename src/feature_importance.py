import os
import joblib
import numpy as np
import torch
from data_utils import get_split, scale

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from torch.utils.data import DataLoader, TensorDataset

class DNNClassifier:
    # Assuming DNNClassifier is the same as in train_ai_detector.py
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

def load_models():
    models = []
    embedding_types = ['clap-laion-music', 'musicnn']
    classifier_types = ['svc', 'rf', 'dnn']
    ai_folders = ['suno_udio']

    for embedding_type in embedding_types:
        for classifier_type in classifier_types:
            filename = f"model_{embedding_type}_{ai_folders[0]}_{classifier_type}.pkl"
            filepath = os.path.join('/home/laura/aimir/models', filename)
            if os.path.exists(filepath):
                if classifier_type == 'dnn':
                    input_dim = 1024 if embedding_type == 'clap-laion-music' else 400
                    model = DNNClassifier(input_dim=input_dim)
                    model.model.load_state_dict(torch.load(filepath))
                else:
                    model = joblib.load(filepath)
                models.append((embedding_type, classifier_type, model))
    return models

def get_feature_importance_svc(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return result.importances_mean

def get_feature_importance_rf(model):
    return model.feature_importances_

def get_feature_importance_dnn(model, X, y):
    # Using permutation importance for DNN
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc')
    return result.importances_mean

def main():
    models = load_models()
    X, y = get_split('test', 'clap-laion-music', 'lastfm', ['suno', 'udio'])  # Load test data
    X, _, _ = scale(X, X, X, 'clap-laion-music', load_scaler=True, save_scaler=False)

    feature_importances = {}
    for embedding_type, classifier_type, model in models:
        if classifier_type == 'svc':
            importances = get_feature_importance_svc(model, X, y)
        elif classifier_type == 'rf':
            importances = get_feature_importance_rf(model)
        elif classifier_type == 'dnn':
            importances = get_feature_importance_dnn(model, X, y)
        
        key = f"{embedding_type}_{classifier_type}"
        feature_importances[key] = importances

    for key, importances in feature_importances.items():
        print(f"Feature importances for {key}:")
        print(importances)

    # Save feature importances
    joblib.dump(feature_importances, '/home/laura/aimir/models/feature_importances.pkl')

if __name__ == "__main__":
    main()
