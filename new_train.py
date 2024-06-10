# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from captum.attr import Saliency
import matplotlib.pyplot as plt
import seaborn as sns

# %%
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_embeddings(real_files, ai_files):
    X = []
    y = []

    for file in ai_files:
        if file.endswith('.npy'):
            embedding = np.load(file)
            if len(embedding.shape) == 2:
                embedding = np.concatenate((np.mean(embedding, axis=0), np.var(embedding, axis=0)))
            X.append(embedding)
            y.append(1)
    
    for file in real_files:
        if file.endswith('.npy'):
            embedding = np.load(file)
            if len(embedding.shape) == 2:
                embedding = np.concatenate((np.mean(embedding, axis=0), np.var(embedding, axis=0))) 
            X.append(embedding)
            y.append(0)

    return np.array(X), np.array(y)

def get_split(split, embedding, real_folder, ai_folders):
    ai_files = {}
    with open(f'/home/laura/aimir/{real_folder}/{split}.txt', 'r') as f:
        real_files = f.read().splitlines()
    for folder in ai_folders:
        with open(f'/home/laura/aimir/{folder}/{split}.txt', 'r') as f:
            ai_files[folder] = f.read().splitlines()
    
    real_files = [f'/home/laura/aimir/{real_folder}/audio/embeddings/{embedding}/{file}.npy' for file in real_files]
    ai_files = [f'/home/laura/aimir/{folder}/audio/embeddings/{embedding}/{file}.npy' for folder in ai_folders for file in ai_files[folder]]

    X, y = load_embeddings(real_files, ai_files)
    return X, y

def train_classifier(X_train, y_train, input_dim):
    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(10):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def evaluate_classifier(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_val, dtype=torch.float32)).squeeze()
        predictions = (outputs > 0.5).int().numpy()
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    return accuracy, precision, recall, f1

def plot_attributions_heatmap(attributions, true_label, num_samples):
    plt.figure(figsize=(12, 8))
    heatmap_data = attributions[:num_samples].cpu().numpy()
    sns.heatmap(heatmap_data, cmap='viridis', xticklabels=False, yticklabels=[f'Sample {i+1} (label {true_label[i]})' for i in range(num_samples)])
    plt.title('Saliency Map Attributions Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Sample Index')
    plt.show()

def main():
    real_folder = 'lastfm'
    ai_folders = ['boomy', 'suno', 'udio']
    embedding = 'clap-laion-music'
    
    X_train, y_train = get_split('sample', embedding, real_folder, ai_folders)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    input_dim = X_train.shape[1]
    model = train_classifier(X_train, y_train, input_dim)
    
    accuracy, precision, recall, f1 = evaluate_classifier(model, X_train, y_train)

    print(f'Train Accuracy: {accuracy}')
    print(f'Train Precision: {precision}')
    print(f'Train Recall: {recall}')
    print(f'Train F1 Score: {f1}')

    y_pred = (model(torch.tensor(X_train, dtype=torch.float32)).squeeze() > 0.5).int().numpy()
    print(confusion_matrix(y_train, y_pred))

    saliency = Saliency(model)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    attributions = saliency.attribute(X_train_tensor, target=0)

    plot_attributions_heatmap(attributions, true_label=y_train, num_samples=10)

if __name__ == "__main__":
    main()
