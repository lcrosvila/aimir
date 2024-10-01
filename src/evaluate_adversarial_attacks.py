import os
import numpy as np
import joblib
import torch
import argparse
import librosa
import soundfile as sf
from data_utils import get_split, scale, get_split_mp3
from data_utils import get_embedding_clap, get_embedding_musicnn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.signal import butter, lfilter
from train_ai_detector import DNNClassifier

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_low_pass_filter(audio, sr, audio_path, cutoff=5000, order=5):
    # the save file is in the same directory as the audio file, in a folder called 'low_pass_{cutoff}', and the filename+'.npy'
    save_file = os.path.join(os.path.dirname(audio_path), f'low_pass_{cutoff}', os.path.basename(audio_path).replace('.mp3', '.npy'))
    if os.path.exists(save_file):
        return np.load(save_file)
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, audio)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, y)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_high_pass_filter(audio, sr, audio_path, cutoff=5000, order=5):
    save_file = os.path.join(os.path.dirname(audio_path), f'high_pass_{cutoff}', os.path.basename(audio_path).replace('.mp3', '.npy'))
    if os.path.exists(save_file):
        return np.load(save_file)
    b, a = butter_highpass(cutoff, sr, order=order)
    y = lfilter(b, a, audio)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, y)
    return y

def add_noise(audio, audio_path, noise_factor=0.005):
    save_file = os.path.join(
        os.path.dirname(audio_path), 
        f'noise_{str(noise_factor).replace(".", "_")}', 
        os.path.basename(audio_path).replace('.mp3', '.npy')
    )
    if os.path.exists(save_file):
        return np.load(save_file)
    noise = np.random.randn(len(audio))
    y = audio + noise_factor * noise
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, y)
    return y

def decrease_sample_rate(audio, sr, audio_path, target_sr=8000):
    save_file = os.path.join(os.path.dirname(audio_path), f'decrease_sr_{target_sr}', os.path.basename(audio_path).replace('.mp3', '.npy'))
    if os.path.exists(save_file):
        return np.load(save_file)
    y = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    np.save(save_file, y)
    return y

def apply_adversarial_attacks(audio, sr, audio_path):
    attacks = {
        'low_pass_5000': apply_low_pass_filter(audio, sr, audio_path, cutoff=5000),
        'low_pass_8000': apply_low_pass_filter(audio, sr, audio_path, cutoff=8000),
        'low_pass_10000': apply_low_pass_filter(audio, sr, audio_path, cutoff=10000),
        'low_pass_16000': apply_low_pass_filter(audio, sr, audio_path, cutoff=16000),
        'high_pass_12000': apply_high_pass_filter(audio, sr, audio_path, cutoff=12000),
        'high_pass_8000': apply_high_pass_filter(audio, sr, audio_path, cutoff=8000),
        'noise_0.005': add_noise(audio, audio_path, noise_factor=0.005),
        'noise_0.01': add_noise(audio, audio_path, noise_factor=0.01),
        'decrease_sr_8000': decrease_sample_rate(audio, sr, audio_path, target_sr=8000),
        'decrease_sr_16000': decrease_sample_rate(audio, sr, audio_path, target_sr=16000),
        'decrease_sr_22050': decrease_sample_rate(audio, sr, audio_path, target_sr=22050),
        'decrease_sr_24000': decrease_sample_rate(audio, sr, audio_path, target_sr=24000)
    }
    return attacks

def extract_embeddings(audio, sr, embedding_type, model=None):
    if embedding_type == 'clap-laion-music':
        return get_embedding_clap(audio, sr, model)
    elif embedding_type == 'musicnn':
        emb = get_embedding_musicnn(audio, sr, model)
        if isinstance(emb, list):
            emb = emb[0]
            return np.array([np.concatenate((np.mean(emb, axis=0), np.var(emb, axis=0)))])
        return np.concatenate((np.mean(emb, axis=0), np.var(emb, axis=0)))
    else:
        raise ValueError("Invalid embedding type. Choose 'clap-laion-music' or 'musicnn'.")

def evaluate_classifier(classifier, X, y):
    y_pred = classifier.predict(X)
    
    # Handle cases where predict_proba returns only one column
    y_proba = classifier.predict_proba(X)
    if y_proba.shape[1] == 1:  # If only one probability column
        y_proba = y_proba[:, 0]  # Take the single column (probability of class 1)
    else:
        y_proba = y_proba[:, 1]  # Take the probability of the positive class (class 1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_proba)
    }
    return metrics

def main(args):
    # Load the sample dataset
    X_sample, y_sample = get_split_mp3('test', args.embedding_type, args.real_folder, args.ai_folders)
    # X_sample = X_sample[:2] + X_sample[-2:]
    # y_sample = y_sample[:2] + y_sample[-2:]

    if args.embedding_type == 'clap-laion-music':
        from model_loader import CLAPMusic
        model = CLAPMusic()
        model.load_model()
    elif args.embedding_type == 'musicnn':
        from model_loader import MusiCNN
        model = MusiCNN()
        model.load_model()

    # Path to pre-trained classifiers
    classifiers = {}
    for classifier_type in ['svc', 'rf', 'dnn']:
        filename = f"model_{args.embedding_type}_{'_'.join(args.ai_folders)}_{classifier_type}.pkl"
        filepath = os.path.join('/home/laura/aimir/models', filename)
        classifiers[classifier_type] = joblib.load(filepath)

    # Iterate through the sample data and apply adversarial attacks
    results = {}
    adv_embeddings_scaled = {}
    labels = {}

    for i, (audio_path, label) in enumerate(zip(X_sample, y_sample)):
        print(audio_path)
        audio, sr = librosa.load(audio_path, sr=None)
        adversarial_examples = apply_adversarial_attacks(audio, sr, audio_path)

        for attack_name, adv_audio in adversarial_examples.items():
            adv_embedding = extract_embeddings(adv_audio, sr, args.embedding_type, model)

            for classifier_type, classifier in classifiers.items():
                adv_embedding_scaled, _, _ = scale(adv_embedding, None, None, args.embedding_type, load_scaler=True, save_scaler=False)
                if attack_name not in adv_embeddings_scaled:
                    adv_embeddings_scaled[attack_name] = {}
                if classifier_type not in adv_embeddings_scaled[attack_name]:
                    adv_embeddings_scaled[attack_name][classifier_type] = []
                adv_embeddings_scaled[attack_name][classifier_type].append(adv_embedding_scaled)

            if attack_name not in labels:
                labels[attack_name] = []
            labels[attack_name].append(label)

    # Evaluate the classifiers with the adversarial examples
    for attack_name, adv_embeddings in adv_embeddings_scaled.items():
        results[attack_name] = {}
        for classifier_type, adv_embeddings_list in adv_embeddings.items():
            X = np.vstack(adv_embeddings_list)
            y = np.array(labels[attack_name])
            results[attack_name][classifier_type] = evaluate_classifier(classifiers[classifier_type], X, y)
        
    # open a text file to save the results
    with open(f'/home/laura/aimir/results/adv_attacks_{args.embedding_type}_{"_".join(args.ai_folders)}.txt', 'w') as f:
        for attack_name, attack_results in results.items():
            f.write(f"Attack: {attack_name}\n")
            for classifier_type, metrics in attack_results.items():
                f.write(f"Classifier: {classifier_type}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")

    for attack_name, attack_results in results.items():
        print(f"Attack: {attack_name}")
        for classifier_type, metrics in attack_results.items():
            print(f"Classifier: {classifier_type}")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pre-trained classifiers with adversarial attacks.')
    parser.add_argument('--ai-folders', nargs='+', help='Folders containing AI-generated content (suno, boomy or udio)')
    parser.add_argument('--embedding-type', choices=['clap-laion-music', 'musicnn'], default='clap-laion-music', help='Type of embedding')
    parser.add_argument('--real-folder', default='lastfm', help='Folder containing real content')
    args = parser.parse_args()
    main(args)
