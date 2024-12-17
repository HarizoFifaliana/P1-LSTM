import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from nltk.tokenize import word_tokenize

# Téléchargement des bibliothèques supplémentaires si nécessaire
# pip install pandas scikit-learn nltk torch

# Étape 1 : Préparation des données
class TextDataset(Dataset):
    def __init__(self, texts, targets, word2idx, max_length):
        self.texts = [torch.tensor([word2idx.get(word, word2idx["<UNK>"]) for word in text], dtype=torch.long) for text in texts]
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.max_length = max_length

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]

def preprocess_data(csv_file, max_vocab_size=10000):
    """
    Charge les données CSV, tokenise les textes et encode les cibles.
    """
    df = pd.read_csv(csv_file)
    if 'target' not in df.columns or 'text' not in df.columns:
        raise ValueError("Le fichier CSV doit contenir les colonnes 'target' et 'text'.")

    # Nettoyer les données
    df = df.dropna(subset=['text', 'target'])

    # Tokenisation
    texts = [word_tokenize(text.lower()) for text in df['text']]
    targets = LabelEncoder().fit_transform(df['target'])

    # Construction du vocabulaire
    counter = Counter(word for text in texts for word in text)
    vocab = [word for word, _ in counter.most_common(max_vocab_size - 2)]
    word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    return texts, targets, word2idx

# Étape 2 : Création du modèle LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(self.dropout(hidden[-1]))
        return self.activation(out)

# Étape 3 : Fonction principale pour exécuter le script
def main():
    # Paramètres
    csv_file = "bodo_poopy.csv"  # Votre fichier d'entrée
    max_vocab_size = 10000
    max_length = 100
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 1  # Classification binaire
    batch_size = 32
    epochs = 5
    lr = 0.001

    print("1. Préparation des données...")
    texts, targets, word2idx = preprocess_data(csv_file, max_vocab_size)
    pad_idx = word2idx["<PAD>"]

    # Préparation des datasets
    train_texts, test_texts, train_targets, test_targets = train_test_split(texts, targets, test_size=0.2, random_state=42)
    train_dataset = TextDataset(train_texts, train_targets, word2idx, max_length)
    test_dataset = TextDataset(test_texts, test_targets, word2idx, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_idx))

    print("2. Initialisation du modèle...")
    model = LSTMClassifier(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           output_dim=output_dim, pad_idx=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary Cross Entropy pour classification binaire

    # Envoi du modèle sur GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    print("3. Entraînement du modèle...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        for texts, targets in train_loader:
            texts, targets = texts.to(device), targets.float().to(device)

            optimizer.zero_grad()
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f}")

    print("4. Évaluation sur l'ensemble de test...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, targets in test_loader:
            texts, targets = texts.to(device), targets.float().to(device)
            predictions = model(texts).squeeze(1)
            predicted_labels = (predictions >= 0.5).long()
            correct += (predicted_labels == targets).sum().item()
            total += targets.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

def collate_fn(batch, pad_idx):
    """
    Fonction de collage pour ajouter du padding aux séquences de texte.
    """
    texts, targets = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    return texts_padded, torch.tensor(targets)

if __name__ == "__main__":
    main()
