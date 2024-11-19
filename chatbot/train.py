import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# Charger les intentions depuis le fichier JSON
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialisation des listes
all_words = []
tags = []
xy = []

# Parcourir les intentions pour extraire les tags et les patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Prétraitement des mots
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(tags)

# Création des ensembles d'entraînement
x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)  # Correction : utiliser le tag ici
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Définition du Dataset
class ChatDataSet(Dataset):
    def __init__(self):  # Correction du nom de la méthode
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):  # Correction du nom de la méthode
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # Correction du nom de la méthode
        return self.n_samples

if __name__ == "__main__":
    # Hyperparamètres
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = 1000

    # Initialisation du Dataset et du DataLoader
    dataset = ChatDataSet()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set num_workers=0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}')

    print(f'Final loss: Loss={loss.item():.4f}')
