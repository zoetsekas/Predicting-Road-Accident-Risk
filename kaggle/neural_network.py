import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, output_size=1):
        super(NeuralNetwork, self).__init__()
        
        layer_sizes = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def create_model(input_size, hidden_size, num_hidden_layers, dropout_rate, learning_rate, optimizer_name):
    model = NeuralNetwork(input_size, hidden_size, num_hidden_layers, dropout_rate)
    
    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    return model, optimizer

def train_model(model, optimizer, train_loader, epochs=10):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    # Parameters
    input_size = 5 # for dummy data
    hidden_size = 64
    num_hidden_layers = 2
    dropout_rate = 0.5
    learning_rate = 0.001
    optimizer_name = 'Adam'

    # Create dummy data for demonstration
    X = np.random.rand(100, input_size).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = []
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create and train the model
        model, optimizer = create_model(
            input_size, 
            hidden_size, 
            num_hidden_layers, 
            dropout_rate, 
            learning_rate, 
            optimizer_name
        )
        
        # Simple training loop
        train_model(model, optimizer, train_loader)
        
        models.append(model)

        print(f"Model for fold {fold+1} created and trained successfully.")

    print(f"\n{n_splits}-fold cross-validation finished. {len(models)} models created.")
