import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# PyTorch model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(8, 1)  # Adjust input size based on your dataset

    def forward(self, x):
        return self.fc(x)

# PyTorch model training function
def train(model, train_loader, epochs=1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Connect to the Flower server
def start_client(client_id, data):
    model = Net()

    # Convert DataFrame to PyTorch tensors
    features = torch.tensor(data.drop("target_column", axis=1).values, dtype=torch.float32)
    labels = torch.tensor(data["target_column"].values, dtype=torch.float32)

    # Create a PyTorch DataLoader
    train_data = torch.utils.data.TensorDataset(features, labels)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    fl.client.start_numpy_client(f"localhost:8080", model=model, train_fn=train, num_rounds=3, client_id=f"client_{client_id}", train_loader=train_loader)

# Load your California housing dataset into a Pandas DataFrame (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('your_data.csv')

# Assuming you want to run 10 clients
for client_id in range(10):
    # Distribute data to clients (adjust data distribution logic based on your needs)
    client_data = df.sample(frac=0.1)  # Adjust the fraction as needed

    # Start each client with its respective data
    start_client(client_id, client_data)
