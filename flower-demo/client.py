import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PyTorch model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(8, 1)  

    def forward(self, x):
        return self.fc(x)

#DATA

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
def start_client(client_id):
    model = Net()

    #DATA

    # Adjust client ID to distinguish between clients
    fl.client.start_numpy_client(f"localhost:8080", model=model, train_fn=train, num_rounds=3, client_id=f"client_{client_id}")

# Assuming you want to run 10 clients
for client_id in range(10):
    start_client(client_id)