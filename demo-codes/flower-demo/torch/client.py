import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import scaler as scaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

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
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate mean absolute error at the end of each epoch
        epoch_absolute_error = evaluate(model, train_loader)
        epoch_errors.append(epoch_absolute_error)

        print(f"Epoch {epoch + 1}/{epochs} - Mean Absolute Error: {epoch_absolute_error}")

    # Plot mean absolute error per epoch
    plt.plot(range(1, epochs + 1), epoch_errors, marker='o')
    plt.title('Mean Absolute Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.show()


def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_true.extend(labels.numpy())
            y_pred.extend(outputs.numpy())

    return mean_absolute_error(y_true, y_pred)

def start_client(client_id, data, sc):
    model = Net()

    scaled_dataset = sc.scale_dataset(data)

    features = scaled_dataset.drop("target_column", axis=1).values
    labels = scaled_dataset["target_column"].values

    train_data = torch.utils.data.TensorDataset(torch.tensor(features, dtype=torch.float32),
                                                torch.tensor(labels, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # Evaluate the initial model before training
    initial_absolute_error = evaluate(model, train_loader)
    print(f"Client {client_id} - Initial Absolute Error: {initial_absolute_error}")

    train(model, train_loader, epochs=3) 

    final_absolute_error = evaluate(model, train_loader)
    print(f"Client {client_id} - Final Absolute Error: {final_absolute_error}")





df = pd.read_csv('cal_housing.csv')
sc = scaler(df)

for client_id in range(10):
    client_data = df.sample(frac=0.1) 

    start_client(client_id, client_data,sc)