import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import scaler as scaler
import pandas as pd

# PyTorch model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(8, 1) 

    def forward(self, x):
        return self.fc(x)

# FedAVG
class PyTorchFlowerStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures, min_fit_clients):
        aggregated_model = super().aggregate_fit(rnd, results, failures, min_fit_clients)
        return self.aggregated_model_to_tensor(aggregated_model)

    def aggregated_model_to_tensor(self, aggregated_model):
        state_dict = aggregated_model.parameters()
        return [param.data.numpy() for param in state_dict]

def start_server():
    model = Net()
    strategy = PyTorchFlowerStrategy(
        fraction_fit=0.2, fraction_eval=0.1, min_fit_clients=3, min_eval_clients=1
    )
    server = fl.Server(model, strategy)

    fl.app.serve(server, host="localhost", port=8080)
    
    df = pd.read_csv('your_data.csv')
    sc = scaler(df)

    train_fraction = 0.8
    train_df = df.sample(frac=train_fraction)
    test_df = df.drop(train_df.index)

    scaled_test_dataset = sc.scale_dataset(test_df)


    # Evaluate the final aggregated model on the test dataset
    final_absolute_error = evaluate(model, scaled_test_dataset)
    print(f"Final Mean Absolute Error on Test Dataset: {final_absolute_error}")


if __name__ == "__main__":
    start_server()
