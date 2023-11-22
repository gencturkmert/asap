import math
import argparse
from typing import Dict, List, Tuple

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split




NUM_CLIENTS = 100
VERBOSE = 0


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val) -> None:
        # Create model
        self.model = get_model()
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train, epochs=1, batch_size=32, verbose=VERBOSE
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(
            self.x_val, self.y_val, batch_size=64, verbose=VERBOSE
        )
        return loss, len(self.x_val), {"accuracy": acc}


def get_model():
    model = tf.keras.models.Sequential(
        [
        layers.Input(shape=(8,)),  
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),  
        layers.Dense(1)  
        ]
    )
    model.compile("adam", "mean square error",metrics=["accuracy",'mae', 'mse'] )
    return model


def get_client_fn(dataset_partitions):
    """Return a function to construc a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        x_train, y_train = dataset_partitions[int(cid)]
        # Use 10% of the client's training data for validation
        split_idx = math.floor(len(x_train) * 0.9)
        x_train_cid, y_train_cid = (
            x_train[:split_idx],
            y_train[:split_idx],
        )
        x_val_cid, y_val_cid = x_train[split_idx:], y_train[split_idx:]

        # Create and return client
        return FlowerClient(x_train_cid, y_train_cid, x_val_cid, y_val_cid)

    return client_fn


def partition_dataset(noise: int):
    #Adds noise to given val -> val = ((noise + 100) * val / 100)) 
    def add_noise(val):
        return val * ((100 + noise) /100)


    # Load California Housing dataset
    california_housing = fetch_california_housing()

    housing = pd.DataFrame(california_housing.data,columns=california_housing.feature_names)

    #Min Max Norm. using noised values
    scaled_features = (housing - add_noise(housing.min())) / add_noise((housing.max()) - add_noise(housing.min()))

    x, y = scaled_features.values, california_housing.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    partitions = []

    partition_size = math.floor(len(x_train) / NUM_CLIENTS)

    for cid in range(NUM_CLIENTS):
        # Split dataset into non-overlapping NUM_CLIENT partitions
        idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
        partitions.append((x_train[idx_from:idx_to] / 255.0, y_train[idx_from:idx_to]))

    # Test set
    testset = (x_test / 255.0, y_test)

    return partitions, testset


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics.

    It ill aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    x_test, y_test = testset

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate




#Simulation
loss_with_noise = []
for i in range(-4,4):
  # Create dataset partitions (needed if your dataset is not pre-partitioned)
  partitions, testset = partition_dataset(i)

  # Create FedAvg strategy
  strategy = fl.server.strategy.FedAvg(
      fraction_fit=0.1,  # Sample 10% of available clients for training
      fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
      min_fit_clients=10,  # Never sample less than 10 clients for training
      min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
      min_available_clients=int(
          NUM_CLIENTS * 0.75
      ),  # Wait until at least 75 clients are available
      evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
      evaluate_fn=get_evaluate_fn(testset),  # global evaluation function
  )



  # Start simulation
  history  = fl.simulation.start_simulation(
      client_fn=get_client_fn(partitions),
      num_clients=NUM_CLIENTS,
      config=fl.server.ServerConfig(num_rounds=10),
      strategy=strategy,
      actor_kwargs={
          "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
      },
  )

  loss_with_noise.append(history.history['loss'][-1])

noise_values = np.arange(-4, 4, 1)

# Create a line plot
plt.plot(noise_values, loss_with_noise, marker='o', linestyle='-')

# Labeling the axes and giving the plot a title
plt.xlabel('Noise')
plt.ylabel('Loss')
plt.title('Loss vs. Noise')

# Display the plot
plt.grid(True)
plt.show()



