import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import utils as ut
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np

def get_evaluate_fn(model, x_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        # Update the model with the latest parameters
        model.set_weights(parameters)

        # Evaluate the model on the test data
        predictions = model.predict(x_test)
        loss = tf.keras.losses.mean_squared_error(y_test, predictions).numpy()
        mae = tf.keras.metrics.mean_absolute_error(y_test, predictions).numpy()

        # Assuming you want to return both loss and mean absolute error
        return loss, {"mae": mae}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 128,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

(x_train, y_train), (x_test, y_test) = ut.partition_dataset(2,3,1)

model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(8,)),  
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),  
        layers.Dense(1)  
    ]
)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.75,
    fraction_evaluate=1,
    min_fit_clients=2,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_fn=get_evaluate_fn(model,x_test,y_test),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
)
# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy = strategy
)