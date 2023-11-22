import os

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from keras import layers
import utils as ut
import numpy as np
from typing import Dict, Optional, Tuple

model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(8,)),  
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),  
        layers.Dense(1)  
    ]
)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

(x_train, y_train), (x_test, y_test) = ut.partition_dataset(2,4,3)


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
        )

        # Return updated model parameters, number of examples trained, and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "mse": history.history["mse"][-1],  # Use the last epoch's MSE
            "mae": history.history["mae"][-1],  # Use the last epoch's MAE
            "val_mse": history.history["val_mse"][-1],  # Use the last epoch's validation MSE
            "val_mae": history.history["val_mae"][-1],  # Use the last epoch's validation MAE
        }

        return parameters_prime, num_examples_train, results

    def get_parameters(self):
        """Get the current parameters of the local model."""
        return self.model.get_weights()

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.model.set_weights(parameters)
        loss = tf.keras.losses.mean_squared_error(self.y_test, self.model.predict(self.x_test)).numpy().mean().item()
        mae = tf.keras.metrics.mean_absolute_error(self.y_test, self.model.predict(self.x_test)).numpy().mean().item()
        print("*************LOSS******************",loss)
        print("************MAE********************",mae)
        return loss, len(self.x_train), {"mae": mae}

    def get_weights(self):
        """Get the current weights of the local model."""
        return self.model.get_weights()

    def set_weights(self, weights):
        """Set the weights of the local model."""
        self.model.set_weights(weights)


client = CifarClient(model, x_train, y_train, x_test, y_test)

history = fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)