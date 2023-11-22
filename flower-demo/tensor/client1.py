import os

import flwr as fl
import tensorflow as tf
import utils as ut


model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(8,)),  
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),  
        layers.Dense(1)  
    ]
)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

(x_train, y_train), (x_test, y_test) = ut.partition_dataset(2,3,1)


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())