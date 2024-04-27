# -*- coding: utf-8 -*-
"""NORMALIZATION_MNIST_sv.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sl9WUQ4OTOD8gzmOpuIgtyYjFSXJtTI1
"""

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
#import os

from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from skew import *

print("bas3")
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
enable_tf_gpu_growth()


"""# Global Constants (Dataset Specific)"""
print("bas4")
# global variables
BATCH_SIZE = 8
EPOCH = 150
LEARNING_RATE = 0.0001
NORMALIZATION_LAYER = ''
NUM_CLIENTS = 10
#NORMALIZATION_TYPE = ''
EARLY_STOPPING_PATIENCE = 7
DATASET_INPUT_SHAPE = (31,1)
IS_IMAGE_DATA = False
#tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE) = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
print("bas2")
# variable arrays for each case
X_trains_fed = np.zeros(1)
Y_trains_fed = np.zeros(1)
X_test_fed = np.zeros(1) # test sets are not actually splitted but we use it as a variable array
Y_test_fed = np.zeros(1)
X_val_fed = np.zeros(1)
Y_val_fed = np.zeros(1)
print("bas")
# # constant arrays to always keep the originals
# X_TRAIN = np.zeros(1)
# Y_TRAIN = np.zeros(1)
# X_VALIDATION = np.zeros(1)
# Y_VALIDATION = np.zeros(1)
# X_TEST= np.zeros(1)
# Y_TEST= np.zeros(1)

"""# Dataset Retrieval"""

from sklearn.model_selection import train_test_split

def load_data(data_dir=""):
    df = pd.read_csv(f"{data_dir}/bcw.csv",header = 0)
    df.drop('id',axis=1,inplace=True)
    df.drop('Unnamed: 32',axis=1,inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
    X = df.drop('diagnosis', axis =1)
    Y = df["diagnosis"]
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, shuffle = True, random_state = 42)

    # Split train data into training and validation sets
    x_train, x_vals, y_train, y_vals = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    global X_trains_fed
    global Y_trains_fed
    global X_test_fed
    global Y_test_fed
    global X_val_fed
    global Y_val_fed

    X_trains_fed = np.array(x_train)
    Y_trains_fed = np.array(y_train)
    X_test_fed = np.array(x_test)
    Y_test_fed = np.array(y_test)
    X_val_fed = np.array(x_vals)
    Y_val_fed = np.array(y_vals)

    return
    #return x_train, y_train, x_test, y_test


def do_skew(skew_type, num_clients, X_data, Y_data):

    #depends on dataset
    ds_info = { 'num_clients' : num_clients , 'sample_shape' : DATASET_INPUT_SHAPE, 'num_classes':10 , 'seed':42}

    if(skew_type == 'feature_0.3'):
      clientsData, clientsDataLabels = feature_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, sigma = 0.3,
            decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'feature_0.7'):
      clientsData, clientsDataLabels = feature_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, sigma = 0.7,
            decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'label_5.0'):
      clientsData, clientsDataLabels = label_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, beta = 5.0,
            decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'label_0.5'):
      clientsData, clientsDataLabels = label_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, beta = 0.5,
        decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'quantity_5.0'):
      clientsData, clientsDataLabels = qty_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, beta = 5.0,
            decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'quantity_0.7'):
      clientsData, clientsDataLabels = qty_skew_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info, beta = 0.7,
            decentralized = True, display=False, is_tf=False)
    elif(skew_type == 'default'):
        clientsData, clientsDataLabels = iid_dist(x_train= X_data, y_train= Y_data, ds_info = ds_info,
            decentralized = True, display=False)
    else:
        print('error')



    return clientsData, clientsDataLabels

"""# Normalization Methods"""

def merge(data):
  return np.concatenate(data, axis=0)

def split(org_data, merged_array):
    total_length = sum(arr.shape[0] for arr in org_data)
    ratios = [arr.shape[0] / total_length for arr in org_data]
    split_indices = np.cumsum([int(ratio * merged_array.shape[0]) for ratio in ratios[:-1]])
    split_arrays = np.split(merged_array, split_indices)
    return split_arrays

def flatten_data(data):
    return data.reshape(data.shape[0],-1)

def flatten_data_for_clients(clientsData):
    flattened_clients_data = [flatten_data(client_data) for client_data in clientsData]
    return flattened_clients_data

def back_to_image(data):
    return data.reshape(data.shape[0], *DATASET_INPUT_SHAPE)

def back_to_image_for_clients(clientsData):
    back_to_image_data = [back_to_image(client_data) for client_data in clientsData]
    return back_to_image_data



def z_score(train,val,test):
    scaler = StandardScaler()
    scaler.fit(train,None)
    return (scaler.transform(train), scaler.transform(val), scaler.transform(test))

def min_max(train,val,test):
    scaler = MinMaxScaler()
    scaler.fit(train,None)
    return (scaler.transform(train), scaler.transform(val), scaler.transform(test))

def log_scaling(train, val, test):
    return (np.log10(train + 1), np.log10(val + 1), np.log10(test + 1))

def batch_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'batch_norm'
    return

def layer_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'layer_norm'
    return

def instance_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'instance_norm'
    return

def group_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'group_norm'
    return

def box_cox(train,val,test):
    transformer = PowerTransformer(method = 'box-cox')
    transformer.fit(train,None)
    #print("fit passed")
    return (transformer.transform(train+1), transformer.transform(val+1),transformer.transform(test+1))

def yeo_johnson(train,val,test):
    transformer = PowerTransformer(method = 'yeo-johnson')
    transformer.fit(train,None)
    return (transformer.transform(train), transformer.transform(val),transformer.transform(test))


def robust_scaling(train,val,test, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range)
    scaler.fit(train,None)
    return (scaler.transform(train), scaler.transform(val),scaler.transform(test))

def do_normalization(normalization_type, num_clients, X_data, val_x, test_x):


    if(IS_IMAGE_DATA):
        X_data = flatten_data_for_clients(X_data)
        val_x = flatten_data_for_clients(val_x)
        test_x = flatten_data_for_clients(test_x)


    global NORMALIZATION_LAYER

    NORMALIZATION_LAYER = 'default'

    if normalization_type == 'batch_norm':
        NORMALIZATION_LAYER = 'batch_norm'
    elif normalization_type == 'layer_norm':
        NORMALIZATION_LAYER = 'layer_norm'
    elif normalization_type == 'instance_norm':
        NORMALIZATION_LAYER = 'instance_norm'
    elif normalization_type == 'group_norm':
        NORMALIZATION_LAYER = 'group_norm'
    elif normalization_type == 'local_box_cox':
        for i in range(num_clients):
            print(i)
            X_data[i] = box_cox(X_data[i],val_x[i],test_x[i])

    elif normalization_type == 'local_yeo_johnson':
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = yeo_johnson(X_data[i], val_x[i], test_x[i])

    elif normalization_type == 'local_min_max':
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = min_max(X_data[i], val_x[i], test_x[i])

    elif normalization_type == 'local_z_score':
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = z_score(X_data[i], val_x[i], test_x[i])

    elif normalization_type == 'log_scaling':
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = log_scaling(X_data[i], val_x[i], test_x[i])

    elif normalization_type == 'local_robust_scaling':
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = robust_scaling(X_data[i], val_x[i], test_x[i])

    elif normalization_type == 'global_z_score':
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)
        merged_train, merged_val, merged_test = z_score(merged_train, merged_val, merged_test)
        X_data = split(X_data, merged_train)
        val_x = split(val_x, merged_val)
        test_x = split(test_x, merged_test)

    elif normalization_type == 'global_min_max':
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)
        merged_train, merged_val, merged_test = min_max(merged_train, merged_val, merged_test)
        X_data = split(X_data, merged_train)
        val_x = split(val_x, merged_val)
        test_x = split(test_x, merged_test)

    elif normalization_type == 'global_box_cox':
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)
        merged_train, merged_val, merged_test = box_cox(merged_train, merged_val, merged_test)
        X_data = split(X_data, merged_train)
        val_x = split(val_x, merged_val)
        test_x = split(test_x, merged_test)

    elif normalization_type == 'global_robust_scaling':
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)
        merged_train, merged_val, merged_test = robust_scaling(merged_train, merged_val, merged_test)
        X_data = split(X_data, merged_train)
        val_x = split(val_x, merged_val)
        test_x = split(test_x, merged_test)

    elif normalization_type == 'global_yeo_johnson':
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)
        merged_train, merged_val, merged_test = yeo_johnson(merged_train, merged_val, merged_test)
        X_data = split(X_data, merged_train)
        val_x = split(val_x, merged_val)
        test_x = split(test_x, merged_test)

    elif normalization_type == 'default':
        pass  # default case

    else:
        print("error")


    if(IS_IMAGE_DATA):
        X_data = back_to_image_for_clients(X_data)
        val_x = back_to_image_for_clients(val_x)
        test_x = back_to_image_for_clients(test_x)

    return X_data, val_x, test_x

"""# Network Model (Dataset Specific)"""

def get_model():    
    model = tf.keras.Sequential()
    model.add(tf.keras.Dense(16, activation='relu', input_dim=30))
    
    if NORMALIZATION_LAYER == 'batch_norm':
        model.add(tf.keras.BatchNormalization())
    elif NORMALIZATION_LAYER == 'instance_norm':
        # Instance normalization is typically not directly supported, hence using GroupNormalization with group size of 1
        model.add(tf.keras.layers.GroupNormalization(groups=1))
    elif NORMALIZATION_LAYER == 'group_norm':
        # Using an arbitrary group size, adjust based on the number of features or experimentation
        model.add(tf.keras.layers.GroupNormalization(groups=4))
    elif NORMALIZATION_LAYER == 'layer_norm':
        model.add(tf.keras.LayerNormalization())

    model.add(tf.keras.Dense(8, activation='relu'))
    model.add(tf.keras.Dense(8, activation='relu'))
    model.add(tf.keras.Dense(1, activation='sigmoid'))

    return model

"""# Flower Client (Dataset Specific)"""

from flwr.common.typing import NDArrays
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model: tf.keras.models.Sequential, X_train: np.ndarray, y_train: np.ndarray):
        self.model = model

        self.X_train = X_train
        self.y_train = y_train


    def get_parameters(self, config):
        return self.model.get_weights()


    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> NDArrays:

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.set_weights(parameters)

        history = self.model.fit(self.X_train, self.y_train ,batch_size=BATCH_SIZE, epochs=1, verbose=0)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
        }
        return self.model.get_weights(), len(self.X_train), results

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar])-> Tuple[float, int, Dict[str, Scalar]]:
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.set_weights(parameters)

        loss, acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"accuracy": acc}

def create_client_fn(cid: str) -> FlowerClient:
    model = get_model()
    cid_int = int(cid)
    return FlowerClient(model, X_trains_fed[cid_int], Y_trains_fed[cid_int])

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

from flwr.server import Server
from logging import INFO
from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager, SimpleClientManager
import timeit

class CustomFlowerServer(Server):
    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)


    # Override
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        # Early Stopping Parameters
        best_loss = float("inf")
        patience_counter = 0
        minimum_delta = 0.001  # Example: require at least 0.001 decrease in loss

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            if res_cen is not None:
                loss_cen, metrics_cen = res_cen

                # Check for improvement
                if loss_cen < best_loss - minimum_delta:
                    best_loss = loss_cen
                    patience_counter = 0  # Reset counter if improvement
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(INFO, "Early stopping triggered at round %s", current_round)
                    break  # Exit the training loop

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

# Required for early stopping
best_accuracy = 0.0
weights = np.array([])
best_loss = float("inf")
minimum_delta = 0.001  # Example: require at least 0.001 decrease in loss

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    """Centralized evaluation function"""

    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy' , metrics=['accuracy'])

    model.set_weights(parameters)

    loss, accuracy = model.evaluate(X_val_fed, Y_val_fed, batch_size=BATCH_SIZE, verbose=0)

    global best_accuracy
    global best_loss
    global weights

    print(f"LOSS: {loss}")
    print(f"BEST_LOSS: {best_loss}")
    print(f"ACCURACY: {accuracy}")
    print(f"BEST_ACCURACY: {best_accuracy}")
    print(f"SERVER_ROUND: {server_round}")

    if loss < best_loss - minimum_delta:
        best_accuracy = accuracy
        weights = parameters
        best_loss = loss

    return loss, {"accuracy": accuracy}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
def get_results():
    '''
    At the end of the federated learning process, calculates results and returns
    Test F1 Score
    Test Loss
    Test Accuracy
    Test Precision
    Test Recall
    '''
    global X_test_fed, Y_test_fed

    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy' , metrics=['accuracy'])
    model.set_weights(weights)

    test_loss, test_accuracy = model.evaluate(X_test_fed, Y_test_fed, batch_size=BATCH_SIZE, verbose=0)

    y_pred_probs = model.predict(X_test_fed)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

    # If y_test is one-hot encoded, convert it back to integer labels
    if len(Y_test_fed.shape) > 1:
        Y_test_fed = np.argmax(Y_test_fed, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(Y_test_fed, y_pred)
    precision = precision_score(Y_test_fed, y_pred, average='weighted')
    recall = recall_score(Y_test_fed, y_pred, average='weighted')
    f1 = f1_score(Y_test_fed, y_pred, average='weighted')

    return accuracy, precision, recall, f1, test_loss

def federated_train(x, y, num_clients):

    global X_trains_fed
    global Y_trains_fed
    X_trains_fed = x
    Y_trains_fed = y

    client_resources = {"num_cpus": 2}
    if tf.config.get_visible_devices("GPU"):
        client_resources["num_gpus"] = 1

    # Specify the Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,  # Wait until all 8 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=EPOCH),
        server=CustomFlowerServer(client_manager=SimpleClientManager(),
        strategy=strategy
        ),
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )

    accuracy, precision, recall, f1, test_loss = get_results()

    return history, accuracy, precision, recall, f1, test_loss

"""# Training and Saving the Results"""

import wandb

wandb.login()

sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'val_loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'b_skew': {
          'values': ['default', 'feature_0.3', 'feature_0.7', 'label_5.0', 'label_0.5', 'quantity_5.0', 'quantity_0.7']
        },
    'a_num_clients': {
          'values': [10, 20, 50]
        },
    'c_normalization': {
          'values': ['local_z_score', 'local_min_max', 'batch_norm', 'layer_norm', 'group_norm', 'instance_norm', 'local_robust_scaling',
                     'global_z_score', 'global_min_max','global_robust_scaling',
                     'default']
        }
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'dataset': {
        'value': 'BCW'},
    'experiment_run': {
        'value': '1'}
    })

sweep_id = wandb.sweep(sweep_config, project="mnist_test_v6")
#sweep_id = "xecrtjox" # to continue a sweep
import time

def train(config = None):
    with wandb.init(config=config):

        config = wandb.config
        tf.keras.backend.clear_session()

        global NUM_CLIENTS
        global X_test_fed
        global X_val_fed


        NUM_CLIENTS = config['a_num_clients']

        load_data()

        clientsData, clientLabels = do_skew(config['b_skew'], config['a_num_clients'], X_trains_fed, Y_trains_fed)
        val_x = split(clientsData, X_val_fed)
        test_x = split(clientsData, X_test_fed)

        #validation and test datasets are normalized with same parameters train dataset is normalized.
        #Data distribution among clients are protected, for local normalization, val and test datasets are normalized with respect to their local train data normalization parameters.
        normalizedData, val_x, test_x = do_normalization(config['c_normalization'], config['a_num_clients'], clientsData, val_x, test_x)

        X_val_fed = merge(val_x)
        X_test_fed = merge(test_x)

        t1 = time.perf_counter(), time.process_time()

        history, test_accuracy, test_precision, test_recall, test_f1, test_loss = federated_train(normalizedData, clientLabels, config['a_num_clients'])

        t2 = time.perf_counter(), time.process_time()

        t = t2[1] - t1[1]

        global best_accuracy, weights, best_loss


        if (len(history.losses_centralized) == EPOCH + 1):
            early_stopped_epoch = len(history.losses_centralized) - 1
        else:
            early_stopped_epoch = len(history.losses_centralized) - EARLY_STOPPING_PATIENCE - 1

        # Saving the results
        wandb.log({"time": t})
        wandb.log({"stopped_epoch": early_stopped_epoch})
        wandb.log({"test_accuracy": test_accuracy})
        wandb.log({"test_precision": test_precision})
        wandb.log({"test_recall": test_recall})
        wandb.log({"test_f1": test_f1})
        wandb.log({"test_loss": test_loss})
        wandb.log({"validation_loss": best_loss})
        wandb.log({"validation_accuracy": best_accuracy})


        table = wandb.Table(data=history.losses_distributed, columns=["x", "y"])
        wandb.log(
            {
                "distributed_loss": wandb.plot.line(
                    table, "x", "y", title="Train Set Loss vs Epoch Plot"
                )
            }
        )


        table2 = wandb.Table(data=history.losses_centralized, columns=["x", "y"])
        wandb.log(
            {
                "centralized_loss": wandb.plot.line(
                    table2, "x", "y", title="Validation Set Loss vs Epoch Plot"
                )
            }
        )

        table3 = wandb.Table(data=history.metrics_distributed['accuracy'], columns=["x", "y"])
        wandb.log(
            {
                "distributed_accuracy": wandb.plot.line(
                    table3, "x", "y", title="Train Set Accuracy vs Epoch Plot"
                )
            }
        )

        table4 = wandb.Table(data=history.metrics_centralized['accuracy'], columns=["x", "y"])
        wandb.log(
            {
                "centralized_accuracy": wandb.plot.line(
                    table4, "x", "y", title="Validation Set Accuracy vs Epoch Plot"
                )
            }
        )

        best_accuracy = 0.0
        weights = np.array([])
        best_loss = float("inf")


wandb.agent(sweep_id, train)
#wandb.agent(sweep_id, project="mnist_test_v6", function=train) # to continue a sweep