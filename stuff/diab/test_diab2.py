import numpy as np
import argparse
import numpy as np
import csv
import tensorflow as tf
import wandb
import time
import math
import gc
import pandas as pd
import sys
import os

from norms import *
from utils import *
from skew import *

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

"""# Global Constants (Dataset Specific)"""
# global variables
BATCH_SIZE = 32
EPOCH = 100
LEARNING_RATE = 0.001
NORMALIZATION_LAYER = ""
EARLY_STOPPING_PATIENCE = 7
DATASET_INPUT_SHAPE = 21


def initialize_globals():
    global X_trains_fed, Y_trains_fed, X_test_fed, Y_test_fed, X_val_fed, Y_val_fed
    global global_weights, best_f1, best_acc, weights, best_loss

    # Dataset-related globals
    X_trains_fed = np.zeros((1,))
    Y_trains_fed = np.zeros((1,))
    X_test_fed = np.zeros((1,))
    Y_test_fed = np.zeros((1,))
    X_val_fed = np.zeros((1,))
    Y_val_fed = np.zeros((1,))

    # Model-related globals
    global_weights = np.zeros((1,))
    best_f1 = 0.0
    best_acc = 0.0
    weights = np.array([])
    best_loss = float("inf")


"""# Dataset Retrieval"""


def load_data(data_dir=""):
    # Load X data
    file_path = "/home/mertgencturk/asap/stuff/diab/diab_5050.csv"
    
    data = pd.read_csv(file_path)
    #data = data.drop(columns=["encounter_id", "patient_nbr"])

    #class_0 = data[data["readmitted"] == 0]
    #class_1 = data[data["readmitted"] == 1]

    # Resample both classes to have 50k samples each
    #class_0_resampled = resample(class_0, replace=False, n_samples=60000, random_state=31)
    #class_1_resampled = resample(class_1, replace=True, n_samples=50000, random_state=31)

    # Combine resampled data
    #resampled_data = pd.concat([class_0_resampled, class_1_resampled])

    # Shuffle the data to mix the classes
    #resampled_data = resampled_data.sample(frac=1, random_state=31).reset_index(drop=True)
    
    y_data = data["Diabetes_binary"].to_numpy()
    
    data.drop("Diabetes_binary",axis=1, inplace=True)
    x_data = data.to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, shuffle=True, random_state=42
    )
    x_train, x_vals, y_train, y_vals = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )

    global X_trains_fed
    global Y_trains_fed
    global X_test_fed
    global Y_test_fed
    global X_val_fed
    global Y_val_fed

    X_trains_fed = x_train
    Y_trains_fed = y_train
    X_test_fed = x_test
    Y_test_fed = y_test
    X_val_fed = x_vals
    Y_val_fed = y_vals

    return


def do_skew(skew_type, num_clients, X_data, Y_data):
    if skew_type == "feature_0.3":
        clientsData, clientsDataLabels = feature_skew_dist(
            X_data, Y_data, num_clients, sigma=0.01, batch_size=BATCH_SIZE
        )
    elif skew_type == "feature_0.7":
        clientsData, clientsDataLabels = feature_skew_dist(
            X_data, Y_data, num_clients, sigma=0.1, batch_size=BATCH_SIZE
        )
    elif skew_type == "label_5.0":
        clientsData, clientsDataLabels = label_skew_dist(
            X_data, Y_data, num_clients, 2, beta=5, batch_size=BATCH_SIZE
        )
    elif skew_type == "label_0.5":
        clientsData, clientsDataLabels = label_skew_dist(
            X_data, Y_data, num_clients, 2, beta=0.5, batch_size=BATCH_SIZE
        )
    elif skew_type == "quantity_5.0":
        clientsData, clientsDataLabels = qty_skew_dist(
            X_data, Y_data, num_clients, beta=5, batch_size=BATCH_SIZE
        )
    elif skew_type == "quantity_0.7":
        clientsData, clientsDataLabels = qty_skew_dist(
            X_data, Y_data, num_clients, beta=0.7, batch_size=BATCH_SIZE
        )
    elif skew_type == "default":
        clientsData, clientsDataLabels = iid_dist(
            X_data, Y_data, num_clients, batch_size=BATCH_SIZE
        )
    else:
        print("error")

    return clientsData, clientsDataLabels


"""# Normalization Methods"""


def do_normalization(
    normalization_type, num_clients, X_data, val_x, test_x, Y_data, val_y, test_y
):

    global NORMALIZATION_LAYER

    NORMALIZATION_LAYER = "default"

    if normalization_type == "batch_norm":
        NORMALIZATION_LAYER = "batch_norm"
    elif normalization_type == "layer_norm":
        NORMALIZATION_LAYER = "layer_norm"
    elif normalization_type == "instance_norm":
        NORMALIZATION_LAYER = "instance_norm"
    elif normalization_type == "group_norm":
        NORMALIZATION_LAYER = "group_norm"
    elif normalization_type == "local_box_cox":
        for i in range(num_clients):
            print(i)
            X_data[i] = box_cox(X_data[i], val_x[i], test_x[i])

    elif normalization_type == "local_yeo_johnson":
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = yeo_johnson(X_data[i], val_x[i], test_x[i])

    elif normalization_type == "local_min_max":
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = min_max(X_data[i], val_x[i], test_x[i])

    elif normalization_type == "local_z_score":
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = z_score(X_data[i], val_x[i], test_x[i])

    elif normalization_type == "log_scaling":
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = log_scaling(X_data[i], val_x[i], test_x[i])

    elif normalization_type == "local_robust_scaling":
        for i in range(num_clients):
            X_data[i], val_x[i], test_x[i] = robust_scaling(
                X_data[i], val_x[i], test_x[i]
            )

    elif normalization_type == "global_z_score":
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)

        merget_train_y = merge(Y_data)
        merged_val_y = merge(val_y)
        merged_test_y = merge(test_y)

        merged_train, merged_val, merged_test = z_score(
            merged_train, merged_val, merged_test
        )

        X_data, Y_data = split(X_data, merged_train, merget_train_y)
        val_x, val_y = split(val_x, merged_val, merged_val_y)
        test_x, test_y = split(test_x, merged_test, merged_test_y)

    elif normalization_type == "global_min_max":
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)

        merget_train_y = merge(Y_data)
        merged_val_y = merge(val_y)
        merged_test_y = merge(test_y)

        merged_train, merged_val, merged_test = min_max(
            merged_train, merged_val, merged_test
        )

        X_data, Y_data = split(X_data, merged_train, merget_train_y)
        val_x, val_y = split(val_x, merged_val, merged_val_y)
        test_x, test_y = split(test_x, merged_test, merged_test_y)

    elif normalization_type == "global_box_cox":
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)

        merget_train_y = merge(Y_data)
        merged_val_y = merge(val_y)
        merged_test_y = merge(test_y)

        merged_train, merged_val, merged_test = box_cox(
            merged_train, merged_val, merged_test
        )

        X_data, Y_data = split(X_data, merged_train, merget_train_y)
        val_x, val_y = split(val_x, merged_val, merged_val_y)
        test_x, test_y = split(test_x, merged_test, merged_test_y)

    elif normalization_type == "global_robust_scaling":
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)

        merget_train_y = merge(Y_data)
        merged_val_y = merge(val_y)
        merged_test_y = merge(test_y)

        merged_train, merged_val, merged_test = robust_scaling(
            merged_train, merged_val, merged_test
        )

        X_data, Y_data = split(X_data, merged_train, merget_train_y)
        val_x, val_y = split(val_x, merged_val, merged_val_y)
        test_x, test_y = split(test_x, merged_test, merged_test_y)

    elif normalization_type == "global_yeo_johnson":
        merged_train = merge(X_data)
        merged_val = merge(val_x)
        merged_test = merge(test_x)

        merget_train_y = merge(Y_data)
        merged_val_y = merge(val_y)
        merged_test_y = merge(test_y)
        merged_train, merged_val, merged_test = yeo_johnson(
            merged_train, merged_val, merged_test
        )

        X_data, Y_data = split(X_data, merged_train, merget_train_y)
        val_x, val_y = split(val_x, merged_val, merged_val_y)
        test_x, test_y = split(test_x, merged_test, merged_test_y)

    elif normalization_type == "default":
        pass  # default case

    else:
        print("error")

    return X_data, val_x, test_x, Y_data, val_y, test_y


"""# Network Model (Dataset Specific)"""


def get_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(10, input_dim=DATASET_INPUT_SHAPE, activation='sigmoid'))

    if NORMALIZATION_LAYER == "batch_norm":
        model.add(tf.keras.layers.BatchNormalization())
    elif NORMALIZATION_LAYER == "instance_norm":
        model.add(tf.keras.layers.GroupNormalization(groups=1))
    elif NORMALIZATION_LAYER == "group_norm":
        model.add(tf.keras.layers.GroupNormalization(groups=2))
    elif NORMALIZATION_LAYER == "layer_norm":
        model.add(tf.keras.layers.LayerNormalization())

    #model.add(tf.keras.layers.Dense(4, activation='relu'))

    #model.add(tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


"""# Flower Client (Dataset Specific)"""


# Required for early stopping


def get_results(global_weights):
    """
    At the end of the federated learning process, calculates results and returns
    Test F1 Score
    Test Loss
    Test Accuracy
    Test Precision
    Test Recall
    """
    global X_test_fed, Y_test_fed

    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],  # Add accuracy as a metric
    )
    model.set_weights(global_weights)

    test_loss, test_accuracy = model.evaluate(X_test_fed, Y_test_fed, batch_size=BATCH_SIZE, verbose=1)

    y_pred = model.predict(X_test_fed)
    y_pred = (y_pred.flatten() >= 0.5).astype(int)
    
    f1 = f1_score(Y_test_fed, y_pred)
    precision = precision_score(Y_test_fed, y_pred)
    recall = recall_score(Y_test_fed, y_pred)
    print(classification_report(Y_test_fed,y_pred))
    
    return test_accuracy, test_loss, f1, precision, recall

def exponential_lr_decay(current_epoch,decay_rate=0.01):
    return LEARNING_RATE # float(LEARNING_RATE * math.exp(-decay_rate * current_epoch))

def federated_train(x, y, num_clients):
    global X_val_fed, Y_val_fed, best_f1, best_acc, best_loss
    
    total_samples = sum([len(data) for data in x])

    client_sample_weights = [len(data) / total_samples for data in x]

    model = get_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
    )   

    # Initialize global weights
    global_weights = model.get_weights()

    # Initialize history and early stopping parameters
    history = {
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "distributed_train_loss": [],
        "centralized_train_loss": [],
        "centralized_train_acc": []
    }
    patience_counter = 0

    for epoch in range(EPOCH):
        client_weights = [np.zeros_like(layer) for layer in global_weights]
        client_train_losses = []

        # Training for each client
        for i in range(num_clients):
            
            model.set_weights(global_weights)
            
            optimizer.learning_rate.assign(exponential_lr_decay(current_epoch=epoch))
            
            history_per_client = model.fit(x[i], y[i], batch_size=BATCH_SIZE, epochs=1, verbose=0)
            client_train_losses.append(history_per_client.history["loss"][-1])
            client_weights = [
                cw + fw for cw, fw in zip(client_weights, fedavg_per_client(model.get_weights(), client_sample_weights[i]))
            ]            
        # Calculate distributed training loss (average of client losses)
        avg_train_loss = np.mean(client_train_losses)
        history["distributed_train_loss"].append(avg_train_loss)

        global_weights = [np.copy(layer) for layer in client_weights]

        # Set global weights for the global model
        model.set_weights(global_weights)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],  # Add accuracy as a metric
        )

        # Evaluate centralized training loss (global model on all client training data)
        combined_x = np.concatenate(x, axis=0)
        combined_y = np.concatenate(y, axis=0)
        centralized_train_loss, centralized_train_acc = model.evaluate(
            combined_x, combined_y, batch_size=BATCH_SIZE, verbose=0
        )
        history["centralized_train_loss"].append(centralized_train_loss)
        history["centralized_train_acc"].append(centralized_train_acc)

        val_loss, val_acc = model.evaluate(
            X_val_fed, Y_val_fed, batch_size=BATCH_SIZE, verbose=0
        )
        
        y_pred = model.predict(X_val_fed)
        y_pred = (y_pred.flatten() >= 0.5).astype(int)
        
        print("Pred Shape", y_pred.shape)
        print("Val Y Shape", Y_val_fed.shape)
        
        val_f1 = f1_score(Y_val_fed, y_pred, average="weighted", zero_division=1)

        # Store validation metrics in history
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Distributed Train Loss: {avg_train_loss}")
        print(f"  Centralized Train Loss: {centralized_train_loss}")
        print(f"  Validation Loss: {val_loss}")
        print(f"  Validation Acc: {val_acc}")
        print(f"  Validation F1: {val_f1}")
        
        # Early stopping logic
        if val_loss < best_loss - 0.00001:
            best_loss = val_loss
            best_f1 = val_f1
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        del client_weights    
        gc.collect()
        

    test_accuracy, test_loss, f1, precision, recall = get_results(global_weights)
    del global_weights
    return history, test_accuracy, test_loss, f1, precision, recall


def fedavg(client_weights, client_sample_weights):
    weighted_avg_weights = []
    for layer in range(len(client_weights[0])):
        weighted_layer = sum(
            client_sample_weights[i] * client_weights[i][layer]
            for i in range(len(client_weights))
        )
        weighted_avg_weights.append(weighted_layer)

    return weighted_avg_weights


def fedavg_per_client(client_weights, client_sample_weight):
    return [client_sample_weight * layer for layer in client_weights]



def train(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tf.keras.backend.clear_session()
        gc.collect()
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            

        global X_test_fed
        global Y_test_fed

        global X_val_fed
        global Y_val_fed


        load_data()

        clientsData, clientLabels = do_skew(
            config["b_skew"], config["a_num_clients"], X_trains_fed, Y_trains_fed
        )
        val_x, val_y = split(clientsData, X_val_fed, Y_val_fed)
        test_x, test_y = split(clientsData, X_test_fed, Y_test_fed)

        # validation and test datasets are normalized with same parameters train dataset is normalized.
        # Data distribution among clients are protected, for local normalization, val and test datasets are normalized with respect to their local train data normalization parameters.
        normalizedData, val_x, test_x, clientLabels, val_y, test_y = do_normalization(
            config["c_normalization"],
            config["a_num_clients"],
            clientsData,
            val_x,
            test_x,
            clientLabels,
            val_y,
            test_y,
        )

        X_val_fed = merge(val_x)
        Y_val_fed = merge(val_y)

        X_test_fed = merge(test_x)
        Y_test_fed = merge(test_y)

        t1 = time.perf_counter(), time.process_time()

        # Call the federated_train function
        history, test_accuracy, test_loss, f1, precision, recall = federated_train(
            normalizedData, clientLabels, config["a_num_clients"]
        )
        print("Test Results:",test_accuracy, test_loss, f1, precision, recall)

        t2 = time.perf_counter(), time.process_time()
        t = t2[1] - t1[1]

        # Use global variables for best metrics and weights
        global best_f1, best_acc, global_weights, best_loss

        # Determine early-stopped epoch based on validation loss history
        early_stopped_epoch = (
            len(history["centralized_train_loss"]) - EARLY_STOPPING_PATIENCE - 1
            if len(history["centralized_train_loss"]) < EPOCH + 1
            else len(history["centralized_train_loss"]) - 1
        )

        # Log time and results to WandB
        wandb.log({"time": t})
        wandb.log({"stopped_epoch": early_stopped_epoch})
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_acc": test_accuracy})
        wandb.log({"test_f1": f1})
        wandb.log({"test_precision": precision})
        wandb.log({"test_recall": recall})
        wandb.log({"validation_loss": best_loss})
        wandb.log({"validation_f1": best_f1})
        wandb.log({"validation_acc": best_acc})
        

        # Log distributed and centralized losses
        distributed_loss_table = wandb.Table(
            data=[
                [i, loss] for i, loss in enumerate(history["distributed_train_loss"])
            ],
            columns=["Epoch", "Distributed Loss"],
        )
        wandb.log(
            {
                "distributed_loss": wandb.plot.line(
                    distributed_loss_table,
                    "Epoch",
                    "Distributed Loss",
                    title="Distributed Training Loss vs Epoch",
                )
            }
        )

        centralized_loss_table = wandb.Table(
            data=[
                [i, loss] for i, loss in enumerate(history["centralized_train_loss"])
            ],
            columns=["Epoch", "Centralized Loss"],
        )
        wandb.log(
            {
                "centralized_loss": wandb.plot.line(
                    centralized_loss_table,
                    "Epoch",
                    "Centralized Loss",
                    title="Centralized Training Loss vs Epoch",
                )
            }
        )
        
        centralized_acc_table = wandb.Table(
            data=[
                [i, loss] for i, loss in enumerate(history["centralized_train_acc"])
            ],
            columns=["Epoch", "Centralized Accuracy"],
        )
        wandb.log(
            {
                "centralized_accuracy": wandb.plot.line(
                    centralized_acc_table,
                    "Epoch",
                    "Centralized Accuracy",
                    title="Centralized Training Accuracy vs Epoch",
                )
            }
        )

        # Log validation metrics
        validation_loss_table = wandb.Table(
            data=[[i, loss] for i, loss in enumerate(history["val_loss"])],
            columns=["Epoch", "Validation Loss"],
        )
        wandb.log(
            {
                "validation_loss": wandb.plot.line(
                    validation_loss_table,
                    "Epoch",
                    "Validation Loss",
                    title="Validation Loss vs Epoch",
                )
            }
        )

        validationf1_table = wandb.Table(
            data=[[i, r2] for i, r2 in enumerate(history["val_f1"])],
            columns=["Epoch", "Validation F!1"],
        )
        wandb.log(
            {
                "validation_f1": wandb.plot.line(
                    validationf1_table,
                    "Epoch",
                    "Validation F1",
                    title="Validation F1 vs Epoch",
                )
            }
        )

        best_f1 = 0.0
        best_acc = 0.0
        global_weights = np.array([])
        best_loss = float("inf")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run WandB sweep with custom sweep_id and project name."
    )
    parser.add_argument("--sweep_id", type=str, required=False, help="WandB Sweep ID")
    parser.add_argument("--project", type=str, required=True, help="WandB Project Name")
    parser.add_argument("--run",type=int, required=True,help="Run Count")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    initialize_globals()
    
    gc.collect()

    sweep_config = {"method": "grid"}

    metric = {"name": "val_loss", "goal": "minimize"}

    sweep_config["metric"] = metric

    parameters_dict = {
        "b_skew": {
            "values": [
                "default",
                "feature_0.3",
                "feature_0.7",
                "label_5.0",
                "label_0.5",
                "quantity_5.0",
                "quantity_0.7",
            ]
        },
        "a_num_clients": {"values": [10, 20, 30]},
        "c_normalization": {
            "values": [
                "local_yeo_johnson",
                "global_yeo_johnson",
                "local_z_score",
                "local_min_max",
                "batch_norm",
                "layer_norm",
                "group_norm",
                "instance_norm",
                "local_robust_scaling",
                "global_z_score",
                "global_min_max",
                "global_robust_scaling",
                "default",
            ]
        },
    }

    sweep_config["parameters"] = parameters_dict

    parameters_dict.update(
        {"dataset": {"value": "diabetes"}, "experiment_run": {"value": f"{args.run}"}}
    )

    sweep_id = wandb.sweep(sweep_config, project=args.project)

    if not args.sweep_id:
        args.sweep_id = sweep_id

    # Initialize WandB sweep with provided sweep_id and project name
    wandb.agent(args.sweep_id, project=args.project, function=train)
