import numpy as np
from norms import *
from utils import *
import argparse
import numpy as np
import csv
import tensorflow as tf
from skew import *
import wandb
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


"""# Global Constants (Dataset Specific)"""
# global variables
BATCH_SIZE = 8
EPOCH = 250
LEARNING_RATE = 0.001
NORMALIZATION_LAYER = ""
NUM_CLIENTS = 10
EARLY_STOPPING_PATIENCE = 20
DATASET_INPUT_SHAPE = (20, 1)


def initialize_globals():
    global X_trains_fed, Y_trains_fed, X_test_fed, Y_test_fed, X_val_fed, Y_val_fed
    global global_weights, best_r2, weights, best_loss

    # Dataset-related globals
    X_trains_fed = np.zeros((1,))
    Y_trains_fed = np.zeros((1,))
    X_test_fed = np.zeros((1,))
    Y_test_fed = np.zeros((1,))
    X_val_fed = np.zeros((1,))
    Y_val_fed = np.zeros((1,))

    # Model-related globals
    global_weights = np.zeros((1,))
    best_r2 = 0.0
    weights = np.array([])
    best_loss = float("inf")


"""# Dataset Retrieval"""


def load_data(data_dir=""):
    # Load X data
    with open(f"/content/drive/MyDrive/asap/stuff/parc/x.csv", mode="r") as file:
        reader = csv.reader(file)
        x_data = np.array(list(reader), dtype=float)

    # Load Y data
    with open(f"/content/drive/MyDrive/asap/stuff/parc/y.csv", mode="r") as file:
        reader = csv.reader(file)
        y_data = np.array(list(reader), dtype=float)

    y_data = y_data[:, 1]
    
    print("x_data shape:", x_data.shape)
    print("y_data shape:", y_data.shape)

    # Split data into training, testing, and validation sets
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
            X_data, Y_data, num_clients, sigma=0.3, batch_size=BATCH_SIZE
        )
    elif skew_type == "feature_0.7":
        clientsData, clientsDataLabels = feature_skew_dist(
            X_data, Y_data, num_clients, sigma=0.7, batch_size=BATCH_SIZE
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
    model.add(tf.keras.layers.Dense(units=128, activation="relu", input_shape=DATASET_INPUT_SHAPE))

    if NORMALIZATION_LAYER == "batch_norm":
        model.add(tf.keras.layers.BatchNormalization())
    elif NORMALIZATION_LAYER == "instance_norm":
        model.add(tf.keras.layers.GroupNormalization(groups=1))
    elif NORMALIZATION_LAYER == "group_norm":
        model.add(tf.keras.layers.GroupNormalization(groups=4))
    elif NORMALIZATION_LAYER == "layer_norm":
        model.add(tf.keras.layers.LayerNormalization())

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=32, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=16, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))
    return model


"""# Flower Client (Dataset Specific)"""


# Required for early stopping


def get_results():
    """
    At the end of the federated learning process, calculates results and returns
    Test F1 Score
    Test Loss
    Test Accuracy
    Test Precision
    Test Recall
    """
    global X_test_fed, Y_test_fed, global_weights

    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
    )
    model.set_weights(global_weights)

    test_loss = model.evaluate(X_test_fed, Y_test_fed, batch_size=BATCH_SIZE, verbose=0)

    y_pred = model.predict(X_test_fed)
    y_pred = np.squeeze(y_pred)  # Removes dimensions of size 1


    test_mae = mean_absolute_error(Y_test_fed, y_pred)
    test_r2 = r2_score(Y_test_fed, y_pred)

    return test_loss, test_mae, test_r2


def federated_train(x, y, num_clients):
    global global_weights, X_val_fed, Y_val_fed, best_r2, best_loss

    # Initialize client models and global model
    models = [get_model() for _ in range(num_clients)]
    global_model = get_model()

    # Initialize global weights
    global_weights = global_model.get_weights()

    # Initialize history and early stopping parameters
    history = {
        "val_loss": [],
        "val_mae": [],
        "val_r2": [],
        "distributed_train_loss": [],
        "centralized_train_loss": [],
    }
    patience_counter = 0

    for epoch in range(EPOCH):
        client_weights = []
        client_train_losses = []

        # Training for each client
        for i in range(num_clients):
            # Set model to current global weights
            models[i].set_weights(global_weights)

            # Compile the client model
            models[i].compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="mean_squared_error",
            )

            # Train the client model on its respective data
            history_per_client = models[i].fit(
                x[i], y[i], batch_size=BATCH_SIZE, epochs=1, verbose=0
            )

            # Record client training loss
            client_train_losses.append(history_per_client.history["loss"][-1])

            # Collect client weights
            client_weights.append(models[i].get_weights())

        # Calculate distributed training loss (average of client losses)
        avg_train_loss = np.mean(client_train_losses)
        history["distributed_train_loss"].append(avg_train_loss)

        # Apply FedAvg to aggregate client weights
        global_weights = fedavg(client_weights,x)

        # Set global weights for the global model
        global_model.set_weights(global_weights)
        global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="mean_squared_error",
        )

        # Evaluate centralized training loss (global model on all client training data)
        combined_x = np.concatenate(x, axis=0)
        combined_y = np.concatenate(y, axis=0)
        centralized_train_loss = global_model.evaluate(
            combined_x, combined_y, batch_size=BATCH_SIZE, verbose=0
        )
        history["centralized_train_loss"].append(centralized_train_loss)

        val_loss = global_model.evaluate(
            X_val_fed, Y_val_fed, batch_size=BATCH_SIZE, verbose=0
        )
        
        y_pred = global_model.predict(X_val_fed)
        y_pred = np.squeeze(y_pred)  # Removes dimensions of size 1
        
        print("pred shape:", y_pred.shape)
        print("Y_val_fed shape:", Y_val_fed.shape)
        
        val_mae = mean_absolute_error(Y_val_fed, y_pred)
        val_r2 = r2_score(Y_val_fed, y_pred)

        # Store validation metrics in history
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)

        # Early stopping logic
        if val_loss < best_loss - 0.00001:
            best_loss = val_loss
            best_r2 = val_r2
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    test_loss, test_mae, test_r2 = get_results()
    return history, test_loss, test_mae, test_r2


def fedavg(client_weights, client_data):
    """
    Implements the FedAvg algorithm to aggregate client weights.
    Aggregation is weighted based on the number of samples per client.

    Args:
        client_weights (list): List of model weights from each client.
        client_data (list): List of client datasets (x), used to calculate weights.

    Returns:
        list: Weighted average of client weights.
    """
    # Calculate the total number of samples across all clients
    total_samples = sum([len(data) for data in client_data])

    # Calculate the weight for each client based on their sample size
    client_sample_weights = [len(data) / total_samples for data in client_data]

    # Perform weighted averaging of client weights
    weighted_avg_weights = []
    for layer in range(len(client_weights[0])):
        weighted_layer = sum(
            client_sample_weights[i] * client_weights[i][layer]
            for i in range(len(client_weights))
        )
        weighted_avg_weights.append(weighted_layer)

    return weighted_avg_weights


"""# Training and Saving the Results"""


def train(config=None):
    with wandb.init(config=config):

        config = wandb.config
        tf.keras.backend.clear_session()

        global NUM_CLIENTS
        global X_test_fed
        global Y_test_fed

        global X_val_fed
        global Y_val_fed

        NUM_CLIENTS = config["a_num_clients"]

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
        history, test_loss, test_mae, test_r2 = federated_train(
            normalizedData, clientLabels, config["a_num_clients"]
        )

        t2 = time.perf_counter(), time.process_time()
        t = t2[1] - t1[1]

        # Use global variables for best metrics and weights
        global best_r2, global_weights, best_loss

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
        wandb.log({"test_r2": test_r2})
        wandb.log({"test_mae": test_mae})
        wandb.log({"validation_loss": best_loss})
        wandb.log({"validation_r2": best_r2})

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

        validation_r2_table = wandb.Table(
            data=[[i, r2] for i, r2 in enumerate(history["val_r2"])],
            columns=["Epoch", "Validation R2"],
        )
        wandb.log(
            {
                "validation_r2": wandb.plot.line(
                    validation_r2_table,
                    "Epoch",
                    "Validation R2",
                    title="Validation R2 vs Epoch",
                )
            }
        )

        best_r2 = 0.0
        global_weights = np.array([])
        best_loss = float("inf")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run WandB sweep with custom sweep_id and project name."
    )
    parser.add_argument("--sweep_id", type=str, required=False, help="WandB Sweep ID")
    parser.add_argument("--project", type=str, required=True, help="WandB Project Name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    initialize_globals()

    wandb.login()

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
        {"dataset": {"value": "parcinson"}, "experiment_run": {"value": "1"}}
    )

    sweep_id = wandb.sweep(sweep_config, project=args.project)

    if not args.sweep_id:
        args.sweep_id = sweep_id

    # Initialize WandB sweep with provided sweep_id and project name
    wandb.agent(args.sweep_id, project=args.project, function=train)
