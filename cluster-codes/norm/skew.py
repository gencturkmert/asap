import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

import numpy as np

import numpy as np

def qty_skew_dist(X_train, y_train, num_clients, beta=0.5, batch_size=1):
    clientsData = []
    clientsDataLabel = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    # Ensure each client receives at least one sample
    min_samples_per_client = 1
    if batch_size > 1:
        min_samples_per_client = batch_size

    # Allocate the minimum required samples first
    initial_samples = num_clients * min_samples_per_client
    if initial_samples > len(X_train):
        raise ValueError("Not enough samples to distribute the minimum required per client. Increase data size or reduce number of clients.")

    start = 0
    for i in range(num_clients):
        end = start + min_samples_per_client
        client_indices = indices[start:end]
        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(y_train[client_indices])
        start = end

    # Use remaining samples for skewed distribution
    remaining_samples = len(X_train) - initial_samples
    remaining_indices = indices[initial_samples:]  # Indices of remaining samples

    if remaining_samples > 0:
        # Calculate proportions for the remaining samples using Dirichlet distribution
        proportions = np.random.dirichlet(np.ones(num_clients) * beta, size=1)[0]
        proportions = np.round(proportions * remaining_samples).astype(int)
        proportions[-1] = remaining_samples - np.sum(proportions[:-1])  # Adjust the last proportion to sum exactly to remaining_samples

        # Distribute remaining samples
        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices = remaining_indices[start:end]
            clientsData[i] = np.concatenate((clientsData[i], X_train[client_indices]))
            clientsDataLabel[i] = np.concatenate((clientsDataLabel[i], y_train[client_indices]))
            start = end

    return clientsData, clientsDataLabel



import numpy as np

def label_skew_dist(X_train, y_train, num_clients, num_classes, beta=5, batch_size=1):
    clientsData = []
    clientsDataLabel = []

    # Create an array to keep track of indices for each class
    class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}

    # Generate proportions for label skew for each client across classes
    label_proportions = np.random.dirichlet(np.ones(num_classes) * beta, size=num_clients)
    
    # Calculate total samples per client proportionally and ensure at least batch_size
    total_samples_per_client = np.round(label_proportions * len(y_train)).astype(int).sum(axis=1)
    total_samples_per_client = np.maximum(total_samples_per_client, batch_size * np.ones(num_clients).astype(int))

    # Normalize if the total exceeds the number of available samples
    while np.sum(total_samples_per_client) > len(y_train):
        total_samples_per_client[np.argmax(total_samples_per_client)] -= 1

    # Shuffle all indices to distribute
    shuffled_indices = np.random.permutation(len(y_train))

    # Distribute indices according to the calculated totals for each client
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + total_samples_per_client[i]
        client_indices = shuffled_indices[start_idx:end_idx]
        start_idx = end_idx

        # Collect the actual data and labels for these indices
        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(y_train[client_indices])

    return clientsData, clientsDataLabel


def feature_skew_dist(X_train, y_train, num_clients, sigma=0.5, batch_size=1):
    clientsData = []
    clientsDataLabel = []
    num_features = X_train.shape[1]
    num_samples = X_train.shape[0]
    samples_per_client = max(batch_size, num_samples // num_clients)  

    for i in range(num_clients):
        feature_weights = np.abs(np.random.normal(0, sigma, num_features))
        feature_weights /= feature_weights.sum()
        skewed_features = np.dot(X_train, np.diag(feature_weights))

        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        if i == num_clients - 1:  
            end_idx = num_samples

        client_indices = np.random.choice(np.arange(start_idx, end_idx), size=min(samples_per_client, end_idx - start_idx), replace=False)
        clientsData.append(skewed_features[client_indices])
        clientsDataLabel.append(y_train[client_indices])

    return clientsData, clientsDataLabel

def iid_dist(X_train, y_train, num_clients, batch_size=1):
    """Distribute data equally to a number of clients in an IID manner."""
    clientsData = []
    clientsDataLabel = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    samples_per_client = max(batch_size, len(indices) // num_clients)
    remainder = len(indices) % num_clients

    start = 0
    for i in range(num_clients):
        if i < remainder:
            end = start + samples_per_client + 1
        else:
            end = start + samples_per_client
        
        client_indices = indices[start:end]
        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(y_train[client_indices])
        start = end

    return clientsData, clientsDataLabel

def plot_class_distribution_per_client(clientsDataLabel, class_labels, title):
    """Plot the distribution of classes for each client."""
    num_clients = len(clientsDataLabel)
    num_classes = len(class_labels)
    # Adding an extra column for the total counts
    class_counts = np.zeros((num_clients, num_classes + 1))

    for i, labels in enumerate(clientsDataLabel):
        for j, label in enumerate(class_labels):
            class_counts[i, j] = np.sum(labels == label)
        # Compute the total and assign it to the last column
        class_counts[i, -1] = np.sum(class_counts[i, :-1])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(class_counts, cmap='Blues', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_clients))
    xlabels = class_labels + ["Total"]
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create text annotations for class counts and totals.
    for i in range(num_clients):
        for j in range(num_classes + 1):  # Adjust to include total in annotations
            count = int(class_counts[i, j])
            ax.text(j, i, count, ha="center", va="center", color="black" if count > class_counts.max()/2 else "white")

    ax.set_title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Client')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def plot_feature_skew(X_train, clientsData, num_features_to_plot=3, num_clients_to_plot=3):
    num_clients = len(clientsData)
    num_features = X_train.shape[1]
    print(num_clients,num_features)

    features_to_plot = np.random.choice(range(num_features), size=min(num_features_to_plot, num_features), replace=False)
    clients_to_plot = np.random.choice(range(num_clients), size=min(num_clients_to_plot, num_clients), replace=False)

    fig, axes = plt.subplots(len(clients_to_plot), len(features_to_plot), figsize=(12, 8))
    for i, client_idx in enumerate(clients_to_plot):
        for j, feature_idx in enumerate(features_to_plot):
            ax = axes[i, j]
            original_feature = X_train[:, feature_idx]
            skewed_feature = clientsData[client_idx][:, feature_idx]
            ax.scatter(original_feature, skewed_feature, alpha=0.5)
            ax.set_xlabel('Original Feature')
            ax.set_ylabel('Skewed Feature')
            ax.set_title(f'Client {client_idx + 1}, Feature {feature_idx + 1}')
    plt.tight_layout()
    plt.show()
