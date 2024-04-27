import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def qty_skew_dist(X_train, y_train, num_clients, beta=0.5):
    clientsData = []
    clientsDataLabel = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    min_samples_per_client = 1
    remaining_samples = len(X_train) - num_clients * min_samples_per_client

    proportions = np.random.dirichlet(np.ones(num_clients) * beta, size=1)[0]
    proportions = np.round(proportions * remaining_samples).astype(int)
    proportions = proportions + min_samples_per_client  
    proportions[-1] = len(X_train) - np.sum(proportions[:-1])  

    start = 0
    for i in range(num_clients):
        end = start + proportions[i]
        client_indices = indices[start:end]
        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(y_train[client_indices])
        start = end

    return clientsData, clientsDataLabel

'''
def label_skew_dist(X_train, y_train, num_clients, num_classes, beta=0.5):
    clientsData = []
    clientsDataLabel = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    label_proportions = np.random.dirichlet(np.ones(num_classes) * beta, size=num_clients)

    for i in range(num_clients):
        client_indices = np.random.choice(indices, size=int(len(indices) / num_clients), replace=False)
        labels = np.concatenate([np.repeat(j, int(p * len(client_indices))) for j, p in enumerate(label_proportions[i])])
        np.random.shuffle(labels)
        labels = labels[:len(client_indices)]

        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(labels)

    return clientsData, clientsDataLabel
'''

def label_skew_dist(X_train, y_train, num_clients, num_classes, beta=0.5):
    clientsData = []
    clientsDataLabel = []
    
    # Create an array to keep track of indices for each class
    class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}

    # Generate proportions for label skew for each client
    label_proportions = np.random.dirichlet(np.ones(num_classes) * beta, size=num_clients)
    
    for i in range(num_clients):
        client_indices = []

        # Ensure at least one sample per class for each client
        for j in range(num_classes):
            if len(class_indices[j]) > 0:
                selected_index = np.random.choice(class_indices[j], 1, replace=False)
                client_indices.extend(selected_index)
                # Update the class indices to remove the selected one
                class_indices[j] = np.setdiff1d(class_indices[j], selected_index)

        # Calculate the remaining number of samples per class for this client based on proportions
        total_assigned = len(client_indices)  # Total already assigned as one per class
        additional_samples_per_class = [int(label_proportions[i][j] * (len(y_train) / num_clients)) for j in range(num_classes)]
        additional_samples_per_class = [max(0, x - 1) for x in additional_samples_per_class]  # Subtract one already assigned

        for j in range(num_classes):
            if additional_samples_per_class[j] > 0 and len(class_indices[j]) >= additional_samples_per_class[j]:
                selected_indices = np.random.choice(class_indices[j], additional_samples_per_class[j], replace=False)
                client_indices.extend(selected_indices)
                # Update the class indices to remove selected ones
                class_indices[j] = np.setdiff1d(class_indices[j], selected_indices)

        # Collect the actual data and labels for these indices
        clientsData.append(X_train[client_indices])
        clientsDataLabel.append(y_train[client_indices])

    return clientsData, clientsDataLabel


def feature_skew_dist(X_train, y_train, num_clients, sigma=0.5):
    clientsData = []
    clientsDataLabel = []
    num_features = X_train.shape[1]
    num_samples = X_train.shape[0]
    samples_per_client = max(1, num_samples // num_clients)  

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

def iid_dist(X_train, y_train, num_clients):
    """Distribute data equally to a number of clients in an IID manner."""
    clientsData = []
    clientsDataLabel = []
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    # Compute number of samples per client
    samples_per_client = len(indices) // num_clients
    remainder = len(indices) % num_clients

    start = 0
    for i in range(num_clients):
        if i < remainder:
            # Handle remainder by adding one more sample to some clients
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
    class_counts = np.zeros((num_clients, num_classes))

    for i, labels in enumerate(clientsDataLabel):
        for j, label in enumerate(class_labels):
            class_counts[i, j] = np.sum(labels == label)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(class_counts, cmap='Blues', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_clients))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels([f'Client {i+1}' for i in range(num_clients)])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Create text annotations.
    for i in range(num_clients):
        for j in range(num_classes):
            text = ax.text(j, i, int(class_counts[i, j]), ha="center", va="center", color="black")

    ax.set_title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Client')
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()
