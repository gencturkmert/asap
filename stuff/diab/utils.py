import numpy as np

def split(org_data, merged_array, label_array):
    total_length = sum(arr.shape[0] for arr in org_data)
    ratios = [arr.shape[0] / total_length for arr in org_data]
    split_indices = np.cumsum([int(ratio * merged_array.shape[0]) for ratio in ratios[:-1]])
    split_arrays = np.split(merged_array, split_indices)
    label_split_arrays = np.split(label_array, split_indices)
    return split_arrays, label_split_arrays

def merge(data):
  return np.concatenate(data, axis=0)

def flatten_data(data):
    return data.reshape(data.shape[0],-1)

def flatten_data_for_clients(clientsData):
    flattened_clients_data = [flatten_data(client_data) for client_data in clientsData]
    return flattened_clients_data

def back_to_image(data, dataset_input_shape):
    return data.reshape(data.shape[0], *dataset_input_shape)

def back_to_image_for_clients(clientsData):
    back_to_image_data = [back_to_image(client_data) for client_data in clientsData]
    return back_to_image_data