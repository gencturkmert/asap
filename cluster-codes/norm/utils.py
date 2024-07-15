import numpy as np


def split(org_data, merged_array):
    lens = [len(org_data[i]) for i in range(len(org_data))]
    split_arrays = np.split(merged_array, np.cumsum(lens)[:-1])
    return split_arrays

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