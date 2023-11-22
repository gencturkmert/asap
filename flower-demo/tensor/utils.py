import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def partition_dataset(noise: int, clicount: int, index: int):
    #Adds noise to given val -> val = ((noise + 100) * val / 100)) 
    def add_noise(val):
        return val * ((100 + noise) /100)


    # Load California Housing dataset
    california_housing = fetch_california_housing()

    housing = pd.DataFrame(california_housing.data,columns=california_housing.feature_names)

    #Min Max Norm. using noised values
    scaled_features = (housing - add_noise(housing.min())) / add_noise((housing.max()) - add_noise(housing.min()))

    x, y = scaled_features.values, california_housing.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Determine the size of each partition
    partition_size = len(x_train) // clicount

    # Select the data corresponding to the index-th partition
    start_idx = index * partition_size
    end_idx = (index + 1) * partition_size if index < clicount - 1 else len(x_train)

    partitioned_x_train = x_train[start_idx:end_idx]
    partitioned_y_train = y_train[start_idx:end_idx]

    return (partitioned_x_train, partitioned_y_train), (x_test,y_test)

    


    