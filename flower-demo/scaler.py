import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DatasetScaler:
    def __init__(self, df):
        self.global_min = df.min()
        self.global_max = df.max()
        self.scaler = MinMaxScaler()

    def scale_dataset(self, fraction_df):
        features = fraction_df.drop("target_column", axis=1)
        labels = fraction_df["target_column"]

        scaled_features = (features - self.global_min) / (self.global_max - self.global_min)
        
        scaled_df = pd.DataFrame(data=np.c_[scaled_features, labels], columns=fraction_df.columns)

        return scaled_df
