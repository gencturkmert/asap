import pandas as pd

# Load the dataset
file_path = 'diabet.csv'
data = pd.read_csv(file_path)

# Separate the majority and minority classes
majority_class = data[data['readmitted'] == 0]
minority_class = data[data['readmitted'] == 1]

# Duplicate the minority class by a factor of 2
minority_class_duplicated = pd.concat([minority_class] * 2, axis=0)

# Combine the datasets
balanced_data = pd.concat([majority_class, minority_class_duplicated], axis=0)

# Shuffle the data
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Output the class distribution
class_distribution = balanced_data['readmitted'].value_counts()

balanced_data_path = "balanced_diabet.csv"
balanced_data.to_csv(balanced_data_path, index=False)

class_distribution, balanced_data_path
