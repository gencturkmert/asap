import pandas as pd

# Read the dataset
df = pd.read_csv('parkinsons_updrs.csv')

# Filter rows where 'test_time' > 0
df = df[df['test_time'] > 0]

# Separate y (columns 'motor_UPDRS' and 'total_UPDRS') and drop them from df
y = df[['motor_UPDRS', 'total_UPDRS']].values
x = df.drop(['motor_UPDRS', 'total_UPDRS'], axis=1).values

# Save the cleaned x and y to separate CSV files
pd.DataFrame(x).to_csv('x.csv', index=False, header=False)
pd.DataFrame(y).to_csv('y.csv', index=False, header=False)
