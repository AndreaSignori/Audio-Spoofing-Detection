import pandas as pd

# Load the CSV file with semicolon delimiter
file_path = 'MFCC_TRAIN_FEATURES_LABELED.csv'
df = pd.read_csv(file_path, delimiter=';')

# Set the first row as header and drop the original header row
df.columns = df.iloc[0]
df = df.drop(df.index[0])

# Transform 'bonafide' to 1 and 'spoof' to 0 in the 'Label' column
df['Label'] = df['Label'].map({'bonafide': 1, 'spoof': 0})

# Display the first few rows to verify
print(df.head())
