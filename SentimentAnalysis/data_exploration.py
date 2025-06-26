import pandas as pd

# Load the dataset without a header
df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)

# Assign column names
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Explore sentiment distribution
print(df['sentiment'].value_counts())
