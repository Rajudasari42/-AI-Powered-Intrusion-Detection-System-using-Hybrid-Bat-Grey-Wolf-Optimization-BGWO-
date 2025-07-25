import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    df = pd.read_csv(path)

    # Encode categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop('class', axis=1)
    y = df['class']
    return X, y
