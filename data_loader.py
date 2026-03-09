
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    data = pd.read_csv(path)
    features = data[['cycle','temperature','voltage','current']]
    target = data['rul']
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, target.values

def create_sequences(X, y, seq_length=5):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)
