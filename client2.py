# ==============================================================================
# Importing libraries
# ==============================================================================
import numpy as np
import pandas as pd
import flwr as fl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras import layers

# ==============================================================================
# Loading dataset
# ==============================================================================
data = pd.read_csv('/home/zeeshan/Documents/1test_purpose/medical/data-ori.csv')
data = data.iloc[2200:, :]

# ==============================================================================
# Prepprocessing dataset
# ==============================================================================
def preprocess_inputs(df):
    df = df.copy()
    
    # Binary encoding
    df['SEX'] = df['SEX'].replace({'F': 0, 'M': 1})
    df['SOURCE'] = df['SOURCE'].replace({'out': 0, 'in': 1})
    
    # Split df into X and y
    y = df['SOURCE']
    X = df.drop('SOURCE', axis=1)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess_inputs(data)

# ==============================================================================
# Defining model with parameters
# ==============================================================================
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[len(x_train.keys())]),
    layers.Dropout(0.2),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================================================================
# Defining client functions
# ==============================================================================
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        h = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
        hist = h.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Eval accuracy : ", accuracy)
        return loss, len(x_test), {"accuracy": float(accuracy)}
    
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=CifarClient())
