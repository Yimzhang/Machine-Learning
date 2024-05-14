import os

import time
import numpy as np
import scipy as sci
import scipy.io as sio
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow import keras


from numpy.random import seed
seed(100)


import matplotlib.pyplot as plt
from IPython.display import clear_output

# load raw data
raw = pd.read_csv('Implied_Volatility_Data_vFinal.csv')
# check the raw data
print("Size of the dataset (row, col): ", raw.shape)
raw.head(n=5)

raw['x1'] = raw['SPX Return']
raw['x2'] = raw['Time to Maturity in Year']
raw['x3'] = raw['Delta']


y = raw['Implied Volatility Change']
X = raw[['x1', 'x2', 'x3','SPX Return','Time to Maturity in Year','Delta']]

# Divide data into training set and test set(note that random seed is set)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

# Scale features based on Z-Score
scaler = StandardScaler()
scaler.fit(X_train)


X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

model = keras.models.Sequential([
Dense(20, activation="relu", input_shape=(3,)),
Dense(20, activation="relu"),
Dense(20, activation="relu"),
Dense(1)
])

model.summary()