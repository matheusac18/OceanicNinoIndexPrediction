import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error


def evaluate_forecasts(y, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in y]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print(rmse)


from keras import backend




dataset = pd.read_csv("results_corr_065_com_modulo_sem_polo_1anoAntes_mensal_final_duas_classes.csv")

#12meses:
#dataset = dataset[['eccentricity','eigenvector','links','global_average_link_distance','average_path_length','grauMedio','modularity','coreness','oni']]

#6meses:
#dataset = dataset[['eccentricity','eigenvector','links','global_average_link_distance','average_path_length','transitivity','oni']]

#todas:
dataset = dataset[['eccentricity','eigenvector','links','global_average_link_distance','average_path_length','grauMedio','modularity','coreness','transitivity','oni']]

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(-1, 1))

dataset = sc.fit_transform(dataset)

X_train = []
y_train = []
X_test = []
y_test = []

X = list()
Y = list()

windowSize = 48
horizon = 6


for i in range((horizon + windowSize), (dataset.shape[0])):
    X.append(np.asarray(dataset[i - (horizon + windowSize):i - horizon, :9]))
    Y.append(np.asarray(dataset[i - horizon:i,9]))

X, Y = np.array(X), np.array(Y)

Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
print("X Shape:")
print(X.shape)

print(Y.shape)
percent = 0.85

# divide the dataset
X_train, y_train, X_validation, y_validation, X_test, y_test = X[:int(X.shape[0] * (percent - 0.1)), :, :], Y[:int(
    Y.shape[0] * (percent - 0.1))], X[int(X.shape[0] * (percent - 0.1)):int(X.shape[0] * (percent)), :, :], Y[int(
    Y.shape[0] * (percent - 0.1)):int(Y.shape[0] * (percent))], X[int(X.shape[0] * percent):, :], Y[int(
    Y.shape[0] * percent):]

print(y_train.shape)
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense

#define regressor
regressor = Sequential()
regressor.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])))
regressor.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
regressor.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
regressor.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
regressor.add(MaxPooling1D(pool_size=2))
regressor.add(Flatten())
regressor.add(Dense(50, activation='relu'))
regressor.add(Dense(Y.shape[1]))
regressor.compile(optimizer='adam', loss='mse',metrics=['mae', 'mape', 'mse'])

history = regressor.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1]), epochs=100, shuffle=False,
        batch_size=4, validation_data=(
        X_validation, y_validation.reshape(y_validation.shape[0], y_validation.shape[1])))  # train the network

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

evaluate = regressor.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1]), batch_size=4)

print(regressor.metrics_names)
print(evaluate)

predicted_oni = regressor.predict(X_test)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
predicted_oni = predicted_oni.reshape(predicted_oni.shape[0], predicted_oni.shape[1])
evaluate_forecasts(y_test, predicted_oni, horizon, horizon)

for i in range(predicted_oni.shape[0]):
    plt.plot(y_test[i], color='black', label='Real ONI value')
    plt.plot(predicted_oni[i], color='green', label='Predict ONI value')
    plt.title('ONI Prediction')
    plt.xlabel('Time')
    plt.ylabel('ONI Value')
    plt.legend()
    plt.show()

predicted_oni = regressor.predict(X_validation)
for i in range(predicted_oni.shape[0]):
    plt.plot(y_validation[i], color='black', label='Real ONI value')
    plt.plot(predicted_oni[i], color='green', label='Predict ONI value')
    plt.title('ONI Prediction')
    plt.xlabel('Time')
    plt.ylabel('ONI Value')
    plt.legend()
    plt.show()
