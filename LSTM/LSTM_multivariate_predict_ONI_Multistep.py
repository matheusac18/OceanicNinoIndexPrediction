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




dataset = pd.read_csv("results_corr_065_com_modulo_sem_polo_1anoAntes_mensal_final.csv")

dataset = dataset[['transitivity','eccentricity','eigenvector','links','coreness','global_average_link_distance','modularity','oni']]

#dataset = dataset[['grauMedio', 'oni']]

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
    X.append(np.asarray(dataset[i - (horizon + windowSize):i - horizon, :7]))
    Y.append(np.asarray(dataset[i - horizon:i,7]))

X, Y = np.array(X), np.array(Y)

Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
print(X.shape)
print(Y.shape)
percent = 0.85

# divide the dataset
X_train, y_train, X_validation, y_validation, X_test, y_test = X[:int(X.shape[0] * (percent - 0.1)), :, :], Y[:int(
    Y.shape[0] * (percent - 0.1))], X[int(X.shape[0] * (percent - 0.1)):int(X.shape[0] * (percent)), :, :], Y[int(
    Y.shape[0] * (percent - 0.1)):int(Y.shape[0] * (percent))], X[int(X.shape[0] * percent):, :], Y[int(
    Y.shape[0] * percent):]

print(y_train.shape)

# building the LSTM network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import plot_model
from keras import optimizers
from keras.layers import TimeDistributed
import keras


regressor = Sequential()  # build th sequential stack of layers
regressor.add(Dense(units=X_train.shape[2], input_shape=(X_train.shape[1], X_train.shape[2])))

regressor.add(LSTM(units=25))
regressor.add(Dropout(0.2))

regressor.add(Dense(y_train.shape[1]))

regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape', 'mse'])

# plot the build network
#plot_model(regressor, show_layer_names=True, to_file='model.png', rankdir='LR', show_shapes=True)

# trains the network
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
