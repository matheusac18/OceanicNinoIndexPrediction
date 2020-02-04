import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado.csv")

dataset = dataset[['transitivity','modularity','coreness','eigenvector','oni']]

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

dataset = sc.fit_transform(dataset)

"""
#20% to test and 80% to train
train_set = dataset[:int(len(dataset)*0.7)]
test_set = dataset[int(len(dataset)*0.7):]

print("Train Set: %d Test Set %d" %(train_set.shape[0], test_set.shape[0]))
"""
X_train = []
y_train = []
X_test = []
y_test = []

X = []
Y = []

for i in range(12,(dataset.shape[0])):
    X.append(dataset[i-12:i,:])
    Y.append(dataset[i,4])#get the oni value

X, Y = np.array(X), np.array(Y)


"""
for i in range(12, train_set.shape[0]):
    print(i)
    X_train.append(train_set[i-12:i,:])
    y_train.append(train_set[i,4])#get the oni value

for i in range(12, test_set.shape[0]):
    print(i)
    X_test.append(test_set[i-12:i,:])
    y_test.append(test_set[i,4])#get the oni value

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
"""
#divide the dataset in 70% to train and 30% to test
X_train, y_train, X_test, y_test = X[:int(X.shape[0]*0.7),:,:], Y[:int(Y.shape[0]*0.7)], X[int(X.shape[0]*0.7):,:,:], Y[int(Y.shape[0]*0.7):]
print(X_test.shape)

#building the LSTM network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import plot_model

regressor = Sequential()#build th sequential stack of layers

regressor.add(Dense(units = 5, input_shape = (X_train.shape[1], 5)))#number of neurons in th input layer according the number of features

#Four layers LSTM with 50 neurons and 4 layers of Dropout to prevent overfitting

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))#output layer

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae','mape','mse'])

#plot the build network
plot_model(regressor,show_layer_names=True,to_file='model.png', rankdir='LR', show_shapes=True)

#trains the network 
#history = regressor.fit(X_train, y_train, epochs = 5000, batch_size = 100)#train the network
history = regressor.fit(X_train, y_train, epochs = 200,  batch_size = 100, validation_data=(X_test, y_test))#train the network

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

predicted_oni = regressor.predict(X_test)

"""
#transform the values predicts to original format
predicted_oni = sc.inverse_transform(predicted_oni)

#transform the real ONI values to original format
y_test = sc.inverse_transform(y_test)
"""

plt.plot(y_test, color = 'black', label = 'Real ONI value')
plt.plot(predicted_oni, color = 'green', label = 'Predict ONI value')
plt.title('ONI Prediction')
plt.xlabel('Time')
plt.ylabel('ONI Value')
plt.legend()
plt.show()


