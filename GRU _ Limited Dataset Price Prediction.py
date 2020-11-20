#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd


# In[94]:


from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import GRU,LSTM
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis


# In[95]:


ds =  pd.read_csv('../input/limited-nyse-price/prices-split-adjusted.csv' , header=0)
#Adjusted prices used for absolute consumer value
ds.head(5)


# In[96]:


ms = ds[ds['symbol']=='MSFT']
ms_stock_prices = ms.close.values.astype('float32')


# In[97]:


ms_stock_prices


# In[98]:


ms_stock_prices.reshape(1762, 1)


# In[99]:


ms_stock_prices.shape


# In[100]:


plt.plot(ms_stock_prices)
plt.show()


# In[101]:


sc = MinMaxScaler(feature_range=(0,1))
ms_dataset = sc.fit_transform(ms_stock_prices.reshape(-1,1))


# In[102]:


ms_dataset.shape


# In[103]:


train_size = int(0.80 * len(ms_dataset))
test_size = len(ms_dataset)-train_size


# In[104]:


train , test = ms_dataset[0:train_size,:], ms_dataset[train_size:len(ms_dataset),:]
print(len(train))
print(len(test))


# In[105]:


#lookback is Number of steps to check for output
def create_dataset(dataset, look_back=5):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# In[106]:


#since singular data 
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[107]:


trainX.shape
sv_trainX = trainX
sv_trainY = trainY
sv_testX = testX
sv_testY = testY


# In[108]:


trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
trainX.shape


# In[109]:


trainX.shape
trainX


# In[110]:


testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


# In[111]:


print(testX.shape)
testY.reshape(-1,1)


# In[112]:


recc_model = Sequential()


# In[113]:


#RNN Model
recc_model.add(GRU(input_shape=(trainX.shape[1],1), units=100, return_sequences=True))
#recc_model.add(Activation('relu'))
recc_model.add(Dropout(0.2))
#hidden_1
recc_model.add(GRU(units=50,return_sequences=True))
#recc_model.add(Activation('relu'))
recc_model.add(Dropout(0.2))
#hidden_2
#recc_model.add(Dense(units=50, activation = 'relu'))
#recc_model.add(GRU(units=50,return_sequences=True))
#recc_model.add(Dropout(0.2))
#Fourth Hidden with no return sequences
recc_model.add(GRU(units=50))
#recc_model.add(Dropout(0.2))
#output Layer
recc_model.add(Dense(units=1))
#recc_model.add(Activation('linear'))


# In[114]:


recc_model.summary()
recc_model.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])


# In[115]:


history = recc_model.fit(
    trainX,
    trainY,
    batch_size=64,
    epochs=30,
    validation_split=0.05)


# In[116]:


recc_model_lstm = Sequential()
#RNN Model
recc_model_lstm.add(LSTM(input_shape=(trainX.shape[1],1), units=100, return_sequences=True))
recc_model_lstm.add(Activation('relu'))
recc_model_lstm.add(Dropout(0.2))
#hidden_1_lstm
recc_model_lstm.add(LSTM(units=50,return_sequences=False))
recc_model_lstm.add(Activation('relu'))
recc_model_lstm.add(Dropout(0.2))
#hidden_2
#recc_model.add(Dense(units=50, activation = 'relu'))
#recc_model_lstm.add(LSTM(units=50,return_sequences=True))
#recc_model_lstm.add(Activation('relu'))
#recc_model_lstm.add(Dropout(0.2))
#Fourth Hi_lstmdden with no return sequences
#recc_model_lstm.add(Dropout(0.2))
#output La_lstmyer
recc_model_lstm.add(Dense(units=1))
#recc_model_lstm.add(Activation('linear'))
recc_model_lstm.summary()
recc_model_lstm.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])
history_lstm = recc_model_lstm.fit(
    trainX,
    trainY,
    batch_size=64,
    epochs=30,
    validation_split=0.05)


# In[117]:


from sklearn.svm import SVR
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(sv_trainX, sv_trainY)
print ("SVR Score =" , clf.score(sv_trainX, sv_trainY, sample_weight=None))
pred = clf.predict(sv_testX)
pred = pred.reshape(347,1)
pred.reshape(1,-1)
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(sv_trainX, sv_trainY)
print ("KNN Score =" , neigh.score(sv_trainX, sv_trainY, sample_weight=None))
pred_KNN = neigh.predict(sv_testX)
pred_KNN = pred_KNN.reshape(347,1)


# In[127]:


plt.figure(figsize=(10, 6), dpi=100)
plt.plot(history.history['loss'], label='GRU train', color='brown')
plt.plot(history.history['val_loss'], label='GRU test', color='blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Training and Validation loss')
plt.show()
predicted_stock_price = recc_model.predict(testX)
predicted_lstm = recc_model_lstm.predict(testX)
predicted_stock_price[:10]


# In[119]:


predicted_stock_price.shape
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_lstm = sc.inverse_transform(predicted_lstm)
pred_sv = sc.inverse_transform(pred)
pred_KNN_l = sc.inverse_transform(pred_KNN)
real_prices = ms_dataset[train_size:]
real_prices = sc.inverse_transform(real_prices)
real_prices
import math


# In[120]:


plt.plot(real_prices, color = 'red',label = 'Real Prices')
plt.plot(predicted_stock_price, color = 'yellow', label = 'GRU')
plt.plot(predicted_lstm, color = 'green', label = 'LSTM')
plt.plot(pred_sv, color= 'black', label = 'Support Regressor')
plt.plot(pred_KNN_l, color = 'blue', label = 'KNN')
plt.title('Price Prediction [Reduced]')
plt.legend()
plt.show()


# In[121]:


train_acc = recc_model.evaluate(trainX, trainY, verbose=0)
test_acc = recc_model.evaluate(testX, testY, verbose=0)


# In[129]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




