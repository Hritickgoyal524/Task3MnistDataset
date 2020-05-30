#!/usr/bin/env python
# coding: utf-8
from keras.datasets import mnist
dataset = mnist.load_data('mymnist.db')
train , test = dataset
X_test , y_test = test
X_train , y_train = train
X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train_1d.shape
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat=to_categorical(y_test)
y_train_cat
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=1000, input_dim=28*28, activation='relu'))
model.summary()
model.add(Dense(units=515, activation='relu'))
model.add(Dense(units=250, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.summary()
model.add(Dense(units=10, activation='softmax'))
from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
h = model.fit(X_train, y_train_cat,epochs=3)
print(h.history['accuracy'][-1])
with open("/mlops2/accuracy.text","+w") as f3:
     f3.write(h.history['accuracy'][-1])




