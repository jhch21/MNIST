from dataclasses import dataclass
from keras.models import sequential
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
data = mnist.load_data()
type(data)
(X_train, y_train), (X_test, y_test) = data
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

model = Sequential()
model.add(Dense(32, input_dim = 28 * 28, activation= 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs = 10, batch_size=20)
scores = model.evaluate(X_test, y_test)
print('Accuracy: ',scores[1]*100)