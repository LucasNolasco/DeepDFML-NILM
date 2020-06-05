from keras import Sequential
from keras.layers import Conv1D

model = Sequential()
model.add(Conv1D(filters=60, kernel_size=9, activation='relu', input_shape=(None,481000)))
