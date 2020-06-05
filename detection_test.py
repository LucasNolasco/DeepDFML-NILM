import numpy as np
import h5py
from keras import Sequential
from keras.layers import Conv1D, LeakyReLU, Dense, Flatten
from random import shuffle

DATASET_PATH = "Synthetic_1_iHall.hdf5"

# Carrega os dados do banco de dados e organiza, dividindo os dados do tamanho
def loadData(datasetPath):
    arq = h5py.File(datasetPath, "r")

    rawSamples = arq["i"]
    rawEvents = arq["events"]    
    rawLabels = arq["labels"]

    x, y = cutData(rawSamples, rawEvents, rawLabels, 38400)

    arq.close()

    return x, y

# Pega os dados originais e faz recortes para diminuir o tamanho (Necessário para diminuir o tamanho do modelo)
def cutData(rawSamples, rawEvents, rawLabels, outputSignalLength):
    begin = True

    output_x = np.array([])
    output_y = np.array([])

    for sample, event, label in zip(rawSamples, rawEvents, rawLabels):
        initIndex = 0
        while (initIndex + outputSignalLength) < len(sample):
            ev = np.argwhere(event[initIndex : initIndex + outputSignalLength] != 0) # Verifica se há algum evento nesse recorte
            if len(ev) > 0:
                #print(ev[0][0], label)
                if begin:
                    output_x = np.copy(np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0))
                    output_y = np.copy(np.array([ev[0][0], label]))
                    begin = False
                else:
                    output_x = np.vstack((output_x, np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0)))
                    output_y = np.vstack((output_y, np.array([ev[0][0], label])))

            initIndex += outputSignalLength
    
    return output_x, output_y

# divide os dados em partes de treino e teste
def splitData(x, y, ratio):
    z = zip(x, y)
    shuffle(z)
    x, y = zip(*z)

    trainLength = round(ratio * len(x))
    x_train = x[:trainLength]
    y_train = y[:trainLength]
    x_test = x[trainLength:]
    y_test = y[trainLength:]

    return x_train, y_train, x_test, y_test

def buildModel():
    model = Sequential()
    model.add(Conv1D(filters=60, kernel_size=9, input_shape=(None,38400)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Conv1D(filters=40, kernel_size=9))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Conv1D(filters=40, kernel_size=9))
    model.add(LeakyReLU(alpha = 0.1))
    #model.flatten()
    model.add(Dense(20))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='sgd', loss = "mean_squared_error", metrics=['accuracy'])

    return model

def main():
    x, y = loadData(DATASET_PATH)

    print(x.shape, y.shape)

    x_train, y_train, x_test, y_test = splitData(x, y, 0.8)

    model = buildModel()
    model.fit(x=x_train, y=y_train, verbose=2)

if __name__ == '__main__':
    main()