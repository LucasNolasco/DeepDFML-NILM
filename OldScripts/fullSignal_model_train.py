import numpy as np
import h5py
from keras import Sequential
from keras import backend as K
from keras.models import load_model
from keras.layers import Conv1D, LeakyReLU, Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization
from random import shuffle
from keras.callbacks import TensorBoard
import tensorflow as tf

DATASET_PATH = "Synthetic_1_iHall.hdf5"
SIGNAL_LENGTH = 30768

# Carrega os dados do banco de dados e organiza, dividindo os dados do tamanho
def loadData(datasetPath):
    arq = h5py.File(datasetPath, "r")

    rawSamples = arq["i"]
    rawEvents = arq["events"]    
    rawLabels = arq["labels"]

    x, y = cutData(rawSamples, rawEvents, rawLabels, SIGNAL_LENGTH)

    arq.close()

    return x, y

# Pega os dados originais e faz recortes para diminuir o tamanho (Necessário para diminuir o tamanho do modelo)
def cutData(rawSamples, rawEvents, rawLabels, outputSignalLength):
    begin = True
    
    output_x = np.array([])
    output_y = np.array([])

    if 0 != (outputSignalLength % 4):
        print("Tamanho do sinal e tamanho do grid incompatíveis")
        while True:
            pass # Trap para segurar o programa após o erro
    
    gridLength = int(outputSignalLength / 4)

    for sample, event, label in zip(rawSamples, rawEvents, rawLabels):
        for initSignal in range(0, sample.shape[0], outputSignalLength):
            ev = np.argwhere(event[initSignal : initSignal + outputSignalLength] != 0)
            if len(ev) > 0:
                out = np.zeros(4 * 29)
                for i in range(0, 4):
                    if ev[0][0] >= i * gridLength and ev[0][0] < (i + 1) * gridLength:
                        out[i * 29 + label[0] + 3] = 1
                        out[i * 29] = (ev[0][0] - i * gridLength)/gridLength
                        if event[initSignal + ev[0][0]] == 1: # ON
                            out[i * 29 + 1] = 1
                        else: # OFF
                            out[i * 29 + 2] = 1
                    
                if begin:
                    output_x = np.copy(np.expand_dims(sample[: outputSignalLength], axis = 0))
                    output_y = np.copy(np.expand_dims(out, axis=0))
                    begin = False
                else:
                    output_x = np.vstack((output_x, np.expand_dims(sample[: outputSignalLength], axis = 0)))
                    output_y = np.vstack((output_y, np.expand_dims(out, axis=0)))
    
    return output_x, output_y

# divide os dados em partes de treino e teste
def splitData(x, y, ratio):
    '''
    z = zip(x, y)
    shuffle(z)
    x, y = zip(*z)
    '''

    x_train = np.array([])
    y_train = np.array([])
    x_test = np.array([])
    y_test = np.array([])

    beginTrain = False
    beginTest = False

    dict_labels = {}
    trainLength = round(ratio * 32) # TODO: modificar para algo automático
    for xi, yi in zip(x, y):
        lab = str(max(np.argmax(yi[3:29]), np.argmax(yi[32:58]), np.argmax(yi[61:87]), np.argmax(yi[90:116])))
        if lab not in dict_labels:
            dict_labels[lab] = 0

        if dict_labels[lab] < trainLength:
            if not beginTrain:
                x_train = np.expand_dims(xi, axis=0)
                y_train = np.expand_dims(yi, axis=0)
                beginTrain = True
            else:
                x_train = np.vstack((x_train, np.expand_dims(xi, axis=0)))
                y_train = np.vstack((y_train, np.expand_dims(yi, axis=0)))
        else:
            if not beginTest:
                x_test = np.expand_dims(xi, axis=0)
                y_test = np.expand_dims(yi, axis=0)
                beginTest = True
            else:
                x_test = np.vstack((x_test, np.expand_dims(xi, axis=0)))
                y_test = np.vstack((y_test, np.expand_dims(yi, axis=0)))

        dict_labels[lab] += 1

    return x_train, y_train, x_test, y_test

def f1(weights, i):
    weights = weights[i * 29].assign(5)

    return weights

def f2(weights, i):
    weigths = weights[i * 29].assign(0)
    for j in range(i * 29 + 3, (i + 1) * 29):
        weigths = weigths[j].assign(0)

    return weigths

def sumSquaredError(y_true, y_pred):
    weights = K.ones(29 * 4)
    for i in range(0, 4):
        '''
        if 0 != K.sum(y_true[i*29 + 1 : i * 29 + 3]): # Caso tenha algum evento nesse grid
            weights = weights[i * 29].assign(5)
        else:
            weigths = weights[i * 29].assign(0)
            for j in range(i * 29 + 3, (i + 1) * 29):
                weigths = weigths[j].assign(0)
        '''

        weights = tf.cond(K.sum(y_true[i*29 + 1 : i * 29 + 3]) != 0, f1(weights, i), f2(weights, i))

    return K.sum(K.square(y_true - y_pred) * weights, axis=-1)

def buildModel():
    model = Sequential()
    model.add(Conv1D(filters=60, kernel_size=9, input_shape=(SIGNAL_LENGTH, 1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.25))
    model.add(Conv1D(filters=40, kernel_size=9))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.25))
    model.add(Conv1D(filters=40, kernel_size=9))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.25))
    model.add(Conv1D(filters=20, kernel_size=9))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(1500))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dense(29 * 4, activation='relu'))
    model.compile(optimizer='sgd', loss = sumSquaredError, metrics=['accuracy'])

    return model

def main():
    x, y = loadData(DATASET_PATH)
    x_train, y_train, x_test, y_test = splitData(x, y, 0.75)

    print(x.shape, y.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    tensorboard_callback = TensorBoard(log_dir='./logs')

    model = buildModel() #load_model('single_loads_customLoss_12800_250.h5', custom_objects={'sumSquaredError': sumSquaredError}) 
    model.summary()

    fileEpoch = 0
    while fileEpoch < 500:
        model.fit(x=x_train, y=y_train, epochs=250, verbose=2, callbacks=[tensorboard_callback])
        fileEpoch += 250

        model.save('single_loads_customLoss_' + str(SIGNAL_LENGTH) + "_" + str(fileEpoch) + '.h5')

        score = model.evaluate(x=x_test, y=y_test)
        print("Evaluate score: ", score)

        y_predict = model.predict(x_test)

        totally_correct = 0
        detection_correct = 0
        classification_correct = 0
        on_off_correct = 0
        for predict, groundTruth in zip(y_predict, y_test):
            print(int(predict[0] * SIGNAL_LENGTH), np.argmax(predict[3:]), int(groundTruth[0] * SIGNAL_LENGTH), np.argmax(groundTruth[3:]))
            if abs(int(predict[0] * SIGNAL_LENGTH) - int(groundTruth[0] * SIGNAL_LENGTH)) < (1 * 256):
                detection_correct += 1
                if np.argmax(predict[3:]) == np.argmax(groundTruth[3:]) and np.argmax(predict[1:3]) == np.argmax(groundTruth[1:3]):
                    totally_correct += 1
            if np.argmax(predict[3:]) == np.argmax(groundTruth[3:]):
                classification_correct += 1
            if np.argmax(predict[1:3]) == np.argmax(groundTruth[1:3]):
                on_off_correct += 1

        print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/x_test.shape[0], 100 * detection_correct/x_test.shape[0], 100 * classification_correct/x_test.shape[0], 100 * on_off_correct/x_test.shape[0]))

if __name__ == '__main__':
    main()