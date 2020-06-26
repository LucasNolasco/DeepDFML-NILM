import numpy as np
import h5py
from keras import Sequential
from keras import backend as K
from keras.models import load_model
from keras.layers import Conv1D, LeakyReLU, Dense, Flatten, MaxPooling1D, Dropout
from random import shuffle
from keras.callbacks import TensorBoard
import logging

DATASET_PATH = "Synthetic_1_iHall.hdf5"
MODEL_NAME = 'AddFinalConvLayer_20/single_loads_customLoss_12800_750.h5'
SIGNAL_LENGTH = 12800

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

    for sample, event, label in zip(rawSamples, rawEvents, rawLabels):
        initIndex = 0
        while (initIndex + outputSignalLength) < len(sample):
            ev = np.argwhere(event[initIndex : initIndex + outputSignalLength] != 0) # Verifica se há algum evento nesse recorte
            if len(ev) > 0:
                out = np.zeros(29)
                out[label[0] + 3] = 1
                out[0] = ev[0][0]/SIGNAL_LENGTH
                if event[initIndex + ev[0][0]] == 1: # ON
                    out[1] = 1
                else: # OFF
                    out[2] = 1
                
                if begin:
                    output_x = np.copy(np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0))
                    output_y = np.copy(np.expand_dims(out, axis=0))
                    begin = False
                else:
                    output_x = np.vstack((output_x, np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0)))
                    output_y = np.vstack((output_y, np.expand_dims(out, axis=0)))

            initIndex += outputSignalLength
    
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
        lab = str(np.argmax(yi[3:]))
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

def sumSquaredError(y_true, y_pred):
    weights = K.ones(29)
    weights = weights[0].assign(5)
    return K.sum(K.square(y_true - y_pred) * weights, axis=-1)

def main():
    x, y = loadData(DATASET_PATH)
    _, _, x_test, y_test = splitData(x, y, 0.75)

    model = load_model("Weights/" + MODEL_NAME, custom_objects={'sumSquaredError': sumSquaredError}) 
    model.summary()

    logging.basicConfig(filename="Results/" + MODEL_NAME.replace(".h5", ".csv"), filemode="w", format='', level=logging.INFO)

    score = model.evaluate(x=x_test, y=y_test)
    print("Evaluate score: ", score)
    logging.info("Evaluate score: %s" % (score))

    y_predict = model.predict(x_test)

    totally_correct = 0
    detection_correct = 0
    classification_correct = 0
    event_type_correct = 0
    EVENTS = ["ON", "OFF"]
    logging.info('y_sample, y_class, y_eventType, groundTruth_sample, groundTruth_class, groundTruth_eventType')
    for predict, groundTruth in zip(y_predict, y_test):
        print(int(predict[0] * SIGNAL_LENGTH), np.argmax(predict[3:]), int(groundTruth[0] * SIGNAL_LENGTH), np.argmax(groundTruth[3:]), K.eval(sumSquaredError(K.variable(groundTruth), K.variable(predict))))
        logging.info("%d, %d, %s, %d, %d, %s, %d" % (int(predict[0] * SIGNAL_LENGTH), np.argmax(predict[3:]), EVENTS[np.argmax(predict[1:3])], int(groundTruth[0] * SIGNAL_LENGTH), np.argmax(groundTruth[3:]), EVENTS[np.argmax(groundTruth[1:3])], 100 * K.eval(sumSquaredError(K.variable(groundTruth), K.variable(predict)))))
        if abs(int(predict[0] * SIGNAL_LENGTH) - int(groundTruth[0] * SIGNAL_LENGTH)) < (1 * 256):
            detection_correct += 1
            if np.argmax(predict[3:]) == np.argmax(groundTruth[3:]):
                totally_correct += 1
        if np.argmax(predict[3:]) == np.argmax(groundTruth[3:]):
            classification_correct += 1
        if np.argmax(predict[1:3]) == np.argmax(groundTruth[1:3]):
            event_type_correct += 1

    print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/y_test.shape[0], 100 * detection_correct/y_test.shape[0], 100 * classification_correct/y_test.shape[0], 100 * event_type_correct/y_test.shape[0]))
    logging.info("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/y_test.shape[0], 100 * detection_correct/y_test.shape[0], 100 * classification_correct/y_test.shape[0], 100 * event_type_correct/y_test.shape[0]))

if __name__ == '__main__':
    main()