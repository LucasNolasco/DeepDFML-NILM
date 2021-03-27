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
MODEL_NAME = 'v3/Teste 4/single_loads_multipleOutputs_12800_750.h5'
SIGNAL_LENGTH = 12800

N_GRIDS = 1

# Carrega os dados do banco de dados e organiza, dividindo os dados do tamanho
def loadData(datasetPath):
    arq = h5py.File(datasetPath, "r")

    rawSamples = arq["i"]
    rawEvents = arq["events"]    
    rawLabels = arq["labels"]

    x, ydet, yclass, ytype = cutData(rawSamples, rawEvents, rawLabels, SIGNAL_LENGTH)

    arq.close()

    return x, ydet, yclass, ytype

# Pega os dados originais e faz recortes para diminuir o tamanho (Necessário para diminuir o tamanho do modelo)
def cutData(rawSamples, rawEvents, rawLabels, outputSignalLength):
    begin = True

    output_x = np.array([])
    output_detection = np.array([])
    output_classification = np.array([])
    output_type = np.array([])

    for sample, event, label in zip(rawSamples, rawEvents, rawLabels):
        initIndex = 0
        while (initIndex + outputSignalLength) < len(sample):
            ev = np.argwhere(event[initIndex : initIndex + outputSignalLength] != 0) # Verifica se há algum evento nesse recorte
            if len(ev) > 0:
                out_detection = np.zeros(N_GRIDS)
                out_classification = np.zeros(26 * N_GRIDS)
                out_type = np.zeros(2 * N_GRIDS)

                gridLength = int(outputSignalLength / N_GRIDS)
                for grid in range(N_GRIDS):
                    gridEv = np.argwhere(event[initIndex + (grid * gridLength) : initIndex + (grid + 1) * gridLength] != 0)
                    if len(gridEv) > 0:
                        out_classification[grid * 26 + label[0]] = 1
                        out_detection[grid] = gridEv[0][0]/gridLength
                        if event[initIndex + grid * gridLength + gridEv[0][0]] == 1: # ON
                            out_type[grid * 2] = 1
                        else: # OFF
                            out_type[grid * 2 + 1] = 1
                    else:
                        out_classification[grid * 27 + 26] = 1
                        #out_type[grid * 3 + 2] = 1
                
                if begin:
                    output_x = np.copy(np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0))
                    output_detection = np.copy(np.expand_dims(out_detection, axis=0))
                    output_classification = np.copy(np.expand_dims(out_classification, axis=0))
                    output_type = np.copy(np.expand_dims(out_type, axis=0))
                    begin = False
                else:
                    output_x = np.vstack((output_x, np.expand_dims(sample[initIndex : initIndex + outputSignalLength], axis = 0)))
                    output_detection = np.vstack((output_detection, np.expand_dims(out_detection, axis=0)))
                    output_classification = np.vstack((output_classification, np.expand_dims(out_classification, axis=0)))
                    output_type = np.vstack((output_type, np.expand_dims(out_type, axis=0)))

            initIndex += outputSignalLength
    
    return output_x, output_detection, output_classification, output_type

# divide os dados em partes de treino e teste
def splitData(x, y_detection, y_classification, y_type, ratio):
    '''
    z = zip(x, y)
    shuffle(z)
    x, y = zip(*z)
    '''

    x_train = np.array([])
    y_detection_train = np.array([])
    y_classification_train = np.array([])
    y_type_train = np.array([])
    x_test = np.array([])
    y_detection_test = np.array([])
    y_classification_test = np.array([])
    y_type_test = np.array([])

    beginTrain = False
    beginTest = False

    dict_labels = {}
    trainLength = round(ratio * 32) # TODO: modificar para algo automático
    for xi, ydet, yclass, ytype in zip(x, y_detection, y_classification, y_type):
        lab = str(np.argmax(yclass))
        if lab not in dict_labels:
            dict_labels[lab] = 0

        if dict_labels[lab] < trainLength:
            if not beginTrain:
                x_train = np.expand_dims(xi, axis=0)
                y_detection_train = np.expand_dims(ydet, axis=0)
                y_classification_train = np.expand_dims(yclass, axis=0)
                y_type_train = np.expand_dims(ytype, axis=0)
                beginTrain = True
            else:
                x_train = np.vstack((x_train, np.expand_dims(xi, axis=0)))
                y_detection_train = np.vstack((y_detection_train, np.expand_dims(ydet, axis=0)))
                y_classification_train = np.vstack((y_classification_train, np.expand_dims(yclass, axis=0)))
                y_type_train = np.vstack((y_type_train, np.expand_dims(ytype, axis=0)))
        else:
            if not beginTest:
                x_test = np.expand_dims(xi, axis=0)
                y_detection_test = np.expand_dims(ydet, axis=0)
                y_classification_test = np.expand_dims(yclass, axis=0)
                y_type_test = np.expand_dims(ytype, axis=0)
                beginTest = True
            else:
                x_test = np.vstack((x_test, np.expand_dims(xi, axis=0)))
                y_detection_test = np.vstack((y_detection_test, np.expand_dims(ydet, axis=0)))
                y_classification_test = np.vstack((y_classification_test, np.expand_dims(yclass, axis=0)))
                y_type_test = np.vstack((y_type_test, np.expand_dims(ytype, axis=0)))

        dict_labels[lab] += 1

    y_train = {"detection" : y_detection_train, "classification" : y_classification_train, "type": y_type_train}
    y_test = {"detection" : y_detection_test, "classification" : y_classification_test, "type": y_type_test}

    return x_train, y_train, x_test, y_test

def sumSquaredError(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=-1)

def processResult(y_predict):
    detection = []
    event_type = []
    classification = []

    for grid in range(N_GRIDS):
        if y_predict[0][0][grid] > 0.1:
            detection.append(int((grid + y_predict[0][0][grid]) * SIGNAL_LENGTH / N_GRIDS))
        else:
            detection.append(0)
        event_type.append(np.argmax(y_predict[1][0][grid * 3 : (grid + 1) * 3]))
        classification.append(np.argmax(y_predict[2][0][grid * 27 : (grid + 1) * 27]))

    return detection, event_type, classification

def main():
    x, ydet, yclass, ytype = loadData(DATASET_PATH)
    _, _, x_test, y_test = splitData(x, ydet, yclass, ytype, 0.75)

    model = load_model("Weights/" + MODEL_NAME, custom_objects={'sumSquaredError': sumSquaredError}) 
    model.summary()

    logging.basicConfig(filename="Results/" + MODEL_NAME.replace(".h5", ".csv"), filemode="w", format='', level=logging.INFO)

    score = model.evaluate(x=x_test, y=[y_test["detection"], y_test["type"], y_test["classification"]])
    print("Evaluate score: ", score)
    logging.info("Evaluate score: %s" % (score))

    totally_correct = 0
    detection_correct = 0
    classification_correct = 0
    on_off_correct = 0
    EVENTS = ["ON", "OFF"]
    logging.info('y_sample, y_class, y_eventType, groundTruth_sample, groundTruth_class, groundTruth_eventType')
    for xi, groundTruth_detection, groundTruth_type, groundTruth_classification in zip(x_test, y_test["detection"], y_test["type"], y_test["classification"]):
        predict = model.predict(np.expand_dims(xi, axis=0))
        predict_detection, predict_event_type, predict_classification = processResult(predict)

        for groundTruth_grid in range(N_GRIDS):
            truth_classification = np.argmax(groundTruth_classification[groundTruth_grid * 27 : (groundTruth_grid + 1) * 27])
            truth_type = np.argmax(groundTruth_type[groundTruth_grid * 3 : (groundTruth_grid + 1) * 3])
            if groundTruth_detection[groundTruth_grid] > 0.1:
                truth_detection = int(SIGNAL_LENGTH * (groundTruth_grid + groundTruth_detection[groundTruth_grid]) / N_GRIDS)
            else:
                truth_detection = 0

            detection = predict_detection[groundTruth_grid]
            event_type = predict_event_type[groundTruth_grid]
            classification = predict_classification[groundTruth_grid]
        
            print(detection, classification, EVENTS[event_type], truth_detection, truth_classification, EVENTS[truth_type])
            logging.info("%d, %d, %s, %d, %d, %s" % (detection, classification, EVENTS[event_type], truth_detection, truth_classification, EVENTS[truth_type]))
            if abs(detection - truth_detection) < (1 * 256):
                detection_correct += 1
                if classification == truth_classification and event_type == truth_type:
                    totally_correct += 1
            if classification == truth_classification:
                classification_correct += 1
            if event_type == truth_type:
                on_off_correct += 1

    print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/(x_test.shape[0] * N_GRIDS), 100 * detection_correct/(x_test.shape[0] * N_GRIDS), 100 * classification_correct/(x_test.shape[0] * N_GRIDS), 100 * on_off_correct/(x_test.shape[0] * N_GRIDS)))
    logging.info("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/(x_test.shape[0] * N_GRIDS), 100 * detection_correct/(x_test.shape[0] * N_GRIDS), 100 * classification_correct/(x_test.shape[0] * N_GRIDS), 100 * on_off_correct/(x_test.shape[0] * N_GRIDS)))

if __name__ == '__main__':
    main()