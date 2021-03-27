import numpy as np
import h5py
from keras import Sequential
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Conv1D, LeakyReLU, Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization, Input, Reshape, Softmax
from keras.activations import sigmoid
import tensorflow as tf
from random import shuffle, randrange, random
from keras.callbacks import TensorBoard, ModelCheckpoint
import pickle
import os.path
import matplotlib.pyplot as plt

DATASET_PATH = "Synthetic_Full_iHall.hdf5"
SIGNAL_LENGTH = 12800

N_GRIDS = 5
N_CLASS = 26
MARGIN = 0

AUGMENTATION_RATIO = 2
MARGIN_RATIO = 0.15

FOLDER_PATH = "Weights/Random Offset/12/a/"
FOLDER_DATA_PATH = "Weights/Random Offset/12/a/"

# --------------------------
#   MARGIN_RATIO ON SIGNAL
#    ___________________
#   |      Event        |
#   |___________________|
#       ^           ^
#       |           |
#   MARGIN_RATIO  (1 - MARGIN_RATIO)
# ---------------------------

def KerasFocalLoss(target, input):
    gamma = 2.
    input = tf.cast(input, tf.float32)
    
    max_val = K.clip(-1 * input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-1 * max_val) + K.exp(-1 * input - max_val))
    invprobs = tf.math.log_sigmoid(-1 * input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))

# Carrega os dados do banco de dados e organiza, dividindo os dados do tamanho
def loadData(datasetPath):
    arq = h5py.File(datasetPath, "r")

    rawSamples = arq["1"]["i"]
    rawEvents = arq["1"]["events"]    
    rawLabels = arq["1"]["labels"]

    x, ydet, yclass, ytype = cutData(rawSamples, rawEvents, rawLabels, SIGNAL_LENGTH)

    arq.close()

    return x, ydet, yclass, ytype

# Pega os dados originais e faz recortes para diminuir o tamanho (Necessário para diminuir o tamanho do modelo)
def cutData(rawSamples, rawEvents, rawLabels, outputSignalLength):
    begin = True
    gridLength = int(outputSignalLength / N_GRIDS)

    output_x = np.array([])
    output_detection = np.array([])
    output_classification = np.array([])
    output_type = np.array([])

    for sample, event, label in zip(rawSamples, rawEvents, rawLabels):
        initIndex = 0 #randrange(0, gridLength)
        while (initIndex + gridLength) < len(sample):
            ev = np.argwhere(event[initIndex : initIndex + gridLength] != 0) # Verifica se há algum evento nesse recorte
            
            #print(initIndex)
            if len(ev) > 0:
                for i in range(0, N_GRIDS):
                    for _ in range(AUGMENTATION_RATIO):
                        out_detection = np.zeros((N_GRIDS, 1))
                        out_classification = np.zeros((N_GRIDS, N_CLASS + 1))
                        out_type = np.zeros((N_GRIDS, 3))   

                        gridOccurrence = -1

                        for j in range(N_GRIDS):
                            out_classification[j][N_CLASS] = 1

                        randomizedInitCoordinates = initIndex + ev[0][0] - (N_GRIDS - 1 - i) * gridLength - randrange(0, gridLength) #randrange(int(gridLength * MARGIN_RATIO), int(gridLength * (1 - MARGIN_RATIO)))
                        for grid in range(N_GRIDS):
                            gridEv = np.argwhere(event[randomizedInitCoordinates + (grid * gridLength) : randomizedInitCoordinates + (grid + 1) * gridLength] != 0)
                            if len(gridEv) > 0:
                                gridOccurrence = grid
                                #out_classification[grid][label[0]] = 1
                                out_detection[grid][0] = gridEv[0][0]/gridLength
                                if event[randomizedInitCoordinates + grid * gridLength + gridEv[0][0]] == 1: # ON
                                    out_type[grid][0] = 1
                                    for j in range(grid, N_GRIDS):
                                        out_classification[j][label[0]] = 1
                                        out_classification[j][N_CLASS] = 0
                                else: # OFF
                                    out_type[grid][1] = 1
                                    for j in range(grid + 1):
                                        out_classification[j][label[0]] = 1
                                        out_classification[j][N_CLASS] = 0
                            else:
                                #out_classification[grid][N_CLASS] = 1
                                out_type[grid][2] = 1
                            
                        if gridOccurrence >= MARGIN and gridOccurrence < N_GRIDS - MARGIN:
                            if begin:
                                output_x = np.copy(np.expand_dims(sample[randomizedInitCoordinates - int(MARGIN_RATIO * outputSignalLength) : randomizedInitCoordinates + outputSignalLength + int(MARGIN_RATIO * outputSignalLength)], axis = 0))
                                output_detection = np.copy(np.expand_dims(out_detection, axis=0))
                                output_classification = np.copy(np.expand_dims(out_classification, axis=0))
                                output_type = np.copy(np.expand_dims(out_type, axis=0))
                                begin = False
                            else:
                                output_x = np.vstack((output_x, np.expand_dims(sample[randomizedInitCoordinates - int(MARGIN_RATIO * outputSignalLength) : randomizedInitCoordinates + outputSignalLength + int(MARGIN_RATIO * outputSignalLength)], axis = 0)))
                                output_detection = np.vstack((output_detection, np.expand_dims(out_detection, axis=0)))
                                output_classification = np.vstack((output_classification, np.expand_dims(out_classification, axis=0)))
                                output_type = np.vstack((output_type, np.expand_dims(out_type, axis=0)))

            initIndex += gridLength
    
    return output_x, output_detection, output_classification, output_type

# divide os dados em partes de treino e teste
def splitData(x, y_detection, y_classification, y_type, ratio):
    z = list(zip(x, y_detection, y_classification, y_type))
    shuffle(z)
    x, y_detection, y_classification, y_type = zip(*z)

    '''
    x = np.array(x)
    y_detection = np.array(y_detection)
    y_classification = np.array(y_classification)
    y_type = np.array(y_type)

    trainLength = round(ratio * x.shape[0])
    x_train = x[:trainLength]
    x_test = x[trainLength:]
    y_detection_train = y_detection[:trainLength]
    y_detection_test = y_detection[trainLength:]
    y_classification_train = y_classification[:trainLength]
    y_classification_test = y_classification[trainLength:]
    y_type_train = y_type[:trainLength]
    y_type_test = y_type[trainLength:]
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
    trainLength = round(ratio * 32 * (N_GRIDS - 2 * MARGIN) * AUGMENTATION_RATIO) # TODO: modificar para algo automático
    for xi, ydet, yclass, ytype in zip(x, y_detection, y_classification, y_type):
        for i in range(N_GRIDS):
            lab = np.argmax(yclass[i]) % N_CLASS
            if yclass[i][lab] == 1:
                lab = str(lab)
                if lab not in dict_labels:
                    dict_labels[lab] = 0
                
                break

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
    #weights = K.ones(29)
    #weights = weights[0].assign(5)
    #weights = weights[1].assign(0.5)
    #weights = weights[2].assign(0.5)
    return K.sum(K.square(y_true - y_pred), axis=-1)

def buildModel():
    input = Input(shape=(SIGNAL_LENGTH + 2 * int(SIGNAL_LENGTH * MARGIN_RATIO), 1))
    x = Conv1D(filters=60, kernel_size=9)(input)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.25)(x)
    x = Conv1D(filters=40, kernel_size=9)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.25)(x)
    x = Conv1D(filters=40, kernel_size=9)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.25)(x)
    x = Conv1D(filters=40, kernel_size=9)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.25)(x)
    x = Conv1D(filters=40, kernel_size=9)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling1D(pool_size=4)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.25)(x)
    x = Flatten()(x)

    detection_output = Dense(200)(x)
    detection_output = LeakyReLU(alpha = 0.1)(detection_output)
    detection_output = Dropout(0.25)(detection_output)
    detection_output = Dense(20)(detection_output)
    detection_output = LeakyReLU(alpha = 0.1)(detection_output)
    detection_output = Dense(1 * N_GRIDS, activation='sigmoid')(detection_output)
    detection_output = Reshape((N_GRIDS, 1), name="detection")(detection_output)

    classification_output = Dense(300)(x)
    classification_output = LeakyReLU(alpha = 0.1)(classification_output)
    classification_output = Dropout(0.25)(classification_output)
    classification_output = Dense(300)(classification_output)
    classification_output = LeakyReLU(alpha=0.1)(classification_output)
    classification_output = Dense((N_CLASS + 1) * N_GRIDS, activation = 'sigmoid')(classification_output)
    classification_output = Reshape((N_GRIDS, (N_CLASS + 1)), name = "classification")(classification_output)
    #classification_output = Softmax(axis=2, name="classification")(classification_output)

    type_output = Dense(10)(x)
    type_output = LeakyReLU(alpha = 0.1)(type_output)
    type_output = Dense(3 * N_GRIDS)(type_output)
    type_output = Reshape((N_GRIDS, 3))(type_output)
    type_output = Softmax(axis=2, name="type")(type_output)
    
    model = Model(inputs = input, outputs=[detection_output, type_output, classification_output])

    model.compile(optimizer='adam', loss = [sumSquaredError, "categorical_crossentropy", "binary_crossentropy"], metrics=['accuracy'])

    return model

def processResult(y_detection, y_type, y_classification):
    detection = []
    event_type = []
    classification = []

    for grid in range(N_GRIDS):
        detection.append(int((grid + y_detection[grid][0]) * SIGNAL_LENGTH / N_GRIDS) + int(MARGIN_RATIO * SIGNAL_LENGTH))
        event_type.append([np.argmax(y_type[grid]), np.max(y_type[grid])])
        classification.append([np.argmax(y_classification[grid]), np.max(y_classification[grid])])

    '''
    for grid in range(N_GRIDS):
        if (event_type[grid][0] == 2 and classification[grid][0] != N_CLASS):
            if event_type[grid][1] > classification[grid][1]:
                classification[grid][0] = N_CLASS
            else:
                event_type[grid][0] = np.argmax(y_type[grid][:2])
        if (event_type[grid][0] != 2 and classification[grid][0] == N_CLASS):
            if event_type[grid][1] > classification[grid][1]:
                classification[grid][0] = np.argmax(y_classification[grid][:N_CLASS])
                classification[grid][1] = np.max(y_classification[grid][:N_CLASS])
            else:
                event_type[grid][0] = 2
    '''

    return detection, event_type, classification

def suppression(detection, event_type, classification):
    events = []

    for grid in range(N_GRIDS):
        if event_type[grid][0] != 2 and classification[grid][0] != N_CLASS and event_type[grid][1] > 0.98:
            not_registered = True
            for i in range(len(events)):
                if events[i][2][0] == classification[grid][0]:
                    not_registered = False
                    if events[i][2][1] < classification[grid][1]:
                        events[i][0][0] = detection[grid]
                        events[i][1] = event_type[grid]
                        events[i][2] = classification[grid]
            
            if not_registered:
                events.append([[detection[grid], 1], event_type[grid], classification[grid]])

    return np.array(events)

def checkGridDistribution(y_train, y_test):
    dict_train = {}
    dict_test = {}
    for i in range(0, N_GRIDS):
        dict_train[str(i)] = 0
        dict_test[str(i)] = 0

    for y in y_train["type"]:
        for i in range(N_GRIDS):
            if np.argmax(y[i]) != 2:
                dict_train[str(i)] += 1

    for y in y_test["type"]:
        for i in range(N_GRIDS):
            if np.argmax(y[i]) != 2:
                dict_test[str(i)] += 1

    print(dict_train)
    print(dict_test)

def checkModel(model, x_test, y_test):
    total_events = 0
    detection_correct = 0
    classification_correct = 0
    type_correct = 0
    totally_correct = 0
    total_wrong = 0
    detection_error = []

    for xi, groundTruth_detection, groundTruth_type, groundTruth_classification in zip(x_test, y_test["detection"], y_test["type"], y_test["classification"]):
        error = False
        
        result = model.predict(np.expand_dims(xi, axis = 0))

        raw_detection, raw_type, raw_classification = processResult(result[0][0], result[1][0], result[2][0])
        raw_gt_detection, raw_gt_type, raw_gt_classification = processResult(groundTruth_detection, groundTruth_type, groundTruth_classification)

        predict_events = suppression(raw_detection, raw_type, raw_classification)
        groundTruth_events = suppression(raw_gt_detection, raw_gt_type, raw_gt_classification)

        event_outside_margin = False

        if not event_outside_margin:
            if len(predict_events) == len(groundTruth_events):
                for prediction, groundTruth in zip(predict_events, groundTruth_events):
                    total_events += 1
                    if abs(prediction[0][0] - groundTruth[0][0]) < 256:
                        detection_correct += 1
                        if prediction[2][0] == groundTruth[2][0] and prediction[1][0] == groundTruth[1][0]:
                            totally_correct += 1
                    if prediction[2][0] == groundTruth[2][0]:
                        classification_correct += 1
                    if prediction[1][0] == groundTruth[1][0]:
                        type_correct += 1

                    if(groundTruth[1][0] != prediction[1][0]) or groundTruth[2][0] != prediction[2][0]:
                        error = True
                        print(prediction, groundTruth)

                    detection_error.append(abs(prediction[0][0] - groundTruth[0][0]))
            else:
                total_events += len(groundTruth_events)
                print(predict_events, groundTruth_events)
                error = True

        if error:
            total_wrong += 1
            for groundTruth_grid in range(N_GRIDS):
                detection = raw_detection[groundTruth_grid]
                event_type = raw_type[groundTruth_grid][0]
                classification = raw_classification[groundTruth_grid][0]

                truth_detection = raw_gt_detection[groundTruth_grid]
                truth_type = raw_gt_type[groundTruth_grid][0]
                truth_classification = raw_gt_classification[groundTruth_grid][0]
                
                print(detection, event_type, classification, truth_detection, truth_type, truth_classification)
                #print(raw_type[groundTruth_grid][1], raw_classification[groundTruth_grid][1])
            
            print("----------------------")

    print("Wrong: %d, Total: %d" % (total_wrong, total_events))
    print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/total_events, 100 * detection_correct/total_events, 100 * classification_correct/total_events, 100 * type_correct/total_events))
    print("Average Detection Error: %.2f, Std Deviation: %.2f" % (np.average(detection_error), np.std(detection_error, ddof=1)))

def plotSignal(x, y_detection, y_type, y_classification):
    eventVec = np.zeros((x.shape[0], 1))
    loadClass = -1
    for j in range(0, len(y_detection)):
        if y_type[j] == 0:
            eventVec[y_detection[j]] = 1
            loadClass = y_classification[j]
        elif y_type[j] == 1:
            eventVec[y_detection[j]] = -1
            loadClass = y_classification[j]

    _, ax = plt.subplots()
    ax.set_title(str(loadClass))
    ax.plot(np.arange(0, x.shape[0]), x)
    ax.plot(np.arange(0, x.shape[0]), eventVec)
    plt.show()

def main():
    # Se não tiver os dados no formato necessário já organizados, faz a organização
    if not os.path.isfile(FOLDER_DATA_PATH + "sorted_aug_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p"):
        print("Sorted data not found, creating new file...")
        x, ydet, yclass, ytype = loadData(DATASET_PATH)
        print("Data loaded")
        x_train, y_train, x_test, y_test = splitData(x, ydet, yclass, ytype, 0.75)
        print("Data sorted")

        dict_data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
        pickle.dump(dict_data, open(FOLDER_DATA_PATH + "sorted_aug_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p", "wb"))
        print("Data stored")

    dict_data = pickle.load(open(FOLDER_DATA_PATH + "sorted_aug_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p", "rb"))
    x_train = dict_data["x_train"]
    x_test = dict_data["x_test"]
    y_train = dict_data["y_train"]
    y_test = dict_data["y_test"]

    checkGridDistribution(y_train, y_test)

    print(x_train.shape, x_test.shape)

    tensorboard_callback = TensorBoard(log_dir='./logs')
    model_checkpoint = ModelCheckpoint(filepath = FOLDER_PATH + "best_model.h5", monitor = 'loss', mode='min', save_best_only=True)

    #model = buildModel() #load_model(FOLDER_PATH + 'single_loads_aug_multipleOutputs_12800_250.h5', custom_objects={'sumSquaredError': sumSquaredError, 'KerasFocalLoss': KerasFocalLoss}) 
    #model.summary()

    fileEpoch = 0
    while fileEpoch < 250:
        #model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], epochs=250, verbose=2, callbacks=[model_checkpoint, tensorboard_callback])
        fileEpoch += 250

        #model.save(FOLDER_PATH + 'single_loads_aug_multipleOutputs_' + str(SIGNAL_LENGTH) + "_" + str(fileEpoch) + '.h5')

        #checkModel(model, x_test, y_test)
        checkModel(load_model(FOLDER_PATH + "best_model.h5", custom_objects={'sumSquaredError': sumSquaredError, 'KerasFocalLoss': KerasFocalLoss}), x_test, y_test)

class AnalysisWindow():
    def __init__(self, initSample):
        self.totalAnalysis = 0
        self.initSample = initSample
        self.events = []

    def setTotalAnalysis(self, totalAnalysis):
        self.totalAnalysis = totalAnalysis

    def add(self, event):
        if self.totalAnalysis < N_GRIDS:
            self.events.append(event)

    def move(self):
        self.totalAnalysis += 1

    def isFinished(self):
        if self.totalAnalysis >= N_GRIDS:
            return True
        else:
            return False

    def compileResults(self):
        finalEvent = [[0, 0], [0, 0], [0, 0]]
        evType = np.zeros((3, 1))
        evClass = np.zeros((N_CLASS + 1, 1))

        if self.isFinished():
            if len(self.events) >= N_GRIDS:
                for ev in self.events:
                    finalEvent[0][0] += ev[0][0] / len(self.events)
                    finalEvent[0][1] += ev[0][1] / len(self.events)
                    finalEvent[1][1] += ev[1][1] / len(self.events)
                    finalEvent[2][1] += ev[2][1] / len(self.events)
                    
                    evType[int(ev[1][0])] += 1
                    evClass[int(ev[2][0])] += 1

                finalEvent[1][0] = np.argmax(evType)
                finalEvent[2][0] = np.argmax(evClass)
            else:
                finalEvent = [[0, 1], [2, 1], [26, 1]]

        return finalEvent

if __name__ == '__main__':
    main()

    '''
    arq = h5py.File("Synthetic_2_iHall.hdf5", "r")

    x = np.array(arq["i"])
    ydet = np.array(arq["events"])
    yclass = np.array(arq["labels"])

    arq.close()

    model = load_model(FOLDER_PATH + 'best_model.h5', custom_objects={'sumSquaredError': sumSquaredError, 'KerasFocalLoss': KerasFocalLoss}) 
    
    for k in range(0, x.shape[0]):
        # 5: [15, 30, 47, 95, 127, 155, 161, 167, 170, 173, 192, 287, 317, 384]
        # 4: [15, 30, 47, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 95, 151, 155, 161, 162, 165, 166, 167, 168, 170, 171, 172, 173, 174, 192, 337, 339, 341, 342]
        
        title = ""
        for lab in yclass[k]:
            if title == "":
                title = title + str(lab)
            else:
                title = title + ", " + str(lab)
        
        title = "[" + title + "]"

        windows = []
        for i in range(N_GRIDS):
            windows.append(AnalysisWindow(i * round(SIGNAL_LENGTH / N_GRIDS) + int(MARGIN_RATIO * SIGNAL_LENGTH)))
            windows[i].setTotalAnalysis(N_GRIDS - 1 + i)

        y_res = np.zeros((x.shape[1], 1))
        for i in range(0, x.shape[1] - SIGNAL_LENGTH - 2 * int(MARGIN_RATIO * SIGNAL_LENGTH), round(SIGNAL_LENGTH / N_GRIDS)):
            result = model.predict(np.expand_dims(x[k][i:i + SIGNAL_LENGTH + 2 * int(MARGIN_RATIO * SIGNAL_LENGTH)], axis = 0))
            raw_detection, raw_event_type, raw_classification = processResult(result[0][0], result[1][0], result[2][0])
            events = suppression(raw_detection, raw_event_type, raw_classification)
            for ev in events:
                ev[0][0] += i
                #print(int(ev[0][0]), ev[0][1], int(ev[1][0]), ev[1][1], int(ev[2][0]), ev[2][1])
                for window in windows:
                    #print(ev[0][0], window.initSample, window.initSample + round(SIGNAL_LENGTH / N_GRIDS))
                    if ev[0][0] >= window.initSample and ev[0][0] < window.initSample + round(SIGNAL_LENGTH / N_GRIDS):
                        window.add(ev)

            for window in windows:
                window.move()

            if windows[0].isFinished():
                ev = windows[0].compileResults()
                if ev[1][0] == 0:
                    y_res[int(ev[0][0])] = 1
                elif ev[1][0] == 1:
                    y_res[int(ev[0][0])] = -1

                if ev[1][0] != 2:
                    title = title + ", " + str(int(ev[2][0]))
                
                windows.append(AnalysisWindow(windows[N_GRIDS - 1].initSample + round(SIGNAL_LENGTH / N_GRIDS)))
                del windows[0]
        
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.plot(np.arange(0, x.shape[1]), x[k])
        ax.plot(np.arange(0, x.shape[1]), ydet[k])
        ax.plot(np.arange(0, x.shape[1]), y_res)
        fig.savefig("Imagens/%d.png" % (k))
        #plt.show()
        plt.close(fig)

        print("%d/%d" % (k + 1, x.shape[0]))
        print("------------")
    '''