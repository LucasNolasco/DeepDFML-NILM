import numpy as np
import h5py
from keras import Sequential
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Conv1D, LeakyReLU, Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization, Input, Reshape, Softmax
from random import shuffle
from keras.callbacks import TensorBoard, ModelCheckpoint
import pickle
import os.path

DATASET_PATH = "Synthetic_1_iHall_vGrid.hdf5"
SIGNAL_LENGTH = 25600

N_GRIDS = 10
N_CLASS = 26

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
                out_detection = np.zeros((N_GRIDS, 1))
                out_classification = np.zeros((N_GRIDS, N_CLASS + 1))
                out_type = np.zeros((N_GRIDS, 3))

                gridLength = int(outputSignalLength / N_GRIDS)
                for grid in range(N_GRIDS):
                    gridEv = np.argwhere(event[initIndex + (grid * gridLength) : initIndex + (grid + 1) * gridLength] != 0)
                    if len(gridEv) > 0:
                        out_classification[grid][label[0]] = 1
                        out_detection[grid][0] = gridEv[0][0]/gridLength
                        if event[initIndex + grid * gridLength + gridEv[0][0]] == 1: # ON
                            out_type[grid][0] = 1
                        else: # OFF
                            out_type[grid][1] = 1
                    else:
                        out_classification[grid][N_CLASS] = 1
                        out_type[grid][2] = 1
                
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
    z = list(zip(x, y_detection, y_classification, y_type))
    shuffle(z)
    x, y_detection, y_classification, y_type = zip(*z)

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
    trainLength = round(ratio * 32) # TODO: modificar para algo automático
    for xi, ydet, yclass, ytype in zip(x, y_detection, y_classification, y_type):
        lab = str(np.argmax(yclass) % N_CLASS)
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
    '''

    y_train = {"detection" : y_detection_train, "classification" : y_classification_train, "type": y_type_train}
    y_test = {"detection" : y_detection_test, "classification" : y_classification_test, "type": y_type_test}

    return x_train, y_train, x_test, y_test

def sumSquaredError(y_true, y_pred):
    #weights = K.ones(29)
    #weights = weights[0].assign(5)
    #weights = weights[1].assign(0.5)
    #weights = weights[2].assign(0.5)
    return K.sum(K.square(y_true - y_pred), axis=-1)
    #return K.square(y_true[0] - y_pred[0]) * 5 + K.square(K.argmax(y_true[1:3]) - K.argmax(y_pred[1:3])) + K.square(K.argmax(y_true[3:]) - K.argmax(y_pred[3:]))

def buildModel():
    input = Input(shape=(SIGNAL_LENGTH, 1))
    x = Conv1D(filters=60, kernel_size=9, input_shape=(SIGNAL_LENGTH, 1))(input)
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
    classification_output = Dense(30)(classification_output)
    classification_output = LeakyReLU(alpha=0.1)(classification_output)
    classification_output = Dense((N_CLASS + 1) * N_GRIDS)(classification_output)
    classification_output = Reshape((N_GRIDS, (N_CLASS + 1)))(classification_output)
    classification_output = Softmax(axis=2, name="classification")(classification_output)

    type_output = Dense(10)(x)
    type_output = LeakyReLU(alpha = 0.1)(type_output)
    type_output = Dense(3 * N_GRIDS)(type_output)
    type_output = Reshape((N_GRIDS, 3))(type_output)
    type_output = Softmax(axis=2, name="type")(type_output)
        
    model = Model(inputs = input, outputs=[detection_output, type_output, classification_output])

    model.compile(optimizer='adam', loss = [sumSquaredError, "binary_crossentropy", "binary_crossentropy"], metrics=['accuracy'])

    return model

def processResult(y_detection, y_type, y_classification):
    detection = []
    event_type = []
    classification = []

    for grid in range(N_GRIDS):
        detection.append(int((grid + y_detection[grid][0]) * SIGNAL_LENGTH / N_GRIDS))
        event_type.append(np.argmax(y_type[grid]))
        classification.append(np.argmax(y_classification[grid]))
        
    return detection, event_type, classification

def checkModel(model, x_test, y_test):
    score = model.evaluate(x=x_test, y=[y_test["detection"], y_test["type"], y_test["classification"]])
    print("Evaluate score: ", score)

    totally_correct = 0
    detection_correct = 0
    classification_correct = 0
    on_off_correct = 0
    total_wrong = 0
    for xi, groundTruth_detection, groundTruth_type, groundTruth_classification in zip(x_test, y_test["detection"], y_test["type"], y_test["classification"]):
        result = model.predict(np.expand_dims(xi, axis = 0))
        predict_detection, predict_event_type, predict_classification = processResult(result[0][0], result[1][0], result[2][0])
        groundTruth_detection, groundTruth_type, groundTruth_classification = processResult(groundTruth_detection, groundTruth_type, groundTruth_classification)

        error = False
        for groundTruth_grid in range(N_GRIDS):
            detection = predict_detection[groundTruth_grid]
            event_type = predict_event_type[groundTruth_grid]
            classification = predict_classification[groundTruth_grid]

            truth_detection = groundTruth_detection[groundTruth_grid]
            truth_type = groundTruth_type[groundTruth_grid]
            truth_classification = groundTruth_classification[groundTruth_grid]
        
            #print(detection, event_type, classification, truth_detection, truth_type, truth_classification)

            if(truth_type != event_type) or (truth_type != 2 and truth_type == event_type and classification != truth_classification):
                error = True

            if abs(detection - truth_detection) < (1 * 256) or (event_type == truth_type and event_type == 2):
                detection_correct += 1
                if (classification == truth_classification or event_type == 2) and event_type == truth_type:
                    totally_correct += 1
            if classification == truth_classification or (event_type == truth_type and event_type == 2):
                classification_correct += 1
            if event_type == truth_type:
                on_off_correct += 1

        if error:
            total_wrong += 1
            for groundTruth_grid in range(N_GRIDS):
                detection = predict_detection[groundTruth_grid]
                event_type = predict_event_type[groundTruth_grid]
                classification = predict_classification[groundTruth_grid]

                truth_detection = groundTruth_detection[groundTruth_grid]
                truth_type = groundTruth_type[groundTruth_grid]
                truth_classification = groundTruth_classification[groundTruth_grid]
                
                print(detection, event_type, classification, truth_detection, truth_type, truth_classification)
                print("-----------------------------")

    print("Wrong: %d, Total: %d" % (total_wrong, x_test.shape[0]))
    print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/(x_test.shape[0] * N_GRIDS), 100 * detection_correct/(x_test.shape[0] * N_GRIDS), 100 * classification_correct/(x_test.shape[0] * N_GRIDS), 100 * on_off_correct/(x_test.shape[0] * N_GRIDS)))

def main():
    # Se não tiver os dados no formato necessário já organizados, faz a organização
    if not os.path.isfile("sorted_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p"):
        x, ydet, yclass, ytype = loadData(DATASET_PATH)
        x_train, y_train, x_test, y_test = splitData(x, ydet, yclass, ytype, 0.8)

        dict_data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
        pickle.dump(dict_data, open("sorted_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p", "wb"))

    dict_data = pickle.load(open("sorted_data_" + str(N_GRIDS) + "_" + str(SIGNAL_LENGTH) + ".p", "rb"))
    x_train = dict_data["x_train"]
    x_test = dict_data["x_test"]
    y_train = dict_data["y_train"]
    y_test = dict_data["y_test"]

    print(x_train.shape, x_test.shape)

    tensorboard_callback = TensorBoard(log_dir='./logs')
    model_checkpoint = ModelCheckpoint(filepath="best_model.h5", monitor = 'loss', mode='min', save_best_only=True)

    model = buildModel() #load_model('single_loads_multipleOutputs_12800_500.h5', custom_objects={'sumSquaredError': sumSquaredError}) 
    model.summary()

    fileEpoch = 0
    while fileEpoch < 500:
        model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], epochs=250, verbose=2, callbacks=[model_checkpoint, tensorboard_callback])
        fileEpoch += 250

        model.save('single_loads_multipleOutputs_' + str(SIGNAL_LENGTH) + "_" + str(fileEpoch) + '.h5')

        checkModel(model, x_test, y_test)
        checkModel(load_model("best_model.h5", custom_objects={'sumSquaredError': sumSquaredError}), x_test, y_test)



if __name__ == '__main__':
    main()