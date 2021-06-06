from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
import pickle
from DataHandler import DataHandler
from ModelHandler import ModelHandler

TRAIN = 0 # Train using a ten folded k-fold
TRAIN_NO_KFOLD = 1

EXECUTION_STATE = TRAIN_NO_KFOLD

configs = {
    "N_GRIDS": 5, 
    "SIGNAL_BASE_LENGTH": 12800, 
    "N_CLASS": 26, 
    "USE_NO_LOAD": False, 
    "AUGMENTATION_RATIO": 5, 
    "MARGIN_RATIO": 0.15, 
    "DATASET_PATH": "Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.8,
    "FOLDER_PATH": "tmp/aug2/", 
    "FOLDER_DATA_PATH": "tmp/aug2/", 
    "N_EPOCHS_TRAINING": 250,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 250,
    "SNRdb": None # Noise level on db
}

def main():
    ngrids = configs["N_GRIDS"]
    signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
    trainSize = configs["TRAIN_SIZE"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    dataHandler = DataHandler(configs)

    # If it doesn't find the data on the specified format, it creates and stores new signal cuts and its labels
    if not os.path.isfile(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p"):
        print("Sorted data not found, creating new file...")
        x, ydet, yclass, ytype = dataHandler.loadData(SNR=configs["SNRdb"])
        print("Data loaded")
        x = np.reshape(x, (x.shape[0], x.shape[1],))
        print(x.shape)
        max_abs_scaler = MaxAbsScaler()
        x = max_abs_scaler.fit_transform(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        print(x.shape)
        print("Data scaled")
        x_train, x_test, y_train_detection, y_test_detection, y_train_type, y_test_type, y_train_classification, y_test_classification = train_test_split(x, ydet, ytype, yclass, train_size=trainSize, random_state = 42)
        y_train = {"detection": y_train_detection, "type": y_train_type, "classification": y_train_classification}
        y_test = {"detection": y_test_detection, "type": y_test_type, "classification": y_test_classification}
        print("Data sorted")

        dict_data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
        pickle.dump(dict_data, open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "wb"))
        print("Data stored")

    dict_data = pickle.load(open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "rb"))
    x_train = dict_data["x_train"]
    x_test = dict_data["x_test"]
    y_train = dict_data["y_train"]
    y_test = dict_data["y_test"]

    dataHandler.checkGridDistribution(y_train, y_test)
    print(x_train.shape, x_test.shape)

    # general_qtd_train, general_qtd_test = dataHandler.generateAcquisitionType(trainSize, augmentation=1)
    # dataHandler.checkAcquisitionType(y_train["classification"], load_type_train, general_qtd_train)
    # dataHandler.checkAcquisitionType(y_test["classification"], load_type_test, general_qtd_test)

    modelHandler = ModelHandler(configs)
    # postProcessing = PostProcessing(configs)

    if EXECUTION_STATE == TRAIN:
        X_all = np.vstack((x_train, x_test))
        ydet_all = np.vstack((y_train["detection"], y_test["detection"]))
        ytype_all = np.vstack((y_train["type"], y_test["type"]))
        yclass_all = np.vstack((y_train["classification"], y_test["classification"]))

        fold = 0
        kfold = KFold(n_splits=10)
        for train_index, test_index in kfold.split(X_all):
            fold += 1
            x_train = X_all[train_index]
            x_test = X_all[test_index]
            y_train["detection"] = ydet_all[train_index]
            y_test["detection"] = ydet_all[test_index]
            y_train["type"] = ytype_all[train_index]
            y_test["type"] = ytype_all[test_index]
            y_train["classification"] = yclass_all[train_index]
            y_test["classification"] = yclass_all[test_index]

            folderPath = configs["FOLDER_PATH"] + str(fold) + "/"

            np.save(folderPath + "train_index.npy", train_index)
            np.save(folderPath + "test_index.npy", test_index)

            tensorboard_callback = TensorBoard(log_dir='./' + folderDataPath + '/logs')
            model_checkpoint = ModelCheckpoint(filepath = folderPath + "best_model.h5", monitor = 'loss', mode='min', save_best_only=True)
            if configs["INITIAL_EPOCH"] > 0:
                model = ModelHandler.loadModel(folderPath + 'multiple_loads_multipleOutputs_12800_{0}.h5'.format(configs["INITIAL_EPOCH"]))
            else:
                model = modelHandler.buildModel()

            model.summary()

            fileEpoch = configs["INITIAL_EPOCH"]
            while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
                model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], \
                        epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[model_checkpoint, tensorboard_callback], batch_size=32)
                
                fileEpoch += configs["N_EPOCHS_TRAINING"]
                model.save(folderPath + 'multiple_loads_multipleOutputs_' + str(signalBaseLength) + "_" + str(fileEpoch) + '.h5')

                # postProcessing.checkModel(model, x_test, y_test)
                # bestModel = modelHandler.loadModel(folderPath + "best_model.h5")
                # postProcessing.checkModel(bestModel, x_test, y_test)
                # postProcessing.checkModel(bestModel, x_train, y_train)
                # MultiLabelMetrics.F1Macro(bestModel, x_test, y_test)
    
    elif EXECUTION_STATE == TRAIN_NO_KFOLD:
        weights = None
        folderPath = configs["FOLDER_PATH"]
        tensorboard_callback = TensorBoard(log_dir='./' + folderDataPath + '/logs')
        model_checkpoint = ModelCheckpoint(filepath = folderPath + "best_model.h5", monitor = 'loss', mode='min', save_best_only=True)
        if configs["INITIAL_EPOCH"] > 0:
            model = ModelHandler.loadModel(folderPath + 'multiple_loads_multipleOutputs_12800_{0}.h5'.format(configs["INITIAL_EPOCH"]), type_weights=weights)
        else:
            model = modelHandler.buildModel(type_weights=weights)

        model.summary()

        fileEpoch = configs["INITIAL_EPOCH"]
        while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
            model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], \
                    epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[model_checkpoint, tensorboard_callback], batch_size=32)
            
            fileEpoch += configs["N_EPOCHS_TRAINING"]
            model.save(folderPath + 'multiple_loads_multipleOutputs_' + str(signalBaseLength) + "_" + str(fileEpoch) + '.h5')

if __name__ == '__main__':
    main()