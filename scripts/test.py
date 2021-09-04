# -*- coding: utf-8 -*-

import numpy as np
import pickle
import sys

sys.path.append("../src/")
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing
from DataHandler import DataHandler

configs = {
    "N_GRIDS": 5, 
    "SIGNAL_BASE_LENGTH": 12800, 
    "N_CLASS": 26, 
    "USE_NO_LOAD": False, 
    "AUGMENTATION_RATIO": 5, 
    "MARGIN_RATIO": 0.15, 
    "DATASET_PATH": "Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.8,
    "FOLDER_PATH": "../TrainedWeights/Final/", 
    "FOLDER_DATA_PATH": "../TrainedWeights/Final/", 
    "N_EPOCHS_TRAINING": 250,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 250,
    "SNRdb": None # Nivel de ruido em db
}

folderPath = configs["FOLDER_PATH"]
folderDataPath = configs["FOLDER_DATA_PATH"]
signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
trainSize = configs["TRAIN_SIZE"]
ngrids = configs["N_GRIDS"]

dict_data = pickle.load(open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "rb")) # Load data
x_train = dict_data["x_train"]
y_train = dict_data["y_train"]
x_test = dict_data["x_test"]
y_test = dict_data["y_test"]

modelHandler = ModelHandler(configs=configs)
postProcessing = PostProcessing(configs=configs)
dataHandler = DataHandler(configs=configs)

group_distribution = {
    "1": 4139,
    "2": 6916,
    "3": 7128,
    "8": 2629
}

general_qtd_train, general_qtd_test = dataHandler.generateAcquisitionType(trainSize, distribution=group_distribution)
X_all = np.vstack((x_train, x_test))
ydet_all = np.vstack((y_train["detection"], y_test["detection"]))
ytype_all = np.vstack((y_train["type"], y_test["type"]))
yclass_all = np.vstack((y_train["classification"], y_test["classification"]))

general_qtd = np.vstack((np.expand_dims(general_qtd_train, axis=1), np.expand_dims(general_qtd_test, axis=1)))

pcMetric, dMetric = [], []
f1 = []
total_time = 0
for fold in range(1, 11):
    foldFolderPath = folderPath + str(fold) + "/"
    
    train_index = np.load(foldFolderPath + "train_index.npy")
    test_index = np.load(foldFolderPath + "test_index.npy")

    bestModel = ModelHandler.loadModel(foldFolderPath + "best_model.h5", type_weights=None) # Load model

    x_train = X_all[train_index]
    x_test = X_all[test_index]
    y_train["detection"] = ydet_all[train_index]
    y_test["detection"] = ydet_all[test_index]
    y_train["type"] = ytype_all[train_index]
    y_test["type"] = ytype_all[test_index]
    y_train["classification"] = yclass_all[train_index]
    y_test["classification"] = yclass_all[test_index]

    general_qtd_test = general_qtd[test_index]

    print("-------------- FOLD %d ---------------" % (fold))
    pcMetric_fold, dMetric_fold, classification_f1, fold_time = postProcessing.checkModelAll(bestModel, x_test, y_test, 
                                                                                             general_qtd=general_qtd_test, 
                                                                                             print_error=False)
    pcMetric.append(pcMetric_fold)
    dMetric.append(dMetric_fold)
    f1.append(classification_f1)
    total_time += fold_time

print("------------ AVERAGE --------------")
print("++++++++++++++ DETECTION ++++++++++++++")
avgPCMetric = np.average(pcMetric, axis=0) * 100
avgDMetric = np.average(dMetric, axis=0)
for i, subset in enumerate(["1", "2", "3", "8", "All"]):
    print("Average, LIT-SYN-%s, PCMetric - On: %.1f, Off: %.1f, Total: %.1f" % (subset, avgPCMetric[i][0], avgPCMetric[i][1], avgPCMetric[i][2]))
    print("Average, LIT-SYN-%s, DMetric - On: %.1f, Off: %.1f, Total: %.1f" % (subset, avgDMetric[i][0], avgDMetric[i][1], avgDMetric[i][2]))

print("++++++++++++++ CLASSIFICATION ++++++++++++++")
print("F1 Score: %.2f\%" % (np.average(f1)))

print("++++++++++++++ TIME PERFORMANCE ++++++++++++++")
print("Total time: %g, Average Time: %g" % (total_time, total_time/X_all.shape[0]))