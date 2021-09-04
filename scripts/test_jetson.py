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
    "FOLDER_PATH": "../TrainedWeights/Jetson/", 
    "FOLDER_DATA_PATH": "../TrainedWeights/Jetson/", 
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

general_qtd_test = np.load(folderDataPath + "general_test_qtd.npy")

bestModel = modelHandler.buildModel(old_sse=True)
bestModel.load_weights(folderPath + "best_model.h5")

print("Loaded Data")
print("Total test examples: {0}".format(x_test.shape[0]))

pcMetric, dMetric, classification_f1, fold_time = postProcessing.checkModelAll(bestModel, x_test, y_test, 
                                                                               general_qtd=general_qtd_test, 
                                                                               print_error=False)