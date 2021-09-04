# -*- coding: utf-8 -*-

import pickle

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

print("Loading data")
dict_data = pickle.load(open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "rb")) # Load data

print("Storing using the new protocol (version 2)")
pickle.dump(dict_data, open("sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "wb"), protocol=2)