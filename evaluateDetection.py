import numpy as np
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing
from DataHandler import DataHandler
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
    "FOLDER_PATH": "tmp/aug2_newloss_kfold/", 
    "FOLDER_DATA_PATH": "tmp/aug2_newloss_kfold/", 
    "N_EPOCHS_TRAINING": 250,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 250,
    "SNRdb": None # Nível de ruído em db
}

folderPath = configs["FOLDER_PATH"]
folderDataPath = configs["FOLDER_DATA_PATH"]
signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
trainSize = configs["TRAIN_SIZE"]
ngrids = configs["N_GRIDS"]

dict_data = pickle.load(open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "rb")) # Load data
x_train = dict_data["x_train"]
x_test = dict_data["x_test"]
y_train = dict_data["y_train"]
y_test = dict_data["y_test"]

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
for fold in range(1, 11):
    print(f"----------- FOLD {fold} ------------")
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

    pcMetric_fold, dMetric_fold = postProcessing.checkModel(bestModel, x_test, y_test, general_qtd=general_qtd_test, print_error=False)
    pcMetric.append(pcMetric_fold)
    dMetric.append(dMetric_fold)

print(f"Average, PCMetric: {np.average(pcMetric, axis=0)}, dMetric: {np.average(dMetric, axis=0)}")