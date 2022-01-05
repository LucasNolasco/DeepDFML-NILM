from sklearn.preprocessing import MaxAbsScaler
from keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
 
configs = {
    "N_GRIDS": 5, 
    "SIGNAL_BASE_LENGTH": 12800, 
    "N_CLASS": 26, 
    "USE_NO_LOAD": False, 
    "AUGMENTATION_RATIO": 1, 
    "MARGIN_RATIO": 0.15, 
    "DATASET_PATH": "drive/MyDrive/YOLO_NILM/Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.9,
    "FOLDER_PATH": "drive/MyDrive/YOLO_NILM/final/001/", 
    "FOLDER_DATA_PATH": "drive/MyDrive/YOLO_NILM/final/001/", 
    "N_EPOCHS_TRAINING": 5000,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 5000,
    "SNRdb": None # Noise level on db
}

def freeze(model):
    for layer in model.layers:
        if 'classification' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    for layer in model.layers:
        print(layer.name, layer.trainable)

    return model

def calculating_class_weights(y_true):
    '''
        Source: https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
    '''
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

ngrids = configs["N_GRIDS"]
signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
trainSize = configs["TRAIN_SIZE"]
folderDataPath = configs["FOLDER_DATA_PATH"]
 
dataHandler = DataHandler(configs)

# Se não tiver os dados no formato necessário já organizados, faz a organização
if not os.path.isfile(folderDataPath + "data.p"):
    print("Sorted data not found, creating new file...")
    x, ydet, yclass, ytype, ygroup = dataHandler.loadData(SNR=configs["SNRdb"])
    print("Data loaded")

    data_mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    strat_classes = np.max(yclass, axis=1)

    train_index, test_index = next(data_mskf.split(x, strat_classes))

    y_train = {
        "detection": ydet[train_index], 
        "type": ytype[train_index], 
        "classification": yclass[train_index], 
        "group": ygroup[train_index]
    }
    
    y_test = {
        "detection": ydet[test_index], 
        "type": ytype[test_index], 
        "classification": yclass[test_index], 
        "group": ygroup[test_index]
    }
    
    dict_data = {
        "x_train": x[train_index], 
        "x_test": x[test_index], 
        "y_train": y_train, 
        "y_test": y_test
    }

    print("Data sorted")

    try:
        os.mkdir(folderDataPath)
    except:
        pass

    pickle.dump(dict_data, open(folderDataPath + "data.p", "wb"))
    print("Data stored")
else:
    dict_data = pickle.load(open(folderDataPath + "data.p", "rb"))

print(dict_data["x_train"].shape)
print(dict_data["x_test"].shape)

modelHandler = ModelHandler(configs)
postProcessing = PostProcessing(configs)
 
X_all = dict_data["x_train"]
ydet_all = dict_data["y_train"]["detection"]
ytype_all = dict_data["y_train"]["type"]
yclass_all = dict_data["y_train"]["classification"]
 
fold = 0
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
strat_classes = np.max(yclass_all, axis=1)
print(strat_classes.shape)

for train_index, validation_index in mskf.split(X_all, strat_classes):
    fold += 1

    print(f"---------- FOLD {fold} -------------")

    scaler = MaxAbsScaler()
    scaler.fit(np.squeeze(X_all[train_index], axis=2))
    x_train = np.expand_dims(scaler.transform(np.squeeze(X_all[train_index], axis=2)), axis=2)
    x_validation = np.expand_dims(scaler.transform(np.squeeze(X_all[validation_index], axis=2)), axis=2)
    
    y_train, y_validation = {}, {}
    y_train["detection"] = ydet_all[train_index]
    y_validation["detection"] = ydet_all[validation_index]
    y_train["type"] = ytype_all[train_index]
    y_validation["type"] = ytype_all[validation_index]
    y_train["classification"] = yclass_all[train_index]
    y_validation["classification"] = yclass_all[validation_index]

    yclass_weights = calculating_class_weights(np.max(y_train["classification"], axis=1))
    print(yclass_weights)
    
    folderPath = configs["FOLDER_PATH"] + str(fold) + "/"
    try:
        os.mkdir(folderPath)
    except:
        pass

    np.save(folderPath + "train_index.npy", train_index)
    np.save(folderPath + "validation_index.npy", validation_index)
    
    tensorboard_callback = TensorBoard(log_dir='./' + configs["FOLDER_PATH"] + '/logs')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50, verbose=True, restore_best_weights=True)
    classification_early_stopping_callback = EarlyStopping(monitor='val_classification_loss', patience=50, verbose=True, restore_best_weights=True)
    detection_early_stopping_callback = EarlyStopping(monitor='val_detection_loss', patience=50, verbose=True, restore_best_weights=True)
    type_early_stopping_callback = EarlyStopping(monitor='val_type_loss', patience=50, verbose=True, restore_best_weights=True)

    if configs["INITIAL_EPOCH"] > 0:
        model = ModelHandler.loadModel(folderPath + 'model_{0}.h5'.format(configs["INITIAL_EPOCH"]))
    else:
        model = modelHandler.buildModel()
 
    model.summary()
 
    fileEpoch = configs["INITIAL_EPOCH"]
    while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
        fileEpoch += configs["N_EPOCHS_TRAINING"]

        if not os.path.isfile(folderPath + 'model.h5'):
            model.compile(optimizer = Adam(), \
                          loss = [ModelHandler.sumSquaredError, "categorical_crossentropy", ModelHandler.get_bce_weighted_loss(yclass_weights)], \
                          metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])
            hist = model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], \
                             validation_data=(x_validation, [y_validation["detection"], y_validation["type"], y_validation["classification"]]), \
                             epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[early_stopping_callback, tensorboard_callback], batch_size=32)

            model.save(folderPath + 'model.h5')
        else: # If this model already exists, loads it
            model = ModelHandler.loadModel(folderPath + "model.h5")        

        if not os.path.isfile(folderPath + 'model_class_opt.h5'):
            print(f"FOLD {fold}: CLASSIFICATION FINE TUNNING PHASE")
            freeze(model)
            model.compile(optimizer = Adam(learning_rate=0.00001), \
                          loss = [ModelHandler.sumSquaredError, "categorical_crossentropy", ModelHandler.get_bce_weighted_loss(yclass_weights)], \
                          metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])

            hist_opt = model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], \
                                 validation_data=(x_validation, [y_validation["detection"], y_validation["type"], y_validation["classification"]]), \
                                 epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[classification_early_stopping_callback, tensorboard_callback], batch_size=32)
          
            model.save(folderPath + 'model_class_opt.h5')
  
    del model, y_validation, y_train, x_validation, x_train