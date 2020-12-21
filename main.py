from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
import pickle
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing
from MultiLabelMetrics import MultiLabelMetrics
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

TRAIN = 0
TEST_ALL = 1
TEST_BEST_MODEL = 2
VISUALIZE_DATA = 4

EXECUTION_STATE = TEST_BEST_MODEL

configs = {
    "N_GRIDS": 5, 
    "SIGNAL_BASE_LENGTH": 12800, 
    "N_CLASS": 26, 
    "USE_NO_LOAD": False, 
    "AUGMENTATION_RATIO": 1, 
    "MARGIN_RATIO": 0.15, 
    "DATASET_PATH": "Synthetic_Full_iHall.hdf5",
    "TRAIN_SIZE": 0.8,
    "FOLDER_PATH": "Weights/MultiLabel/7/", 
    "FOLDER_DATA_PATH": "Weights/MultiLabel/2/", 
    "N_EPOCHS_TRAINING": 50,
    "INITIAL_EPOCH": 50,
    "TOTAL_MAX_EPOCHS": 250,
    "SNRdb": 10 # Nível de ruído em db
}

def main():
    ngrids = configs["N_GRIDS"]
    signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
    trainSize = configs["TRAIN_SIZE"]
    folderPath = configs["FOLDER_PATH"]
    folderDataPath = configs["FOLDER_DATA_PATH"]

    dataHandler = DataHandler(configs)

    # Se não tiver os dados no formato necessário já organizados, faz a organização
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

    # Combinações 1x2, 1x3, 1x8 e 3x8
    '''
    if not os.path.isfile(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p"):
        print("Sorted data not found, creating new file...")
        xa, yadet, yaclass, yatype = dataHandler.loadData(["1"], augmentation_ratio=1)
        xb, ybdet, ybclass, ybtype = dataHandler.loadData(["3"], augmentation_ratio=1)
        print(xa.shape, xb.shape)
        print("Data loaded")
        x = np.vstack((xa, xb))
        x = np.reshape(x, (x.shape[0], x.shape[1],))
        print(x.shape)
        max_abs_scaler = MaxAbsScaler()
        x = max_abs_scaler.fit_transform(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        print(x.shape)
        print("Data scaled")
        x_train = x[:yadet.shape[0]]
        x_test = x[yadet.shape[0]:]
        y_train = {"detection": yadet, "type": yatype, "classification": yaclass}
        y_test = {"detection": ybdet, "type": ybtype, "classification": ybclass}
        print("Data sorted")

        dict_data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}
        pickle.dump(dict_data, open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "wb"))
        print("Data stored")
    '''

    dict_data = pickle.load(open(folderDataPath + "sorted_aug_data_" + str(ngrids) + "_" + str(signalBaseLength) + ".p", "rb"))
    x_train = dict_data["x_train"]
    x_test = dict_data["x_test"]
    y_train = dict_data["y_train"]
    y_test = dict_data["y_test"]

    dataHandler.checkGridDistribution(y_train, y_test)
    print(x_train.shape, x_test.shape)

    '''
    from sklearn.utils.class_weight import compute_class_weight
    tipo = y_train["type"]
    tipo = np.reshape(tipo, (tipo.shape[0] * tipo.shape[1], tipo.shape[2]))
    tipo = np.argmax(tipo, axis=1)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(tipo), y=tipo)
    #weights = dict(enumerate(weights))
    print(weights)
    '''
    weights = None

    tensorboard_callback = TensorBoard(log_dir='./' + folderDataPath + '/logs')
    model_checkpoint = ModelCheckpoint(filepath = folderPath + "best_model.h5", monitor = 'loss', mode='min', save_best_only=True)

    modelHandler = ModelHandler(configs)
    postProcessing = PostProcessing(configs)

    if EXECUTION_STATE == TRAIN:
        if configs["INITIAL_EPOCH"] > 0:
            model = ModelHandler.loadModel(folderPath + 'multiple_loads_multipleOutputs_12800_{0}.h5'.format(configs["INITIAL_EPOCH"]), type_weights=weights)
        else:
            model = modelHandler.buildModel(type_weights=weights)

        model.summary()

        #modelHandler.plotModel(model, folderPath)

        fileEpoch = configs["INITIAL_EPOCH"]
        while fileEpoch < configs["TOTAL_MAX_EPOCHS"]:
            model.fit(x=x_train, y=[y_train["detection"], y_train["type"], y_train["classification"]], \
                      epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[model_checkpoint, tensorboard_callback], batch_size=32)
                      #epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[model_checkpoint], batch_size=32)
            
            fileEpoch += configs["N_EPOCHS_TRAINING"]
            model.save(folderPath + 'multiple_loads_multipleOutputs_' + str(signalBaseLength) + "_" + str(fileEpoch) + '.h5')

            postProcessing.checkModel(model, x_test, y_test)
            bestModel = modelHandler.loadModel(folderPath + "best_model.h5")
            postProcessing.checkModel(bestModel, x_test, y_test)
            MultiLabelMetrics.F1Macro(bestModel, x_test, y_test)
    
    elif EXECUTION_STATE == TEST_ALL:
        postProcessing.checkModel(ModelHandler.loadModel(folderPath + 'multiple_loads_multipleOutputs_12800_250.h5'), x_test, y_test)
        bestModel = modelHandler.loadModel(folderPath + "best_model.h5")
        postProcessing.checkModel(bestModel, x_test, y_test)
        MultiLabelMetrics.F1Macro(bestModel, x_train, y_train)
        MultiLabelMetrics.F1Macro(bestModel, x_test, y_test)
    
    elif EXECUTION_STATE == TEST_BEST_MODEL:
        bestModel = modelHandler.loadModel(folderPath + "best_model.h5", type_weights=weights)
        postProcessing.checkModel(bestModel, x_test, y_test, print_error=False)
        postProcessing.checkModel(bestModel, x_train, y_train, print_error=False)
        MultiLabelMetrics.F1Macro(bestModel, x_test, y_test)
        MultiLabelMetrics.F1Macro(bestModel, x_train, y_train)

    elif EXECUTION_STATE == VISUALIZE_DATA:
        from sklearn.manifold import TSNE
        from keras.models import Model
        import matplotlib.pyplot as plt
        model = modelHandler.loadModel(folderPath + "best_model.h5")
        model.summary()

        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        extracted_features = intermediate_layer_model.predict(x_train)

        tsne_features = TSNE(n_components=2).fit_transform(extracted_features)
        print(tsne_features.shape)

        plt.plot(tsne_features[:,0], tsne_features[:,1], '.')
        plt.show()


if __name__ == '__main__':
    main()