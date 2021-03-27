from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import os
import pickle
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from PostProcessing import PostProcessing
from MultiLabelMetrics import MultiLabelMetrics

TRAIN = 0
TEST_ALL = 1
TEST_BEST_MODEL = 2
VISUALIZE_DATA = 4
TRAIN_NO_KFOLD = 5

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
    "FOLDER_PATH": "scattering/", 
    "FOLDER_DATA_PATH": "scattering/", 
    "N_EPOCHS_TRAINING": 500,
    "INITIAL_EPOCH": 0,
    "TOTAL_MAX_EPOCHS": 500,
    "SNRdb": None # Nível de ruído em db
}

def createLogger(dir_path, save_format = 'txt'):
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	handler = logging.FileHandler(dir_path + 'log.' + save_format)
	handler.setLevel(logging.INFO)

	logger.addHandler(handler)

	return logger

def main():
    ngrids = configs["N_GRIDS"]
    signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
    trainSize = configs["TRAIN_SIZE"]
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

    load_type_train, load_type_test, general_qtd_train, general_qtd_test = dataHandler.generateAcquisitionType(trainSize, augmentation=1)
    dataHandler.checkAcquisitionType(y_train["classification"], load_type_train, general_qtd_train)
    dataHandler.checkAcquisitionType(y_test["classification"], load_type_test, general_qtd_test)

    modelHandler = ModelHandler(configs)
    postProcessing = PostProcessing(configs)

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
                    epochs=configs["N_EPOCHS_TRAINING"], verbose=2, callbacks=[tensorboard_callback], batch_size=32)
            
            fileEpoch += configs["N_EPOCHS_TRAINING"]
            model.save(folderPath + 'multiple_loads_multipleOutputs_' + str(signalBaseLength) + "_" + str(fileEpoch) + '.h5')
    
    elif EXECUTION_STATE == TEST_ALL:
        postProcessing.checkModel(ModelHandler.loadModel(folderPath + 'multiple_loads_multipleOutputs_12800_250.h5'), x_test, y_test)
        bestModel = modelHandler.loadModel(folderPath + "best_model.h5")
        postProcessing.checkModel(bestModel, x_test, y_test)
        MultiLabelMetrics.F1Macro(bestModel, x_train, y_train)
        MultiLabelMetrics.F1Macro(bestModel, x_test, y_test)
    
    elif EXECUTION_STATE == TEST_BEST_MODEL:
        pcMetric, dMetric = [], []
        # f1_macro, f1_micro = [], []
        # f1_macro_with_detection, f1_micro_with_detection = [], []

        # acc_on, acc_off, acc_total = [], [], []
        # multiclass_f1_macro, multiclass_f1_micro = [], []

        for fold in range(1, 2):
            fold = "/"
            folderPath = configs["FOLDER_PATH"] + str(fold) + "/"

            # train_index = np.load(folderPath + "train_index.npy")
            # test_index = np.load(folderPath + "test_index.npy")

            # x_train = X_all[train_index]
            # x_test = X_all[test_index]
            # y_train["detection"] = ydet_all[train_index]
            # y_test["detection"] = ydet_all[test_index]
            # y_train["type"] = ytype_all[train_index]
            # y_test["type"] = ytype_all[test_index]
            # y_train["classification"] = yclass_all[train_index]
            # y_test["classification"] = yclass_all[test_index]

            print("-------- FOLD #{0} --------".format(fold))

            bestModel = modelHandler.loadModel(folderPath + "best_model.h5", type_weights=weights)
            print("-------- TEST ---------")
            pcMetric_fold, dMetric_fold = postProcessing.checkModel(bestModel, x_test, y_test, load_type_test, general_qtd_test, print_error=False)
            # pcMetric_fold, dMetric_fold = postProcessing.checkModel(bestModel, x_train, y_train, load_type_train, general_qtd_train, print_error=False)
            pcMetric.append(pcMetric_fold)
            dMetric.append(dMetric_fold)

            # MultiClass Metrics
            '''
            fold_acc_on, fold_acc_off, fold_acc_total, fold_f1_macro, fold_f1_micro = postProcessing.checkMultiClassAccuracy(bestModel, x_test, y_test)

            acc_on.append(fold_acc_on)
            acc_off.append(fold_acc_off)
            acc_total.append(fold_acc_total)
            multiclass_f1_macro.append(fold_f1_macro)
            multiclass_f1_micro.append(fold_f1_micro)

            print("Acc on with detection: {0}, without detection: {1}".format(fold_acc_on[0]*100, fold_acc_on[1]*100))
            print("Acc off with detection: {0}, without detection: {1}".format(fold_acc_off[0]*100, fold_acc_off[1]*100))
            print("Acc total with detection: {0}, without detection: {1}".format(fold_acc_total[0]*100, fold_acc_total[1]*100))
            print("F1 macro with detection: {0}, without detection: {1}".format(fold_f1_macro[0]*100, fold_f1_macro[1]*100))
            print("F1 micro with detection: {0}, without detection: {1}".format(fold_f1_micro[0]*100, fold_f1_micro[1]*100))
            '''

            '''
            print("-------- TRAIN ---------")
            postProcessing.checkModel(bestModel, x_train, y_train, load_type_train, general_qtd_train, print_error=False)
            
            print("-------- ALL ---------")
            all_x = np.vstack((x_train, x_test))
            all_y = {}
            all_y["classification"] = np.vstack((y_train["classification"], y_test["classification"]))
            all_y["type"] = np.vstack((y_train["type"], y_test["type"]))
            all_y["detection"] = np.vstack((y_train["detection"], y_test["detection"]))
            all_load_type = np.vstack((np.expand_dims(load_type_train, axis=1), np.expand_dims(load_type_test, axis=1)))
            all_general_qtd = np.vstack((np.expand_dims(general_qtd_train, axis=1), np.expand_dims(general_qtd_test, axis=1)))

            postProcessing.checkModel(bestModel, all_x, all_y, all_load_type, all_general_qtd, print_error=False)
            '''

            # F1-MultiLabel TRAIN
            # from sklearn.metrics import f1_score, precision_score, recall_score     
            # for threshold in [0.5]:
            #     final_prediction = []
            #     final_prediction_with_detection = []
            #     final_groundTruth = []
            #     for xi, yclass, ytype in zip(x_test, y_test["classification"], y_test["type"]):
            #         pred = bestModel.predict(np.expand_dims(xi, axis=0))
            #         prediction = np.max(pred[2][0],axis=0) > threshold
            #         groundTruth = np.max(yclass,axis=0) > threshold

            #         det = np.array([np.argmax(i) for i in ytype])
            #         prediction_with_detection = np.max(pred[2][0],axis=0) > 2 # Gambiarra
            #         if (det != 2).any():
            #             prediction_with_detection = np.max(pred[2][0],axis=0) > threshold

            #         final_prediction.append(prediction)
            #         final_groundTruth.append(groundTruth)
            #         final_prediction_with_detection.append(prediction_with_detection)
                
            #     f1_macro.append(f1_score(final_groundTruth, final_prediction, average='macro'))
            #     f1_micro.append(f1_score(final_groundTruth, final_prediction, average='micro'))

            #     fold_f1_macro_with_detection, fold_f1_micro_with_detection = postProcessing.f1_with_detection(bestModel, 
            #                                                                                                     x_test, 
            #                                                                                                     y_test, 
            #                                                                                                     #general_acquisition_type=load_type_test, 
            #                                                                                                     #target=2, 
            #                                                                                                     print_error=False)

            #     f1_macro_with_detection.append(fold_f1_macro_with_detection)
            #     f1_micro_with_detection.append(fold_f1_micro_with_detection)

            #     print("Threshold: {0}, F1 Macro: {1}, F1 Micro: {2}, Precision Macro: {3}, Precision Micro: {4}, Recall Macro: {5}, Recal Micro: {6}".format(\
            #         threshold, \
            #         f1_macro[-1], \
            #         f1_micro[-1], \
            #         precision_score(final_groundTruth, final_prediction, average="macro"), \
            #         precision_score(final_groundTruth, final_prediction, average="micro"), \
            #         recall_score(final_groundTruth, final_prediction, average="macro"), \
            #         recall_score(final_groundTruth, final_prediction, average="micro")))

            #     print("[WITH DETECTION] Threshold: {0}, F1 Macro: {1}, F1 Micro: {2}, Precision Macro: {3}, Precision Micro: {4}, Recall Macro: {5}, Recal Micro: {6}".format(\
            #         threshold, \
            #         f1_macro_with_detection[-1], \
            #         f1_micro_with_detection[-1], \
            #         precision_score(final_groundTruth, final_prediction_with_detection, average="macro"), \
            #         precision_score(final_groundTruth, final_prediction_with_detection, average="micro"), \
            #         recall_score(final_groundTruth, final_prediction_with_detection, average="macro"), \
            #         recall_score(final_groundTruth, final_prediction_with_detection, average="micro")))

            '''
                final_prediction = []
                final_groundTruth = []
                for xi, yclass in zip(x_train, y_train["classification"]):
                    pred = bestModel.predict(np.expand_dims(xi, axis=0))
                    prediction = np.max(pred[2][0],axis=0) > threshold
                    groundTruth = np.max(yclass,axis=0) > threshold

                    final_prediction.append(prediction)
                    final_groundTruth.append(groundTruth)

                print("Threshold: {0}, F1 Macro: {1}, F1 Micro: {2}, Precision Macro: {3}, Precision Micro: {4}, Recall Macro: {5}, Recal Micro: {6}".format(\
                    threshold, \
                    f1_score(final_groundTruth, final_prediction, average='macro'), \
                    f1_score(final_groundTruth, final_prediction, average='micro'), \
                    precision_score(final_groundTruth, final_prediction, average="macro"), \
                    precision_score(final_groundTruth, final_prediction, average="micro"), \
                    recall_score(final_groundTruth, final_prediction, average="macro"), \
                    recall_score(final_groundTruth, final_prediction, average="micro")))
            '''

            del x_train
            del x_test
            del bestModel
        
        averagePCMetric = np.average(pcMetric, axis=0)
        averageDMetric = np.average(dMetric, axis=0)

        print("-------- AVERAGE --------")

        print("LIT-SYN-1 PCmetric: {0}".format(averagePCMetric[0]))
        print("LIT-SYN-1 Dmetric: {0}".format(averageDMetric[0]))

        print("LIT-SYN-2 PCmetric: {0}".format(averagePCMetric[1]))
        print("LIT-SYN-2 Dmetric: {0}".format(averageDMetric[1]))

        print("LIT-SYN-3 PCmetric: {0}".format(averagePCMetric[2]))
        print("LIT-SYN-3 Dmetric: {0}".format(averageDMetric[2]))

        print("LIT-SYN-8 PCmetric: {0}".format(averagePCMetric[3]))
        print("LIT-SYN-8 Dmetric: {0}".format(averageDMetric[3]))

        print("LIT-SYN-All PCmetric: {0}".format(averagePCMetric[4]))
        print("LIT-SYN-All Dmetric: {0}".format(averageDMetric[4]))

        # print("Average F1 Macro: {0}".format(np.average(f1_macro, axis=0)))
        # print("Average F1 Micro: {0}".format(np.average(f1_micro, axis=0)))

        # print("Average F1 Macro with detection: {0}".format(np.average(f1_macro_with_detection, axis=0)))
        # print("Average F1 Micro with detection: {0}".format(np.average(f1_micro_with_detection, axis=0)))

        # average_acc_on = np.average(acc_on, axis=0)
        # average_acc_off = np.average(acc_off, axis=0)
        # average_acc_total = np.average(acc_total, axis=0)
        # average_multiclass_f1_macro = np.average(multiclass_f1_macro, axis=0)
        # average_multiclass_f1_micro = np.average(multiclass_f1_micro, axis=0)

        # print("Average acc on with detection: {0}, withoud detection: {1}".format(average_acc_on[0]*100, average_acc_on[1]*100))
        # print("Average acc off with detection: {0}, withoud detection: {1}".format(average_acc_off[0]*100, average_acc_off[1]*100))
        # print("Average acc total with detection: {0}, withoud detection: {1}".format(average_acc_total[0]*100, average_acc_total[1]*100))
        # print("Average f1 macro with detection: {0}, withoud detection: {1}".format(average_multiclass_f1_macro[0]*100, average_multiclass_f1_macro[1]*100))
        # print("Average f1 micro with detection: {0}, withoud detection: {1}".format(average_multiclass_f1_micro[0]*100, average_multiclass_f1_micro[1]*100))

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