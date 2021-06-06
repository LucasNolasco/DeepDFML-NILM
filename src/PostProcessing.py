import numpy as np
import timeit
from sklearn.metrics import f1_score

class PostProcessing:
    def __init__(self, configs):
        try:
            self.m_ngrids = configs["N_GRIDS"]
            self.m_nclass = configs["N_CLASS"]
            self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
            self.m_marginRatio = configs["MARGIN_RATIO"]
            self.m_gridLength = int(self.m_signalBaseLength / self.m_ngrids)
            self.configs = configs

            if "USE_NO_LOAD" in self.configs and self.configs["USE_NO_LOAD"] == True:
                self.m_nclass += 1
        except:
            print("Erro no dicionário de configurações")
            exit(-1)

    def processResult(self, y_detection, y_type, y_classification):
        detection = []
        event_type = []
        classification = []
        
        loads = np.argwhere(np.max(y_classification, axis=0) > 0.5)
        for grid in range(self.m_ngrids):
            detection.append(int((grid + y_detection[grid][0]) * self.m_gridLength) + int(self.m_marginRatio * self.m_signalBaseLength))
            event_type.append([np.argmax(y_type[grid]), np.max(y_type[grid])])
            
            grid_classes = []
            for l in loads:
                grid_classes.append([l[0], y_classification[grid][l[0]]])
            classification.append(grid_classes)

        return detection, event_type, classification

    def suppression(self, detection, event_type, classification):
        events = []

        for grid in range(len(detection)):            
            if event_type[grid][0] != 2:
                events.append([[detection[grid], 1], event_type[grid], classification[grid]])

        if len(events) > 1:
            max_total_probability = 0
            eventsCopy = events.copy()
            events = []
            for detectedEvent in eventsCopy:
                if len(detectedEvent[2]) > 0:
                    total_probability = np.sum(detectedEvent[2],axis=0)[1]
                    if total_probability > max_total_probability:
                        events = []
                        events.append(detectedEvent.copy())
                        max_total_probability = total_probability

        return np.array(events, dtype=object)

    def suppressionMultiClass(self, detection, event_type, classification):
        events = []

        for grid in range(len(detection)):            
            if event_type[grid][0] != 2:
                # events.append([[detection[grid], 1], event_type[grid], classification[grid]])  LUCAS: OLD CONFIG FOR MULTILABEL, GOTTA CHANGE IT LATER
                if event_type[grid][0] == 0 and grid != 0:
                    #print(classification[grid - 1], classification[grid])
                    real_class = -1
                    prob = 0
                    for possible_class, possible_class_prev in zip(classification[grid], classification[grid - 1]):
                        if possible_class[1] - possible_class_prev[1] > prob:
                            real_class = possible_class[0]
                            prob = possible_class[1] - possible_class_prev[1]

                    #print(real_class, prob)
                    events.append([[detection[grid], 1], event_type[grid], [[real_class, prob]]])
                
                elif event_type[grid][0] == 1 and grid != self.m_ngrids - 1:
                    #print(classification[grid], classification[grid + 1])
                    real_class = -1
                    prob = 0
                    for possible_class, possible_class_next in zip(classification[grid], classification[grid + 1]):
                        if possible_class[1] - possible_class_next[1] > prob:
                            real_class = possible_class[0]
                            prob = possible_class[1] - possible_class_next[1]

                    #print(real_class, prob)
                    events.append([[detection[grid], 1], event_type[grid], [[real_class, prob]]])

        if len(events) > 1:
            max_total_probability = 0
            eventsCopy = events.copy()
            events = []
            for detectedEvent in eventsCopy:
                if len(detectedEvent[2]) > 0:
                    total_probability = np.sum(detectedEvent[2],axis=0)[1]
                    if total_probability > max_total_probability:
                        events = []
                        events.append(detectedEvent.copy())
                        max_total_probability = total_probability

            #print(events)

        return np.array(events, dtype=object)

    # --------------------------------------------------------------------
    #   detectEvents: Method responsible for converting the model output
    #                 to an event list, where each event is a list on the
    #                 following format:
    #                       [[eventSample, probability (for now it's always 1)],
    #                        [eventType, probability],
    #                        [[connectedLoad, probability]]]
    #
    #   Parameters: 
    #       - model: Structure of the trained model
    #       - x_test: Input data for each example
    #       - y_test: Expected output for each input (ground truth)
    #
    #   Return:
    #       - groundTruth: list with ground truth events
    #       - predicted: list with predicted events
    # --------------------------------------------------------------------
    def detectEvents(self, model, x_test, y_test, multilabel=True):
        total_time = 0
        groundTruth, predicted = [], []
        for xi, groundTruth_detection, groundTruth_type, groundTruth_classification in zip(x_test, y_test["detection"], y_test["type"], y_test["classification"]):
            init_time = timeit.default_timer()
            result = model.predict(np.expand_dims(xi, axis = 0))

            raw_detection, raw_type, raw_classification = self.processResult(result[0][0], result[1][0], result[2][0])
            raw_gt_detection, raw_gt_type, raw_gt_classification = self.processResult(groundTruth_detection, groundTruth_type, groundTruth_classification)

            if multilabel:
                predict_events = self.suppression(raw_detection, raw_type, raw_classification)
                groundTruth_events = self.suppression(raw_gt_detection, raw_gt_type, raw_gt_classification)
            else:
                predict_events = self.suppressionMultiClass(raw_detection, raw_type, raw_classification)
                groundTruth_events = self.suppressionMultiClass(raw_gt_detection, raw_gt_type, raw_gt_classification)

            final_time = timeit.default_timer()

            total_time += final_time - init_time

            predicted.append(predict_events)
            groundTruth.append(groundTruth_events)

        print("Total time: {0}, Average Time: {1}".format(total_time, total_time/x_test.shape[0]))

        return groundTruth, predicted

    # --------------------------------------------------------------------
    #   PCMetric: Method to calculate the metrics PCon and PCoff.
    #
    #   Parameters: 
    #       - groundTruth: List of ground truth events
    #       - predicted: List of predicted events
    #
    #   Return:
    #       - PCon: Ratio between ON events detected within 10 semicycles
    #               of distance and the total number of events;
    #       - PCoff: Same as PCon but considering OFF events;
    #       - PCtotal: Same as PCon but considering both ON and OFF;
    # --------------------------------------------------------------------
    def PCMetric(self, groundTruth, predicted, general_acquisition_type=None, target=None):
        total_on = 0
        total_off = 0
        correct_on = 0
        correct_off = 0
        i = -1
        for gt_events, pred_events in zip(groundTruth, predicted):
            i += 1
            if general_acquisition_type is not None and target is not None:
                if general_acquisition_type[i] != target:
                    continue
    
            if len(gt_events) == 0:
                continue

            if gt_events[0][1][0] == 0:
                total_on += 1
                if len(pred_events) > 0 and gt_events[0][1][0] == pred_events[0][1][0] and abs(gt_events[0][0][0] - pred_events[0][0][0]) < (128 * 10): # 10 semicycles tolerance
                    correct_on += 1
            elif gt_events[0][1][0] == 1:
                total_off += 1
                if len(pred_events) > 0 and gt_events[0][1][0] == pred_events[0][1][0] and abs(gt_events[0][0][0] - pred_events[0][0][0]) < (128 * 10): # 10 semicycles tolerance
                    correct_off += 1

        PCon = (correct_on / total_on)
        PCoff = (correct_off / total_off)
        PCtotal = ((correct_on + correct_off) / (total_off + total_on))

        return PCon, PCoff, PCtotal

    # --------------------------------------------------------------------
    #   averageDistanceMetric: Method to calculate the metrics Don and Doff
    #
    #   Parameters: 
    #       - groundTruth: List of ground truth events
    #       - predicted: List of predicted events
    #
    #   Return:
    #       - Don: Average distance between an ON event and its detection
    #       - Doff: Average distance between an OFF event and its detection
    #       - Dtotal: Average distance between all events and their detection
    # --------------------------------------------------------------------
    def averageDistanceMetric(self, groundTruth, predicted, general_acquisition_type=None, target=None):
        total_correct_on = 0
        total_correct_off = 0
        distance_sum_on = 0
        distance_sum_off = 0
        i = -1
        for gt_events, pred_events in zip(groundTruth, predicted):
            i += 1
            if general_acquisition_type is not None and target is not None:
                if general_acquisition_type[i] != target:
                    continue

            if len(gt_events) == 0:
                continue
    
            if gt_events[0][1][0] == 0:
                if len(pred_events) > 0 and gt_events[0][1][0] == pred_events[0][1][0] and abs(gt_events[0][0][0] - pred_events[0][0][0]) < (128 * 10): # 10 semicycles tolerance
                    total_correct_on += 1
                    distance_sum_on += abs(round(gt_events[0][0][0]/128) - round(pred_events[0][0][0]/128))
            elif gt_events[0][1][0] == 1:
                if len(pred_events) > 0 and gt_events[0][1][0] == pred_events[0][1][0] and abs(gt_events[0][0][0] - pred_events[0][0][0]) < (128 * 10): # 10 semicycles tolerance
                    total_correct_off += 1
                    distance_sum_off += abs(round(gt_events[0][0][0]/128) - round(pred_events[0][0][0]/128))

        Don = distance_sum_on / total_correct_on
        Doff = distance_sum_off / total_correct_off
        Dtotal = (distance_sum_on + distance_sum_off) / (total_correct_on + total_correct_off)

        return Don, Doff, Dtotal

    def checkMultiClassAccuracy(self, model, x, y, load_type=None, general_qtd=None):
        acc_on = np.zeros((26,))
        no_det_acc_on = np.zeros((26,))
        total_on = np.zeros((26,))
        acc_off = np.zeros((26,))
        total_off = np.zeros((26,))
        no_det_acc_off = np.zeros((26,))

        groundTruth, predicted = self.detectEvents(model, x, y, multilabel=False)
        for gt, pred in zip(groundTruth, predicted):
            if len(gt) == 0:
                continue

            # TOTAL EVENTS PER CLASS
            if gt[0][1][0] == 0:
                total_on[gt[0][2][0][0]] += 1
            elif gt[0][1][0] == 1:
                total_off[gt[0][2][0][0]] += 1

            if len(pred) == 0:
                continue

            # CORRECT EVENTS WITHIN DETECTION MARGIN
            if gt[0][1][0] == pred[0][1][0] and abs(gt[0][0][0] - pred[0][0][0]) < 128 * 10:
                if gt[0][2][0][0] == pred[0][2][0][0]:
                    if gt[0][1][0] == 0:
                        acc_on[gt[0][2][0][0]] += 1
                    elif gt[0][1][0] == 1:
                        acc_off[gt[0][2][0][0]] += 1

            # CORRECT EVENTS NOT CONSIDERING DETECTION
            if gt[0][2][0][0] == pred[0][2][0][0]:
                if gt[0][1][0] == 0:
                    no_det_acc_on[gt[0][2][0][0]] += 1
                elif gt[0][1][0] == 1:
                    no_det_acc_off[gt[0][2][0][0]] += 1

        loads_gt, loads_pred = [], []
        loads_pred_with_detection = []
        for gt, pred in zip(groundTruth, predicted):
            if len(gt) == 0:
                continue

            new_gt = gt[0][2][0][0]
            loads_gt.append(new_gt)

            new_pred_with_detection = 26
            if len(pred) != 0 and gt[0][1][0] == pred[0][1][0] and abs(gt[0][0][0] - pred[0][0][0]) < 128 * 10:
                new_pred_with_detection = pred[0][2][0][0]
            
            loads_pred_with_detection.append(new_pred_with_detection)

            new_pred = 26
            if len(pred) != 0:
                new_pred = pred[0][2][0][0]

            loads_pred.append(new_pred)

        final_acc_on = np.average(np.nan_to_num(acc_on/total_on, nan=1), axis=0)
        final_acc_off = np.average(np.nan_to_num(acc_off/total_off, nan=1), axis=0)
        final_no_det_acc_on = np.average(np.nan_to_num(no_det_acc_on/total_on, nan=1), axis=0)
        final_no_det_acc_off = np.average(np.nan_to_num(no_det_acc_off/total_off, nan=1), axis=0)
        final_acc = np.average(np.nan_to_num((acc_on + acc_off)/(total_on + total_off), nan=1), axis=0)
        final_acc_no_det = np.average(np.nan_to_num((no_det_acc_on + no_det_acc_off)/(total_on + total_off), nan=1), axis=0)

        f1_macro_without_detection = f1_score(loads_gt, loads_pred, labels=range(0,26), average='macro')
        f1_micro_without_detection = f1_score(loads_gt, loads_pred, labels=range(0,26), average='micro')

        f1_macro_with_detection = f1_score(loads_gt, loads_pred_with_detection, labels=range(0,26), average='macro')
        f1_micro_with_detection = f1_score(loads_gt, loads_pred_with_detection, labels=range(0,26), average='micro')

        return [final_acc_on, final_no_det_acc_on], [final_acc_off, final_no_det_acc_off], [final_acc, final_acc_no_det], [f1_macro_with_detection, f1_macro_without_detection], [f1_micro_with_detection, f1_micro_without_detection]
    
    def checkModel(self, model, x, y, general_qtd=None, print_error=True):
        groundTruth, predicted = self.detectEvents(model, x, y)

        PCmetric = []
        Dmetric = []

        if general_qtd is not None:
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd, target=1))
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd, target=2))
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd, target=3))
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd, target=8))
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd))

            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd, target=1))
            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd, target=2))
            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd, target=3))
            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd, target=8))
            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd))

            print("LIT-SYN-1 PCmetric: {0}".format(PCmetric[0]))
            print("LIT-SYN-1 Dmetric: {0}".format(Dmetric[0]))

            print("LIT-SYN-2 PCmetric: {0}".format(PCmetric[1]))
            print("LIT-SYN-2 Dmetric: {0}".format(Dmetric[1]))

            print("LIT-SYN-3 PCmetric: {0}".format(PCmetric[2]))
            print("LIT-SYN-3 Dmetric: {0}".format(Dmetric[2]))

            print("LIT-SYN-8 PCmetric: {0}".format(PCmetric[3]))
            print("LIT-SYN-8 Dmetric: {0}".format(Dmetric[3]))

            print("LIT-SYN-All PCmetric: {0}".format(PCmetric[4]))
            print("LIT-SYN-All Dmetric: {0}".format(Dmetric[4]))
        
        else:
            PCmetric.append(self.PCMetric(groundTruth, predicted, general_qtd))
            Dmetric.append(self.averageDistanceMetric(groundTruth, predicted, general_qtd))

            print("LIT-SYN-All PCmetric: {0}".format(PCmetric[-1]))
            print("LIT-SYN-All Dmetric: {0}".format(Dmetric[-1]))

        return PCmetric, Dmetric

    def f1_with_detection(self, model, x, y, general_acquisition_type=None, target=None, print_error=True):
        groundTruth, predicted = self.detectEvents(model, x, y)

        groundTruthLoads, predictedLoads = [], []

        i = -1
        for gt_events, pred_events in zip(groundTruth, predicted):
            i += 1
            if general_acquisition_type is not None and target is not None:
                if general_acquisition_type[i] != target:
                    continue

            pred = np.zeros((26, 1))
            if gt_events[0][1][0] != 2:
                if len(pred_events) > 0 and gt_events[0][1][0] == pred_events[0][1][0] and abs(gt_events[0][0][0] - pred_events[0][0][0]) < (128 * 10): # 10 semicycles tolerance
                    for c in pred_events[0][2]:
                        pred[c[0]] = 1

            gt = np.zeros((26,1))
            for c in gt_events[0][2]:
                gt[c[0]] = 1

            predictedLoads.append(pred.T[0])
            groundTruthLoads.append(gt.T[0])

        return f1_score(groundTruthLoads, predictedLoads, average='macro'), f1_score(groundTruthLoads, predictedLoads, average='micro')

    # --------------------------------------------------------------------
    #   fullCheckModel: Old checkModel
    #
    #   Parameters: 
    #       - model: Structure of the trained model
    #       - x_test: Input data for each example
    #       - y_test: Expected output for each input (ground truth)
    #       - print_error: Flag to indicate if errors should be printed (helps on debugging)
    #
    # --------------------------------------------------------------------
    def fullCheckModel(self, model, x_test, y_test, print_error=True):
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

            raw_detection, raw_type, raw_classification = self.processResult(result[0][0], result[1][0], result[2][0])
            raw_gt_detection, raw_gt_type, raw_gt_classification = self.processResult(groundTruth_detection, groundTruth_type, groundTruth_classification)

            predict_events = self.suppression(raw_detection, raw_type, raw_classification)
            groundTruth_events = self.suppression(raw_gt_detection, raw_gt_type, raw_gt_classification)

            event_outside_margin = False

            '''
            for groundTruth in groundTruth_events:
                if groundTruth[0][0] > MARGIN_RATIO * SIGNAL_LENGTH + SIGNAL_LENGTH * (N_GRIDS - 1) / N_GRIDS or \
                groundTruth[0][0] < MARGIN_RATIO * SIGNAL_LENGTH + SIGNAL_LENGTH * 1 / N_GRIDS:
                    event_outside_margin = True
            '''
            
            if not event_outside_margin:
                if len(predict_events) == len(groundTruth_events):
                    for prediction, groundTruth in zip(predict_events, groundTruth_events):
                        total_events += 1

                        if len(prediction[2]) == len(groundTruth[2]):
                            correct_flag = True
                            for pred, gt in zip(prediction[2], groundTruth[2]):
                                if pred[0] != gt[0]:
                                    correct_flag = False
                        else:
                            correct_flag = False

                        if correct_flag:
                                classification_correct += 1

                        if abs(prediction[0][0] - groundTruth[0][0]) < 256:
                            detection_correct += 1
                            if correct_flag and prediction[1][0] == groundTruth[1][0]:
                                totally_correct += 1
                        #if prediction[2][0] == groundTruth[2][0]:
                        #    classification_correct += 1

                        if prediction[1][0] == groundTruth[1][0]:
                            type_correct += 1

                        if(groundTruth[1][0] != prediction[1][0]) or not correct_flag:
                            if print_error:
                                print(prediction[2], groundTruth[2], len(predict_events))
                            error = True

                        detection_error.append(abs(prediction[0][0] - groundTruth[0][0]))
                else:
                    total_events += len(groundTruth_events)
                    error = True

            if error:
                total_wrong += 1
                for groundTruth_grid in range(self.m_ngrids):
                    detection = raw_detection[groundTruth_grid]
                    event_type = raw_type[groundTruth_grid][0]
                    classification = raw_classification[groundTruth_grid]

                    truth_detection = raw_gt_detection[groundTruth_grid]
                    truth_type = raw_gt_type[groundTruth_grid][0]
                    truth_classification = raw_gt_classification[groundTruth_grid]
                    
                    if print_error:
                        print(detection, event_type, classification, truth_detection, truth_type, truth_classification)
                        #print(raw_type[groundTruth_grid][1], raw_classification[groundTruth_grid][1])
                
                if print_error:
                    print("----------------------")

        print("Wrong: %d, Total: %d" % (total_wrong, total_events))
        print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/total_events, 100 * detection_correct/total_events, 100 * classification_correct/total_events, 100 * type_correct/total_events))
        print("Average Detection Error: %.2f, Std Deviation: %.2f" % (np.average(detection_error), np.std(detection_error, ddof=1)))