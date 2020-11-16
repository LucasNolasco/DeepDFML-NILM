import numpy as np

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
        
        #loads = np.reshape(np.arange(0, self.m_nclass), (self.m_nclass, 1))
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
            if event_type[grid][0] == 0:
                if grid > 0 and len(classification[grid]) != 0:
                    max_diff = 0
                    event = [-1, 0]
                    for load, previous_load in zip(np.max(classification[grid:],axis=0), np.max(classification[:grid],axis=0)):
                        if abs(load[1] - previous_load[1]) > abs(max_diff) and load[1] >= 0.15:
                            max_diff = load[1] - previous_load[1]
                            event[0] = load[0]
                            event[1] = load[1]
                        
                    if max_diff < 0:
                        max_diff *= -1
                        #event_type[grid][0] = 1 - event_type[grid][0]
        
                    if event[0] != -1:
                        events.append([[detection[grid], 1], event_type[grid], [event]])
                        #events.append([[detection[grid], 1], event_type[grid], classification[grid]])
                    
                else:
                    max_diff = 0
                    event = [-1, 1]
                    for load in classification[grid]:
                        if event[1] - load[1] > max_diff and load[1] > 0.1:
                            max_diff = event[1] - load[1]
                            event[0] = load[0]
                            event[1] = load[1]

                    #events.append([[detection[grid], 1], event_type[grid], [event]])                
                    events.append([[detection[grid], 1], event_type[grid], classification[grid]])
            
            elif event_type[grid][0] == 1:
                if grid < self.m_ngrids - 1 and len(classification[grid]) != 0:
                    max_diff = 0
                    event = [-1, 1]
                    for load, next_load in zip(np.max(classification[:grid + 1], axis=0), np.max(classification[grid + 1:],axis=0)):
                        if abs(load[1] - next_load[1]) > abs(max_diff) and load[1] >= 0.1:
                            max_diff = load[1] - next_load[1]
                            event[0] = load[0]
                            event[1] = load[1]

                    if max_diff < 0:
                        max_diff *= -1
                        #event_type[grid][0] = 1 - event_type[grid][0]
                    
                    if(event[0] != -1):
                        events.append([[detection[grid], 1], event_type[grid], [event]])
                        #events.append([[detection[grid], 1], event_type[grid], classification[grid]])
                    
                else:
                    max_diff = 0
                    event = [-1, 1]
                    for load in classification[grid]:
                        if event[1] - load[1] > max_diff and load[1] > 0.1:
                            max_diff = event[1] - load[1]
                            event[0] = load[0]
                            event[1] = load[1]

                    #events.append([[detection[grid], 1], event_type[grid], [event]])
                    events.append([[detection[grid], 1], event_type[grid], classification[grid]])

            '''
            i = 0
            while i < len(events):
                j = 0
                while j < len(events):
                    #if events[j][2][0] == events[i][2][0] and i != j:
                    if i != j:
                        if np.sum(events[j][2],axis=0)[1] > np.sum(events[i][2],axis=0)[1]:
                            del events[i]
                            break
                        else:
                            del events[j]
                    j += 1
                i += 1
            '''

        return np.array(events)
    
    def checkModel(self, model, x_test, y_test):
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
            result = [[[0, 0, 0, 0, 0]], result[0][0], result[1][0]]

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
                    
                    print(detection, event_type, classification, truth_detection, truth_type, truth_classification)
                    #print(raw_type[groundTruth_grid][1], raw_classification[groundTruth_grid][1])
                
                print("----------------------")

        print("Wrong: %d, Total: %d" % (total_wrong, total_events))
        print("Accuracy: %.2f, Detection Accuracy: %.2f, Classification Accuracy: %.2f, Event Type Accuracy: %.2f" % (100 * totally_correct/total_events, 100 * detection_correct/total_events, 100 * classification_correct/total_events, 100 * type_correct/total_events))
        print("Average Detection Error: %.2f, Std Deviation: %.2f" % (np.average(detection_error), np.std(detection_error, ddof=1)))