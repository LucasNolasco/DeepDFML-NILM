import h5py
import numpy as np
from random import randrange
from sklearn.model_selection import train_test_split
from SignalProcessing import SignalProcessing
from ExponentialDetection import ExponentialDetection
from tqdm import tqdm

class DataHandler:
    def __init__(self, configs):
        try:
            self.m_ngrids = configs["N_GRIDS"]
            self.m_nclass = configs["N_CLASS"]
            self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
            self.m_gridLength = int(self.m_signalBaseLength / self.m_ngrids)
            self.configs = configs
            self.m_datasetPath = configs["DATASET_PATH"]

            self.exponential = ExponentialDetection()

        except:
            print("Erro no dicionário de configurações")
            exit(-1)

    def loadData(self, loads_list=None, augmentation_ratio=1, SNR=None):
        arq = h5py.File(self.m_datasetPath, "r")

        x = np.array([])
        ydet = np.array([])
        yclass = np.array([])
        ytype = np.array([])
        
        if loads_list is None:
            loads_list = arq.keys()

        for load_qtd in loads_list:
            print("Loading %s" % (load_qtd))
            rawSamples = arq[load_qtd]["i"]
            rawEvents = arq[load_qtd]["events"]    
            rawLabels = arq[load_qtd]["labels"]                 

            comb_x, comb_ydet, comb_yclass, comb_ytype = self.cutData(rawSamples, rawEvents, rawLabels, augmentation_ratio, SNR)
            print(comb_x.shape)

            if 0 == x.size:
                x = np.copy(comb_x)
                ydet = np.copy(comb_ydet)
                yclass = np.copy(comb_yclass)
                ytype = np.copy(comb_ytype)
            else:
                x = np.vstack((x, comb_x))
                ydet = np.vstack((ydet, comb_ydet))
                yclass = np.vstack((yclass, comb_yclass))
                ytype = np.vstack((ytype, comb_ytype))

        arq.close()

        return x, ydet, yclass, ytype
    
    def calcEventsDuration(self, event, label):
        events_samples = np.argwhere(event != 0)
        if len(events_samples) != len(label):
            print("Quantidade de eventos encontrados não corresponde ao esperado")
            exit(-1)
        
        label_events_tuple = list(zip(label, np.transpose(events_samples)[0])) # cria tuplas com a coordenada do evento e o seu respectivo label (label, amostra de ocorrencia do evento)
        events_duration = []
        while len(label_events_tuple) != 0:
            for i in range(1, len(label_events_tuple)):
                if(label_events_tuple[0][0] == label_events_tuple[i][0]):
                    events_duration.append([label_events_tuple[0][0], label_events_tuple[0][1], label_events_tuple[i][1]])
                    del label_events_tuple[i]
                    del label_events_tuple[0]
                    break
        
        return events_samples, events_duration

    def mapSignal(self, event, events_duration, initSample, eventSample):
        out_detection = np.zeros((self.m_ngrids, 1))
        out_classification = np.zeros((self.m_ngrids, self.m_nclass))
        out_type = np.zeros((self.m_ngrids, 3))   

        '''
        for i in range(N_GRIDS):
            out_classification[i][N_CLASS] = 1
        '''

        for grid in range(self.m_ngrids):
            if eventSample >= initSample + (grid * self.m_gridLength) and eventSample < initSample + (grid + 1) * self.m_gridLength:
                out_detection[grid][0] = (eventSample - (initSample + (grid * self.m_gridLength))) / self.m_gridLength
                if event[eventSample] == 1: # ON
                    out_type[grid][0] = 1
                elif event[eventSample] == -1: # OFF
                    out_type[grid][1] = 1
                else:
                    out_type[grid][2] = 1    
            else:
                out_type[grid][2] = 1
            
            for load in events_duration:
                begin_coord = initSample + (grid * self.m_gridLength)
                end_coord = begin_coord + self.m_gridLength
                #out_classification[grid][load[0]] = max(out_classification[grid][load[0]], (min(end_coord, load[2]) - max(begin_coord, load[1])) / gridLength)
                if max(out_classification[grid][load[0]], (min(end_coord, load[2]) - max(begin_coord, load[1])) / self.m_gridLength) > 0:
                    out_classification[grid][load[0]] = 1
        
        return out_detection, out_type, out_classification

    # Pega os dados originais e faz recortes para diminuir o tamanho (Necessário para diminuir o tamanho do modelo)
    def cutData(self, rawSamples, rawEvents, rawLabels, augmentation_ratio, SNR):
        output_x = np.array([])
        output_detection = np.array([])
        output_classification = np.array([])
        output_type = np.array([])

        for sample, event, label in tqdm(zip(rawSamples, rawEvents, rawLabels)):
            events_samples, events_duration = self.calcEventsDuration(event, label)

            if SNR is not None: # Adiciona ruído ao sinal
                sample = SignalProcessing.awgn(sample, SNR)

            for ev in events_samples:
                eventSample = ev[0]

                margin_ratio = 0
                if "MARGIN_RATIO" in self.configs:
                    margin_ratio = self.configs["MARGIN_RATIO"]

                initSample = eventSample - randrange(0, self.m_signalBaseLength)
                out_detection, out_type, out_classification = self.mapSignal(event, events_duration, initSample, eventSample)
                signal = sample[initSample - int(margin_ratio * self.m_signalBaseLength) : \
                                initSample + self.m_signalBaseLength + int(margin_ratio * self.m_signalBaseLength)]

                if output_x.size == 0:
                    output_x = np.expand_dims(signal, axis = 0)
                    output_detection = np.expand_dims(out_detection, axis=0)
                    output_classification = np.expand_dims(out_classification, axis=0)
                    output_type = np.expand_dims(out_type, axis=0)
                else:
                    output_x = np.vstack((output_x, np.expand_dims(signal, axis = 0)))
                    output_detection = np.vstack((output_detection, np.expand_dims(out_detection, axis=0)))
                    output_classification = np.vstack((output_classification, np.expand_dims(out_classification, axis=0)))
                    output_type = np.vstack((output_type, np.expand_dims(out_type, axis=0)))
            

            # =========================================================================================================
            detected_events = self.exponential.detection(sample)

            last_event = 0
            valid, total = 0, 0
            for ev in detected_events:
                total += 1
                eventSample = ev[0]

                invalid_event = False
                for true_event in events_samples: # Check if this event is equivalent to any of the true events
                    if abs(eventSample - true_event[0]) < 12800 or \
                        eventSample < 12800 + 2 * int(margin_ratio * self.m_signalBaseLength) or \
                        eventSample > sample.shape[0]  - (12800 + 2 * int(margin_ratio * self.m_signalBaseLength)):
                        
                        invalid_event = True

                if invalid_event or abs(eventSample - last_event) < 12800: # If this is an invalid event, goes to the next one
                    continue

                valid += 1
                last_event = eventSample

                margin_ratio = 0
                if "MARGIN_RATIO" in self.configs:
                    margin_ratio = self.configs["MARGIN_RATIO"]

                initSample = eventSample - randrange(0, self.m_signalBaseLength)
                out_detection, out_type, out_classification = self.mapSignal(event, events_duration, initSample, -1)
                signal = sample[initSample - int(margin_ratio * self.m_signalBaseLength) : \
                                initSample + self.m_signalBaseLength + int(margin_ratio * self.m_signalBaseLength)]

                if sum(np.max(out_classification, axis=0)) == 0:
                    valid -= 1
                    continue

                if output_x.size == 0:
                    output_x = np.expand_dims(signal, axis = 0)
                    output_detection = np.expand_dims(out_detection, axis=0)
                    output_classification = np.expand_dims(out_classification, axis=0)
                    output_type = np.expand_dims(out_type, axis=0)
                else:
                    output_x = np.vstack((output_x, np.expand_dims(signal, axis = 0)))
                    output_detection = np.vstack((output_detection, np.expand_dims(out_detection, axis=0)))
                    output_classification = np.vstack((output_classification, np.expand_dims(out_classification, axis=0)))
                    output_type = np.vstack((output_type, np.expand_dims(out_type, axis=0)))


            print(f"Found {total} events, but only {valid} is/are valid!")
            # =========================================================================================================
        
        return output_x, output_detection, output_classification, output_type
    
    def checkGridDistribution(self, y_train, y_test):
        dict_train = {}
        dict_test = {}
        for i in range(0, self.m_ngrids):
            dict_train[str(i)] = 0
            dict_test[str(i)] = 0

        for y in y_train["type"]:
            for i in range(self.m_ngrids):
                if np.argmax(y[i]) != 2:
                    dict_train[str(i)] += 1

        for y in y_test["type"]:
            for i in range(self.m_ngrids):
                if np.argmax(y[i]) != 2:
                    dict_test[str(i)] += 1

        return dict_train, dict_test

    def generateAcquisitionType(self, trainSize, distribution=None, augmentation=1):
        if distribution is None:
            # 1: 832
            # 2: 2688
            # 3: 3168
            # 4: 1728

            distribution = {
                "1": 832,
                "2": 2688,
                "3": 3168,
                "8": 1728
            }

        general = []
        for k in range(augmentation):
            for i in range(0, distribution["1"]):
                general.append(1)
            for i in range(0, distribution["2"]):
                general.append(2)
            for i in range(0, distribution["3"]):
                general.append(3)
            for i in range(0, distribution["8"]):
                general.append(8)

        general_qtd_train, general_qtd_test = train_test_split(general, train_size=trainSize, random_state=42)
        return general_qtd_train, general_qtd_test

    def checkAcquisitionType(self, yclass, load_type, general_qtd):
        correct_order = 0
        for classification, qtd, g_qtd in zip(yclass, load_type, general_qtd):
            if sum(np.max(classification, axis=0) > 0.5) == qtd:
                correct_order += 1
            else:
                print(sum(np.max(classification, axis=0) > 0.5), qtd, g_qtd)

        print("Correct Order: {0}, Total: {1}".format(correct_order, len(yclass)))