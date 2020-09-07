from keras.models import load_model
import AnalysisWindow
import numpy as np
import h5py
import ModelHandler
import PostProcessing
import matplotlib.pyplot as plt

class CheckFullSignal:
    def __init__(self, configs):
        self.m_ngrids = configs["N_GRIDS"]
        self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
        self.m_marginRatio = configs["MARGIN_RATIO"]
        self.postProcessing = PostProcessing(configs)

    def check(self, folderPath):
        arq = h5py.File("Synthetic_2_iHall.hdf5", "r")

        x = np.array(arq["i"])
        ydet = np.array(arq["events"])
        yclass = np.array(arq["labels"])

        arq.close()

        model = load_model(folderPath + 'best_model.h5', custom_objects={'sumSquaredError': ModelHandler.sumSquaredError, 'KerasFocalLoss': ModelHandler.KerasFocalLoss}) 

        for k in range(0, x.shape[0]):
            # 5: [15, 30, 47, 95, 127, 155, 161, 167, 170, 173, 192, 287, 317, 384]
            # 4: [15, 30, 47, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 95, 151, 155, 161, 162, 165, 166, 167, 168, 170, 171, 172, 173, 174, 192, 337, 339, 341, 342]
            
            title = ""
            for lab in yclass[k]:
                if title == "":
                    title = title + str(lab)
                else:
                    title = title + ", " + str(lab)
            
            title = "[" + title + "]"

            windows = []
            for i in range(self.m_ngrids):
                windows.append(AnalysisWindow(i * round(self.m_signalBaseLength / self.m_ngrids) + int(self.m_marginRatio * self.m_signalBaseLength)))
                windows[i].setTotalAnalysis(self.m_ngrids - 1 + i)

            y_res = np.zeros((x.shape[1], 1))
            for i in range(0, x.shape[1] - self.m_signalBaseLength - 2 * int(self.m_marginRatio * self.m_signalBaseLength), round(self.m_signalBaseLength / self.m_ngrids)):
                result = model.predict(np.expand_dims(x[k][i:i + self.m_signalBaseLength + 2 * int(self.m_marginRatio * self.m_signalBaseLength)], axis = 0))
                raw_detection, raw_event_type, raw_classification = self.postProcessing.processResult(result[0][0], result[1][0], result[2][0])
                events = self.postProcessing.suppression(raw_detection, raw_event_type, raw_classification)
                for ev in events:
                    ev[0][0] += i
                    #print(int(ev[0][0]), ev[0][1], int(ev[1][0]), ev[1][1], int(ev[2][0]), ev[2][1])
                    for window in windows:
                        #print(ev[0][0], window.initSample, window.initSample + round(SIGNAL_LENGTH / N_GRIDS))
                        if ev[0][0] >= window.initSample and ev[0][0] < window.initSample + round(self.m_signalBaseLength / self.m_ngrids):
                            window.add(ev)

                for window in windows:
                    window.move()

                if windows[0].isFinished():
                    ev = windows[0].compileResults()
                    if ev[1][0] == 0:
                        y_res[int(ev[0][0])] = 1
                    elif ev[1][0] == 1:
                        y_res[int(ev[0][0])] = -1

                    if ev[1][0] != 2:
                        title = title + ", " + str(int(ev[2][0]))
                    
                    windows.append(AnalysisWindow(windows[self.m_ngrids - 1].initSample + round(self.m_signalBaseLength / self.m_ngrids)))
                    del windows[0]
            
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(np.arange(0, x.shape[1]), x[k])
            ax.plot(np.arange(0, x.shape[1]), ydet[k])
            ax.plot(np.arange(0, x.shape[1]), y_res)
            fig.savefig("Imagens/%d.png" % (k))
            #plt.show()
            plt.close(fig)

            print("%d/%d" % (k + 1, x.shape[0]))
            print("------------")