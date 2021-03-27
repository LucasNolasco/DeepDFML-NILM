import numpy as np

class AnalysisWindow:
    def __init__(self, initSample, config):
        self.totalAnalysis = 0
        self.initSample = initSample
        self.events = []
        self.m_ngrids = config["N_GRIDS"]
        self.m_nclass = config["N_CLASS"]

        if "MIN_EVENTS" not in config:
            self.minEvents = self.m_ngrids
        else:
            self.minEvents = config["MIN_EVENTS"]

    def setTotalAnalysis(self, totalAnalysis):
        self.totalAnalysis = totalAnalysis

    def add(self, event):
        if self.totalAnalysis < self.m_ngrids:
            self.events.append(event)

    def move(self):
        self.totalAnalysis += 1

    def isFinished(self):
        if self.totalAnalysis >= self.m_ngrids:
            return True
        else:
            return False

    def compileResults(self):
        finalEvent = [[0, 0], [0, 0], [0, 0]]
        evType = np.zeros((3, 1))
        evClass = np.zeros((self.m_nclass, 1))

        if self.isFinished():
            if len(self.events) >= self.minEvents:
                for ev in self.events:
                    finalEvent[0][0] += ev[0][0] / len(self.events)
                    finalEvent[0][1] += ev[0][1] / len(self.events)
                    finalEvent[1][1] += ev[1][1] / len(self.events)
                    finalEvent[2][1] += ev[2][0][1] / len(self.events)
                    
                    evType[int(ev[1][0])] += 1
                    evClass[int(ev[2][0][0])] += 1

                finalEvent[1][0] = np.argmax(evType)
                finalEvent[2][0] = np.argmax(evClass)
            else:
                finalEvent = [[0, 1], [2, 1], [-1, 1]]

        return finalEvent