import numpy as np
import matplotlib.pyplot as plt

class Plot:
    @staticmethod
    def plotSignal(x, y_detection, y_type, y_classification):
        eventVec = np.zeros((x.shape[0], 1))
        loadClass = -1
        for j in range(0, len(y_detection)):
            if y_type[j] == 0:
                eventVec[y_detection[j]] = 1
                loadClass = y_classification[j]
            elif y_type[j] == 1:
                eventVec[y_detection[j]] = -1
                loadClass = y_classification[j]

        _, ax = plt.subplots()
        ax.set_title(str(loadClass))
        ax.plot(np.arange(0, x.shape[0]), x)
        ax.plot(np.arange(0, x.shape[0]), eventVec)
        plt.show()

    @staticmethod
    def plot(x, events, begin, end, grids_division = None):
        _, ax = plt.subplots()
        #ax.set_title(str(loadClass))
        ax.plot(np.arange(begin, end), x[begin:end])
        ax.plot(np.arange(begin, end), events[begin:end])
        if grids_division is not None:
            ax.plot(np.arange(begin, end), grids_division[begin:end])
        plt.show()