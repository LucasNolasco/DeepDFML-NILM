import numpy as np

class ExponentialDetection:
    def __init__(self, gamma = 0.00001, samples_per_cycle=256, n_windows = 1):
        self.gamma = gamma
        self.samples_per_cycle = samples_per_cycle
        self.n_windows = n_windows

    def detection(self, x):
        window_size = self.samples_per_cycle * self.n_windows

        x = np.reshape(x, (x.shape[0],))

        max_index_list, max_list = [], []
        for i in range(0, x.shape[0], self.samples_per_cycle):
            max_index = i + np.argmax(abs(x[i : i + self.samples_per_cycle])) # We add i so the max_index refers to the x and not just the cut [i : i + samples]
            max_index_list.append(max_index)
            max_list.append(abs(x[max_index]))

        x_interp = np.interp(np.arange(x.shape[0]), max_index_list, max_list)

        vetor_mean = np.zeros_like(x_interp)
        vetor_std = np.zeros_like(x_interp)
        for i in range(0, x_interp.shape[0] - window_size, window_size):
            vetor_mean[i : i + window_size] = np.mean(x_interp[i : i + window_size])
            # vetor_std[i : i + window_size] = np.std(x_interp[i : i + window_size])

        for i in range(window_size, x_interp.shape[0] - window_size, window_size):
            for j in range(0, window_size):
                vetor_mean[i + j] = vetor_mean[i + j - 1] + (x_interp[i + j] -  x_interp[i + j - window_size]) / float(window_size)
                vetor_std[i + j] = vetor_std[i + j - 1] + ((x_interp[i + j] ** 2 - x_interp[i + j - window_size] ** 2) / (float(window_size) - 1)) + \
                                    ((float(window_size) / (float(window_size) - 1)) * (vetor_mean[i + j - 1] ** 2 - vetor_mean[i + j] ** 2))

        vetor_std = vetor_std / np.max(vetor_std) # Scale

        detec_signal = np.where(vetor_std > self.gamma, 1, 0)
        detec_signal = np.diff(detec_signal)

        events = np.argwhere(detec_signal == 1)

        return events