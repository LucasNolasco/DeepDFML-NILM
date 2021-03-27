import scipy.io
import numpy as np
import h5py
import glob

def main():
    arq = h5py.File("Synthetic_1.hdf5", "w")

    address = glob.glob("Acquisitions/*.mat")

    for file_num, addr in enumerate(address):
        mat = scipy.io.loadmat(addr)

        dataset = arq.create_group('1' + chr(ord('A') + file_num) + '0')

        for waveform_num in range(16):
            group = dataset.create_group('waveform' + str(waveform_num))
            duration_t = np.array(mat['waveform' + str(waveform_num)]['duration_t'][0][0])
            vGrid = mat['waveform' + str(waveform_num)]['vGrid'][0][0]
            iShunt = mat['waveform' + str(waveform_num)]['iShunt'][0][0]
            iHall = mat['waveform' + str(waveform_num)]['iHall'][0][0]

            group.create_dataset("vGrid", data = vGrid)
            group.create_dataset("iShunt", data = iShunt)
            group.create_dataset("iHall", data = iHall)
            group.create_dataset("duration_t", data = duration_t)

    arq.close()

if __name__ == '__main__':
    main()