import scipy.io
import numpy as np
import h5py
import glob

def main():
    arq = h5py.File("Synthetic_1_iHall_vGrid.hdf5", "w")
    data_vGrid = np.array([])
    data_iHall = np.array([])
    #data_iShunt = np.array([])
    events = np.array([])
    labels = np.array([])
    angle = np.array([])

    begin = True

    address = glob.glob("Acquisitions/*.mat")

    MAX_SAMPLE_SIZE = 461000

    for addr in address:
        mat = scipy.io.loadmat(addr)

        lab = addr.replace("Acquisitions\\Struct", "")
        lab = lab.replace(".mat", "")
        lab = int(lab)

        print(addr, lab)

        for waveform_num in range(16):
            vGrid = mat['waveform' + str(waveform_num)]['vGrid'][0][0][:MAX_SAMPLE_SIZE]
            #iShunt = mat['waveform' + str(waveform_num)]['iShunt'][0][0]
            iHall = mat['waveform' + str(waveform_num)]['iHall'][0][0][:MAX_SAMPLE_SIZE]
            events_r = mat['waveform' + str(waveform_num)]['events_r'][0][0][:MAX_SAMPLE_SIZE]

            if begin:
                labels = np.expand_dims(np.array([lab]), axis = 0)
                #data_vGrid = np.expand_dims(vGrid, axis = 0)
                data_iHall = np.expand_dims(iHall * vGrid, axis = 0)
                #data_iShunt = np.expand_dims(iShunt, axis = 0)
                events = np.expand_dims(events_r, axis = 0)
                angle = np.expand_dims(np.array([waveform_num]), axis = 0)

                begin = False
            else:
                labels = np.vstack((labels, np.expand_dims(np.array([lab]), axis = 0))) # A correspondencia pode dar problema depois
                #data_vGrid = np.vstack((data_vGrid, np.expand_dims(vGrid, axis = 0)))
                data_iHall = np.vstack((data_iHall, np.expand_dims(iHall * vGrid, axis = 0)))
                #data_iShunt = np.vstack((data_iShunt, np.expand_dims(iShunt, axis = 0)))
                events = np.vstack((events, np.expand_dims(events_r, axis = 0)))
                angle = np.vstack((angle, np.expand_dims(np.array([waveform_num]), axis = 0)))

            print(data_iHall.shape)

    #arq.create_dataset("vGrid", data = data_vGrid)
    #arq.create_dataset("iShunt", data = data_iShunt)
    arq.create_dataset("i", data = data_iHall)
    arq.create_dataset("events", data = events)
    arq.create_dataset("angle", data = angle)
    arq.create_dataset("labels", data = labels)

    arq.close()

if __name__ == '__main__':
    main()