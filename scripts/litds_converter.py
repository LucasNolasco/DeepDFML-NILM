import scipy.io
import numpy as np
import h5py
import glob

labels_dict = {"0": 26, "A0": 0, "B0": 1, "C0": 2, "D0": 3, "E0": 4, "F0": 5, "G0": 6, "H0": 7, "I0": 8, "J0": 9, \
               "K0": 10, "L0": 11, "M0": 12, "N0": 13, "O0": 14, "P0": 15, "Q0": 16, "R0": 17, "S0": 18, "T0": 19, \
               "U0": 20, "V0": 21, "W0": 22, "X0": 23, "Y0": 24, "Z0": 25}

synthetic_1_order = ['1A0', '1B0', '1C0', '1D0', '1E0', '1F0', '1G0', '1H0', '1I0', '1J0', '1K0', '1L0', '1M0', '1N0', '1O0', '1P0', '1Q0', '1R0', '1S0', '1T0', '1U0', '1V0', '1W0', '1X0', '1Y0', '1Z0']
synthetic_2_order = ['2A0B0', '2A0H0', '2A0L0', '2A0M0', '2A0N0', '2A0Q0', '2B0A0', '2B0H0', '2B0L0', '2B0M0', '2B0N0', '2B0Q0', '2H0A0', '2H0B0', '2H0L0', '2H0M0', '2H0N0', '2H0Q0', '2L0A0', '2L0B0', '2L0H0', '2L0M0', '2L0N0', '2L0Q0', '2M0A0', '2M0B0', '2M0H0', '2M0L0', '2M0N0', '2M0Q0', '2N0A0', '2N0B0', '2N0H0', '2N0L0', '2N0M0', '2N0Q0', '2Q0A0', '2Q0B0', '2Q0H0', '2Q0L0', '2Q0M0', '2Q0N0']
synthetic_3_order = ['3B0K0D0', '3D0Q0I0', '3D0Q0N0', '3D0Y0S0', '3E0N0Q0', '3H0P0W0', '3M0R0W0', '3M0R0X0', '3M0W0I0', '3N0S0I0', '3P0U0Z0', '3P0Z0H0', '3Q0K0B0', '3Q0X0E0', '3R0E0S0', '3S0E0Y0', '3S0I0Y0', '3T0Q0N0', '3T0V0M0', '3U0N0Z0', '3U0R0Y0', '3U0T0H0', '3V0D0M0', '3V0H0X0', '3V0M0Q0', '3W0E0H0', '3W0E0T0', '3W0I0Z0', '3X0D0P0', '3X0N0R0', '3Y0P0U0', '3Y0T0E0', '3Z0P0V0']
synthetic_8_order = ['8D0G0P0Q0M0N0H0E0', '8D0M0S0G0H0N0R0E0', '8E0P0I0M0N0H0W0Y0', '8I0E0H0D0M0N0U0Z0', '8Q0H0N0M0P0E0I0V0', '8X0E0H0I0M0P0N0D0']

MAX_SAMPLE_SIZE = {"1": 461000, "2": 461000, "3": 461000, "8": 580000} #461000

synthetic_1_sequence = [1, 1]
synthetic_2_sequence = [1, 2, 1, 2]
synthetic_3_sequence = [1, 2, 3, 1, 2, 3]
synthetic_8_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 3, 5, 7, 1, 8, 2, 6, 1, 4, 1]

synthetic_events_sequence = {"1": synthetic_1_sequence, \
                             "2": synthetic_2_sequence, \
                             "3": synthetic_3_sequence, \
                             "8": synthetic_8_sequence}

synthetic_files_order = {"1": synthetic_1_order, \
                         "2": synthetic_2_order, \
                         "3": synthetic_3_order, \
                         "8": synthetic_8_order}

def main():
    arq = h5py.File("Synthetic_Full_Power.hdf5", "w")

    for loads_qtd in ["1", "2", "3", "8"]:
        #data_vGrid = np.array([])
        data_iHall = np.array([])
        #data_iShunt = np.array([])
        events = np.array([])
        labels = np.array([])
        angle = np.array([])

        begin = True

        loads_qtd_group = arq.create_group(loads_qtd)
        address = glob.glob("Acquisitions/" + loads_qtd +"/*.mat")
        for addr in address:
            mat = scipy.io.loadmat(addr, chars_as_strings = False)

            identification = addr.replace("Acquisitions/"+ loads_qtd + "\\Struct", "")
            identification =  identification.replace(".mat", "")
            identification = int(identification)

            identification = synthetic_files_order[loads_qtd][identification]
            lab = []
            for v in synthetic_events_sequence[loads_qtd]:
                lab.append(labels_dict[identification[2 * v - 1 : 2 * v + 1]])
            lab = np.array(lab)
            #lab = np.array([labels_dict[identification[1:3]], labels_dict[identification[3:5]], \
            #                labels_dict[identification[1:3]], labels_dict[identification[3:5]]])

            print(addr, lab)

            for waveform_num in range(16):
                vGrid = mat['waveform' + str(waveform_num)]['vGrid'][0][0][:MAX_SAMPLE_SIZE[loads_qtd]]
                #iShunt = mat['waveform' + str(waveform_num)]['iShunt'][0][0]
                iHall = mat['waveform' + str(waveform_num)]['iHall'][0][0][:MAX_SAMPLE_SIZE[loads_qtd]]
                events_r = mat['waveform' + str(waveform_num)]['events_r'][0][0][:MAX_SAMPLE_SIZE[loads_qtd]]
                #lab = mat['waveform' + str(waveform_num)]['labels'][0] #[:MAX_SAMPLE_SIZE]

                if begin:
                    labels = np.expand_dims(lab, axis = 0)
                    #data_vGrid = np.expand_dims(vGrid, axis = 0)
                    data_iHall = np.expand_dims(np.multiply(iHall,vGrid), axis = 0)
                    #data_iShunt = np.expand_dims(iShunt, axis = 0)
                    events = np.expand_dims(events_r, axis = 0)
                    angle = np.expand_dims(np.array([waveform_num]), axis = 0)

                    begin = False
                else:
                    labels = np.vstack((labels, np.expand_dims(lab, axis = 0))) # A correspondencia pode dar problema depois
                    #data_vGrid = np.vstack((data_vGrid, np.expand_dims(vGrid, axis = 0)))
                    data_iHall = np.vstack((data_iHall, np.expand_dims(np.multiply(iHall,vGrid), axis = 0)))
                    #data_iShunt = np.vstack((data_iShunt, np.expand_dims(iShunt, axis = 0)))
                    events = np.vstack((events, np.expand_dims(events_r, axis = 0)))
                    angle = np.vstack((angle, np.expand_dims(np.array([waveform_num]), axis = 0)))

                print(data_iHall.shape)

        #arq.create_dataset("vGrid", data = data_vGrid)
        #arq.create_dataset("iShunt", data = data_iShunt)
        loads_qtd_group.create_dataset("i", data = data_iHall)
        loads_qtd_group.create_dataset("events", data = events)
        loads_qtd_group.create_dataset("angle", data = angle)
        loads_qtd_group.create_dataset("labels", data = labels)

    arq.close()

if __name__ == '__main__':
    main()