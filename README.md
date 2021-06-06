# DeepDFML-NILM

We propose a new CNN architecture to perform detection, feature extraction, and multi-label classification of loads, in non-intrusive load monitoring (NILM) approaches, with a single model for high-frequency signals. This model follows the idea of YOLO network, which provides the detection and multi-label classification of images. The obtained results are equivalent or superior (in most analyzed cases) to state-of-the-art methods for the evaluated datasets.

![Architecture](.images/architecture.png)

## Dependencies

The model was implemented on Python 3 with the following libraries:

- h5py:

    ```
    $ pip3 install h5py
    ```

- numpy:
    ```
    $ pip3 install numpy
    ```

- tqdm:
    ```
    $ pip3 install tqdm
    ```

- scikit-learn:
    ```
    $ pip3 install sklearn
    ```

- keras:
    ```
    $ pip3 install keras
    ```

- tensorflow:
    ```
    $ pip3 install tensorflow
    ```

- matplotlib:
    ```
    $ pip3 install matplotlib
    ```

## Dict structure

This implementation uses a dict structure to define some of the execution parameters. The fields of this dict are:

* `N_GRIDS`: Total positions of the grid (default = 5).
* `N_CLASS`: Total of loads on the dataset (default = 26).
* `SIGNAL_BASE_LENGTH`: Total of mapped samples on each signal cut (default = 12800, 50 electrical network cycles).
* `AUGMENTATION_RATIO`: Augmentation ratio. In case this value is greater than 1, the program will generate more cuts for the same event applying a different offset on the window (default = 1). (Deprecated)
* `MARGIN_RATIO`: Size of the unmapped margins defined by a portion of the signal. (default = 0.15).
* `USE_NO_LOAD`: Flag to indicated if the appliance "NO LOAD" must be considered. (Deprecated)
* `DATASET_PATH`: Path to the .hdf5 file containing the samples.
* `TRAIN_SIZE`: Ratio of the examples used for training (default = 0.8). (Only used if the kfold is not performed)
* `FOLDER_PATH`: Path to the folder where the model shall be stored.
* `FOLDER_DATA_PATH`: Path to the *.p files with the already processed data. Usually it's the same that FOLDER_PATH.
* `N_EPOCHS_TRAINING`: Total of epochs for training. (default = 250)
* `INITIAL_EPOCH`: Initial epoch to continue a training, only useful if a training will be continued. (default = 0).
* `TOTAL_MAX_EPOCHS`: Max of training epochs.
* `SNRdB`: Noise level on dB.

## How to run

To train, just install all dependencies, configure the dict structure on the file `src/main.py` and run it as follows:

```
$ cd src
$ python3 main.py
```

Also, on the folder `notebooks` there are a few notebooks for evaluation of the models and some visualization tests.