# How to run on Jetson TX1

Tested with the following package versions:
 
- L4T 28.2.0 [ JetPack 3.2 ]
- Ubuntu 16.04.6 LTS
- Kernel Version: 4.4.38-tegra
- CUDA 9.0.252
- CUDA Architecture: 5.3
- CUDNN: 7.0.5.15

## Dependencies

- Tensorflow:

    The latest tensorflow version available for this jetpack is v1.4, which doesn't offer some functionalities used on the implementation provided here. To install a different version, it's necessary to compile it. Luckily, there is a [repository](https://github.com/peterlee0127/tensorflow-nvJetson) that provides compiled versions using wheel files. Therefore, to install the tensorflow at version `1.8` using these compiled files, just run: 


    ```
    $ wget https://github.com/peterlee0127/tensorflow-nvJetson/releases/download/binary/tensorflow-1.8.0-cp27-cp27mu-linux_aarch64.whl
    $ sudo pip install tensorflow-1.8.0-cp27-cp27mu-linux_aarch64.whl
    ``` 

- Keras:

    Keras is a framework that acts as a front-end for tensorflow. To install a version compatible with the tensorflow version previously install, just run the following command:

    ```
    $ pip install keras==2.2.4
    ```

    This will install the version `2.2.4` of Keras.

- NumPy:

    To maintain the compatibility between libraries, it might be necessary to update the numpy version. To install the version `1.14.5` (the one that was installed for the tests), just run:

    ```
    $ pip install numpy==1.14.5
    ```

## Running the test

1. Cloning the repository

    The first step is to clone this repository and download the trained weights. If you already did this, just skip to the step 3. If you didn't, to clone this repository just run this:

    ```
    $ git clone https://github.com/LucasNolasco/DeepDFML-NILM.git
    cd DeepDFML-NILM
    ```

2. Downloading the weights

    The weights might be downloaded directly from the following [url](https://drive.google.com/file/d/18lcnLgRms-Sb_AovSSFTFTPKWgKu5V8Y/view) on a browser. Another easy option is using the `gdown` program. First of all, it's necessary to install it:

    ```
    $ pip3 install gdown
    ```

    After the instalation, just run the following command to download the weights:

    ```
    $ gdown https://drive.google.com/u/2/uc\?id\=18lcnLgRms-Sb_AovSSFTFTPKWgKu5V8Y\&export\=download
    ``` 

    This will create a new file named `TrainedWeights.zip` on the current directory. The final step is to unzip this new file:

    ```
    $ unzip TrainedWeights.zip
    ```

    After all this, the final result should be a folder called `TrainedWeights` inside the repository folder (`DeepDFML-NILM`).

3. Running the test

    To maximize the board performance, these tests assume that the [jetson_clocks](https://developer.ridgerun.com/wiki/index.php?title=Xavier/JetPack_4.1/Performance_Tuning/Maximizing_Performance#Jetson_Clocks) script is running.

    Inside the repository folder (`DeepDFML-NILM`), enter the script folder and run the test script. All the code on this repository is designed for Python3, but the tensorflow is compiled for Python2, so this code must be executed using Python2 on Jetson. To run it:

    ```
    $ cd script
    $ python test_jetson.py
    ```

    In order to check the board resource consumption, the package [jetson-stats](https://github.com/rbonghi/jetson_stats) was employed. Install instructions and usage examples are found on [its GitHub repository](https://github.com/rbonghi/jetson_stats).