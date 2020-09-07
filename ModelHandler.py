import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv1D, LeakyReLU, MaxPooling1D, BatchNormalization, Dropout, Dense, Reshape, Flatten, Softmax
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

class ModelHandler:
    def __init__(self, configs):
        try:
            self.m_ngrids = configs["N_GRIDS"]
            self.m_nclass = configs["N_CLASS"]
            self.m_signalBaseLength = configs["SIGNAL_BASE_LENGTH"]
            self.m_marginRatio = configs["MARGIN_RATIO"]
            self.m_gridLength = int(self.m_signalBaseLength / self.m_ngrids)
            self.configs = configs

            if "USE_NO_LOAD" in self.configs and self.configs["USE_NO_LOAD"] == True:
                self.m_nclass += 1
        except:
            print("Erro no dicionário de configurações")
            exit(-1)

    def buildModel(self):
        input = Input(shape=(self.m_signalBaseLength + 2 * int(self.m_signalBaseLength * self.m_marginRatio), 1))
        x = Conv1D(filters=60, kernel_size=9)(input)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Conv1D(filters=40, kernel_size=9)(x)
        x = LeakyReLU(alpha = 0.1)(x)
        x = MaxPooling1D(pool_size=4)(x)
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.25)(x)
        x = Flatten()(x)

        detection_output = Dense(200)(x)
        detection_output = LeakyReLU(alpha = 0.1)(detection_output)
        detection_output = Dropout(0.25)(detection_output)
        detection_output = Dense(20)(detection_output)
        detection_output = LeakyReLU(alpha = 0.1)(detection_output)
        detection_output = Dense(1 * self.m_ngrids, activation='sigmoid')(detection_output)
        detection_output = Reshape((self.m_ngrids, 1), name="detection")(detection_output)

        classification_output = Dense(300)(x)
        classification_output = LeakyReLU(alpha = 0.1)(classification_output)
        classification_output = Dropout(0.25)(classification_output)
        classification_output = Dense(300)(classification_output)
        classification_output = LeakyReLU(alpha=0.1)(classification_output)
        classification_output = Dense((self.m_nclass) * self.m_ngrids, activation = 'sigmoid')(classification_output)
        classification_output = Reshape((self.m_ngrids, (self.m_nclass)), name = "classification")(classification_output)
        #classification_output = Softmax(axis=2, name="classification")(classification_output)

        type_output = Dense(10)(x)
        type_output = LeakyReLU(alpha = 0.1)(type_output)
        type_output = Dense(3 * self.m_ngrids)(type_output)
        type_output = Reshape((self.m_ngrids, 3))(type_output)
        type_output = Softmax(axis=2, name="type")(type_output)
        
        model = Model(inputs = input, outputs=[detection_output, type_output, classification_output])

        model.compile(optimizer='adam', loss = [sumSquaredError, "categorical_crossentropy", "binary_crossentropy"], metrics=['accuracy'])

        return model

    def loadModel(self, path):
        return load_model(path, custom_objects={'sumSquaredError': self.sumSquaredError, 'KerasFocalLoss': self.KerasFocalLoss})

    
    def plotModel(self, model, pathToDirectory):
        if pathToDirectory[-1] != "/":
            pathToDirectory += "/"

        plot_model(model, to_file = pathToDirectory + 'model_plot.png', show_shapes=True, show_layer_names=True)
    
    @staticmethod
    def KerasFocalLoss(target, input):
        gamma = 2.
        input = tf.cast(input, tf.float32)
        
        max_val = K.clip(-1 * input, 0, 1)
        loss = input - input * target + max_val + K.log(K.exp(-1 * max_val) + K.exp(-1 * input - max_val))
        invprobs = tf.math.log_sigmoid(-1 * input * (target * 2.0 - 1.0))
        loss = K.exp(invprobs * gamma) * loss
        
        return K.mean(K.sum(loss, axis=1))

    @staticmethod
    def sumSquaredError(y_true, y_pred):
        return K.sum(K.square(y_true - y_pred), axis=-1)