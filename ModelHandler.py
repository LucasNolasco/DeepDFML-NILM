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

    def buildModel(self, type_weights=None):
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

        if type_weights is not None:
            model.compile(optimizer='adam', loss = [ModelHandler.sumSquaredError, ModelHandler.weighted_categorical_crossentropy(type_weights), "binary_crossentropy"], metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])
        else:
            model.compile(optimizer='adam', loss = [ModelHandler.sumSquaredError, "categorical_crossentropy", "binary_crossentropy"], metrics=[['mean_squared_error'], ['categorical_accuracy'], ['binary_accuracy']])

        return model

    @staticmethod
    def loadModel(path, type_weights={}):
        return load_model(path, custom_objects={'sumSquaredError': ModelHandler.sumSquaredError,\
                                                'KerasFocalLoss': ModelHandler.KerasFocalLoss,\
                                                'loss': ModelHandler.weighted_categorical_crossentropy(type_weights)})

    
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
        #event_exists = tf.math.ceil(y_true)

        #return K.sum(K.square(y_true - y_pred) * event_exists, axis=-1)
        return K.sum(K.square(y_true - y_pred), axis=-1)

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
        
        #weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            import numpy as np
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc

            weights_mask = []
            for true_class in K.reshape(K.argmax(y_true, axis=2), (y_true.shape[0] * y_true.shape[1],)):
                weights_mask.append(weights[K.get_value(true_class)])
                weights_mask.append(weights[K.get_value(true_class)])
                weights_mask.append(weights[K.get_value(true_class)])
            
            weights_mask = np.array(weights_mask)
            weights_mask = np.reshape(weights_mask, y_true.shape)

            weights_mask = K.variable(weights_mask)

            loss = y_true * K.log(y_pred) * weights_mask
            loss = -K.sum(loss, -1)
            return loss
    
        return loss

    @staticmethod
    def w_categorical_crossentropy(weights):
        def loss(y_true, y_pred):
            from itertools import product
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1, keepdims=True)
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        
            return K.categorical_crossentropy(y_pred, y_true) * final_mask
        
        return loss