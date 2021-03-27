import numpy as np

class MultiLabelMetrics:
    @staticmethod
    def HammingScore(model, x, y):
        # Hamming Score: https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
        accuracy = 0
        for xi, yclass in zip(x, y["classification"]):
            pred = model.predict(np.expand_dims(xi, axis=0))
            prediction = np.max(pred[2][0],axis=0) > 0.5
            groundTruth = np.max(yclass,axis=0) > 0.5

            union = sum(np.logical_or(prediction, groundTruth))
            intersection = sum(np.logical_and(prediction, groundTruth))

            if union == 0 and intersection == 0:
                accuracy += 1
            else:
                accuracy += intersection / union
        
        accuracy /= x.shape[0]
        print(accuracy)

    # TODO: Ajustar isso. A métrica é por classes, o cálculo feito aqui é para toda a base.
    @staticmethod
    def F1EB(model, x, y):
        # F1-eb: Paper CNN Multi-Label
        tp, len_yi, len_gt_yi = 0, 0, 0
        for xi, yclass in zip(x, y["detection"], y["type"], y["classification"]):
            pred = model.predict(np.expand_dims(xi, axis=0))
            prediction = np.max(pred[2][0],axis=0) > 0.5
            groundTruth = np.max(yclass,axis=0) > 0.5

            tp += sum(np.logical_and(prediction, groundTruth))
            len_yi += sum(prediction)
            len_gt_yi += sum(groundTruth)
        
        accuracy = 2 * tp / (len_yi + len_gt_yi)
        print(accuracy)

    @staticmethod
    def F1Macro(model, x, y):
        # F1-macro: Paper CNN Multi-Label
        throw_warning = False # Alerta para indicar que há casos sem nenhuma classe => tp = 0 e fp + fn = 0
        accuracy = 0
        for xi, yclass in zip(x, y["classification"]):
            pred = model.predict(np.expand_dims(xi, axis=0))
            prediction = np.max(pred[2][0],axis=0) > 0.5
            groundTruth = np.max(yclass,axis=0) > 0.5

            tp = sum(np.logical_and(prediction, groundTruth))
            fp_fn = sum(np.logical_xor(prediction, groundTruth))

            if 2 * tp + fp_fn != 0:
                accuracy += 2 * tp / (2 * tp + fp_fn)
            else:
                throw_warning = True
                accuracy += 0
                
        accuracy /= x.shape[0]
        print(accuracy)

        if throw_warning:
            print("Aviso: Há amostras que não possuem nenhuma classe, pode influenciar a resposta da métrica")