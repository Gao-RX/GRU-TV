from cProfile import label
from unittest import result
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score

def get_confusion_matrix():
    label_path = r'/media/liu/Data/Project/Python/ICURelated/NeuralODE/GEUD-ODE-gao/result_sample_70/T-LSTM/label_best_TEST.npy'
    pred_path = r'/media/liu/Data/Project/Python/ICURelated/NeuralODE/GEUD-ODE-gao/result_sample_70/T-LSTM/pred_best_TEST.npy'
    label_all = np.load(label_path)
    pred_all = np.load(pred_path)
    for i in range(4):
        label = np.array(label_all[:, i], dtype=np.uint8).tolist()
        pred = np.array(pred_all[:, i] > 0.5, dtype=np.uint8).tolist()
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        auprc = average_precision_score(label, pred)
        print(auprc)


if __name__ == '__main__':
    get_confusion_matrix()
