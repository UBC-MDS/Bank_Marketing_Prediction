import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    average_precision_score, 
    auc
)

def plot_confusion_mat(tar, pred, model_name):
    """
    Create plot for confusion matrix for
    binary classification results from
    target and predicted labels.

    Parameters
    ----------
    tar : ndarray
        numpy array of target label
        
    pred : ndarray
        numpy array of predicted label

    model_name : str
        Name of model for plot title

    Returns
    ----------
        Plot of the confusion matrix along
        with the score matrics
    """
    conf_mat = confusion_matrix(tar, pred)

    flags = ['True Neg','False Pos','False Neg','True Pos']
    cnt = ['{0:0.0f}'.format(value) for value in conf_mat.flatten()]
    pct = ['{0:.2%}'.format(value) for value in conf_mat.flatten()/np.sum(conf_mat)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(flags, cnt, pct)]

    labels = np.asarray(labels).reshape(2,2)
    categories = ['Rejected subscription', 'Accepted subscription']

    accuracy  = np.trace(conf_mat) / float(np.sum(conf_mat))
    precision = conf_mat[1,1] / sum(conf_mat[:,1])
    recall    = conf_mat[1,1] / sum(conf_mat[1,:])
    f1  = 2*precision*recall / (precision + recall)

    fig = sns.heatmap(conf_mat, 
                annot=labels, 
                fmt = '', 
                cmap='Blues',
                xticklabels=categories, 
                yticklabels=categories, 
                cbar=False)

    fig.set_ylabel('True label')
    fig.set_xlabel(f"Predicted label \n\n Accuracy={accuracy:0.3f} \n Precision={precision:0.3f} \n Recall={recall:0.3f} \n F1 Score={f1:0.3f}")
    fig.set_title(f'Confusion Matrix of {model_name}')
    return fig
