import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix
from scipy import interp
from itertools import cycle

# draw confusion matrix
import seaborn as sn
import pandas as pd

import dataloader.VSLdataset as VSLdataset
import os

# for marco-F1 and micro-F1
from sklearn.metrics import f1_score

class_name_to_id_ = {
'OOO':0,
'BOO':1,
'OLO':2,
'OOR':3,
'BLO':4,
'BOR':5,
'OLR':6,
'BLR':7
}

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def draw_roc_bin(y_label, y_predicted):
    n_classes = 8
    class_list = list(class_name_to_id_)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_label_bin = label_binarize(y_label, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label_bin[:, i], y_predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label_bin.ravel(), y_predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
    # === multi-label === 
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'gold', 'olivedrab', 'maroon', 'forestgreen', 'royalblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, alpha = 0.3,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(class_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: multi-class')
    plt.legend(loc="lower right")
    save_file = os.path.join('../output', 'roc.png')
    plt.savefig(save_file)
    # === multi-label === 
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    save_file = os.path.join('../output', 'roc.png')
    plt.savefig(save_file)
    '''
# https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def draw_confusion_matrix(y_label, y_predicted_flatten):
    class_list = list(VSLdataset.class_name_to_id_)
    cm = confusion_matrix(y_label, y_predicted_flatten, normalize='true')
    cm = np.around(cm, decimals=2)
    df_cm = pd.DataFrame(cm,  index=[ class_list[i] for i in range(8) ], columns=[ class_list[i] for i in range(8)] )
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, linewidths=1, annot=True, fmt='g', cmap="YlGnBu") # font size
    save_file = os.path.join('../output', 'confusion_matrix.png')
    plt.savefig(save_file)

    
# calculate Macro F1 and Micro F1
def calculate_f1(y_label, y_predicted_flatten):
    macro_f1 = f1_score(y_label, y_predicted_flatten, average='macro')
    micro_f1 = f1_score(y_label, y_predicted_flatten, average='micro')
    print("macro_f1, micro_f1")
    print(macro_f1, micro_f1)


    
    
