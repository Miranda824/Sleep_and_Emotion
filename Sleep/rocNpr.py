# 引入必要的库
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from scipy import interp


def pr_plot(y_score, y_stage):
    rec=dict()
    pre=dict()
    pr_auc=dict()
    y_stage = label_binarize(y_stage, classes=[0, 1, 2, 3, 4])


    pre["micro"],rec["micro"], _ = precision_recall_curve(y_stage.ravel(), y_score.ravel())
    pr_auc["micro"] = auc(rec["micro"],pre["micro"])


    lw = 2
    plt.figure()
    plt.plot(rec["micro"], pre["micro"],
             label='micro-average pr curve (area = {0:0.2f})'
                   ''.format(pr_auc["micro"]),
             color='blue', linestyle='-', linewidth=2)

    plt.plot([0, 1], [1, 0], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precison')
    plt.title('P_R')
    plt.legend(loc="lower right")
    plt.savefig('./pr.png')
    #   plt.pause(0.05)
    plt.close()


def roc_plot(y_score, y_stage):
    # 绘制ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_stage = label_binarize(y_stage, classes=[0, 1, 2, 3, 4])
    # for i in range(5):
    #     fpr[i], tpr[i], _ = roc_curve(y_stage[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_stage.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue', linestyle='-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('micro-average roc')
    plt.legend(loc="lower right")
    plt.savefig('./roc.png')
#   plt.pause(0.05)
    plt.close()