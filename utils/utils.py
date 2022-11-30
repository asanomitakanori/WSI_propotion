import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import confusion_matrix


def rotation_img(imgs):
    random_int = random.randint(0, imgs.shape[0] - 1)
    target = imgs[random_int].unsqueeze(0)
    target_source = torch.cat([target, imgs[0:random_int]])
    imgs = torch.cat([target_source, imgs[random_int+1::]])
    return imgs

def min_max_normalization(input):
    output = (input - input.min()) / (input.max() - input.min())
    return output

def semantic_segmentation(input):
    output = torch.zeros_like(input)
    max_channel = torch.max(input, dim=1)[1]
    for i in range(output.shape[1]):
        output[:, i][max_channel == i] = 1
    return output

def early_stop(max_score, dice_score, early_score, model, model_param):
    if dice_score > max_score:
            early_score = 0
            max_score = dice_score
            model_param = model.state_dict()
    else:
        early_score += 1
    return max_score, early_score, model_param


def all_confusion_score(cm):
    eps = 0.01
    labels = ['tumorbed', 'no_label', 'residual']
    cm = cm / (cm.sum(axis = 0) + eps)
    columns_labels = ["pred_" + str(l) for l in labels]
    index_labels = ["true_" + str(l) for l in labels]
    # cm = pd.DataFrame(cm, columns=columns_labels, index=index_labels)
    return cm


def confusion_score(y_true, y_pred, labels=[0, 1, 2]):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    y_t, y_p = y_true.copy(), y_pred.copy()
    cm = confusion_matrix(y_true=y_t.flatten(), y_pred=y_p.flatten(), labels=labels)
    return cm


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    # plt.xlim(-0.5, len(np.unique(y))-0.5)
    # plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax

class evalMet(object):
    def Accuracy(self, cm):
        len_label = len(cm)
        inter_all = 0
        for idx in range(len_label):
            inter_all += cm[idx][idx]
        return inter_all / np.sum(cm)

    def Precision(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / np.sum(cm[:, idx])
        return iou / len_label

    def Recall(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / np.sum(cm[idx, :])
        return iou / len_label

    def F1(self, cm):
        precision = self.Precision(cm)
        recall = self.Recall(cm)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def mIoU(self, cm):
        len_label = len(cm)
        iou = 0
        for idx in range(len_label):
            inter = cm[idx][idx]
            iou += inter / (np.sum(cm[idx, :]) + np.sum(cm[:, idx]) - inter)
        return iou / len_label

def eval_metrics(cm):
    Met = evalMet()
    met_val = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mIoU': 0}
    met_val['accuracy'] = Met.Accuracy(cm)
    met_val['precision'] = Met.Precision(cm)
    met_val['recall'] = Met.Recall(cm)
    met_val['f1'] = Met.F1(cm)
    met_val['mIoU'] = Met.mIoU(cm)
    return met_val