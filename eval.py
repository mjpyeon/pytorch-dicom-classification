import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import argparse
from model import *
from matplotlib import pyplot as plt
import itertools
from itertools import cycle
import pandas as pd
import seaborn as sns
from scipy import interp

def str2bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        raise NotImplementedError

def get_output(model, loader, with_prob=True):
    y_pred, y_true, = [], []
    if with_prob:
        y_prob = []
    else:
        y_prob = None
    for inputs, labels in loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        if with_prob:
            probs = torch.nn.functional.softmax(outputs, dim=1)
        else:
            probs = None
        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())
        if with_prob:
            y_prob.append(probs.detach().cpu().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    if with_prob:
        y_prob = np.concatenate(y_prob)
    return y_pred, y_true, y_prob

def print_roc_curve(y_test, y_score, n_classes, figsize = (8, 6)):
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fig = plt.figure(figsize=figsize)
    """
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    """
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    return fig


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def main(args):
    # obtain outputs of the model
    model = torch.load(args.ckpt)
    if args.multilabel:
        alloc_label = multi_label
    else:
        alloc_label = binary_label
    test_dataset = EarDataset(binary_dir=args.data_dir,
                                   alloc_label = alloc_label,
                                    transforms=transforms.Compose([Rescale((256, 256)), ToTensor(), Normalize()]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    y_pred, y_true, y_score = get_output(model, test_loader)
    print(y_pred.shape, y_true.shape, y_score.shape)

    # save the confusion matrix
    with open(args.labels, 'r+') as f:
        labels = f.readlines()
        labels = [l.replace('\n', '') for l in labels]
    if not args.multilabel:
        labels = ['Normal', 'Abnormal']
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    np.set_printoptions(precision=2)
    fig = print_confusion_matrix(cnf_matrix, labels, figsize=(16,14), fontsize=10)
    fig.savefig(os.path.join(args.result_dir, args.cfmatrix_name))

    # save the roc curve
    y_onehot = np.zeros((y_true.shape[0], len(labels)), dtype=np.uint8)
    y_onehot[np.arange(y_true.shape[0]), y_true] = 1
    sums = y_onehot.sum(axis=0)
    useless_cols = []
    for i, c in enumerate(sums):
        if c == 0:
            print('useless column {}'.format(i))
            useless_cols.append(i)
    useful_cols = np.array([i for i in range(len(labels)) if i not in useless_cols])
    if args.multilabel:
        y_onehot = y_onehot[:,useful_cols]
        y_score = y_score[:,useful_cols]
    fig = print_roc_curve(y_onehot, y_score, useful_cols.shape[0], figsize=(8,6))
    fig.savefig(os.path.join(args.result_dir, args.roc_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--data_dir', type=str, help='path to the dataset')
    parser.add_argument('--result_dir', default='results', type=str, help='path in which we save the result')
    parser.add_argument('--cfmatrix_name', default='confusion_matrix', type=str, help='fname of confusion matrix')
    parser.add_argument('--roc_name', default='roc_curve', type=str, help='fname of roc curve')
    parser.add_argument('--multilabel', default=True, type=str2bool, help='if multilabel, then true, else false')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--labels', default='labels.csv', type=str, help='fname including labels')


    args = parser.parse_args()

    main(args)
