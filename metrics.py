import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def metrics(pred, target, n_aug):
    '''
    Function that evaluates plcc, srcc and accuracy metrics. Pred and target are one-dimensional numpy arrays
    that consist of parts of equal sizes such as in each of them there are images with only one type augmentation.
    For example: let there be 3 augmentation and size of array 12. Then it will be like that:
    [A1, B1, C1, D1, A2, B2, C2, D2, A3, B3, C3, D3] where first letter in element is image indicator and
    second one is augmentation number.
    :param pred: predictions list. Must be numpy array that consist only on numbers: one prediction - one number
    :param target: target list. Must be numpy array that consist only on numbers: one target example - one number
    :param n_aug: number of augmentation used in training.
    :return:
    '''
    assert len(pred) == len(target), 'Number of predictions must be equal to the number of target data!'
    assert len(pred) % n_aug == 0, \
        'Output must consists of parts of equal sizes such as in each of them there are images with only one type augmentation!'

    N = len(pred)//n_aug

    plcc_list = []
    srcc_list = []
    acc_list = []
    #For each augmentet part independently compute all metrics
    for i in range(n_aug):
        pred_part = pred[i*N: (i+1)*N]
        target_part = target[i*N: (i+1)*N]

        #accuracy is a percent (divided by 100) of right answers
        acc_list.append(np.sum((pred_part >= 5)*1 == 5)/len(pred_part))
        #plcc is a pearson correlation coefficient
        plcc_list.append(pearsonr(pred_part, target_part)[0])
        #srcc is a spearman's rank correlation coefficient
        xranks = pd.Series(pred_part).rank()
        yranks = pd.Series(target_part).rank()
        srcc_list.append(pearsonr(xranks, yranks)[0])

    #After independent computing let's average all results to get one number for one metric
    plcc = np.mean(plcc_list)
    srcc = np.mean(srcc_list)
    acc = np.mean(acc_list)
    return plcc, srcc, acc

#Just example for validation if function is working. You can delete it if you want
n_aug = 8
n_examples = 100
x = np.random.randn(n_aug*n_examples)
x = (x - x.min())/(x.max()-x.min())
y = np.random.randn(n_aug*n_examples)
y = (y >= 0)*1

plcc, srcc, acc = metrics(x, y, n_aug)

print(acc)
print(plcc)
print(srcc)
