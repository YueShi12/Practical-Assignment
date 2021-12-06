import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import alphas
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix

img_size = 28


def datareadin():
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    # plt.imshow(digits[2].reshape(img_size, img_size))
    # plt.show()
    return mnist_data, labels, digits


def inkfeature(digits, labels):
    ink = np.array([sum(row) for row in digits])

    ink_scale = scale(ink).reshape(-1, 1)
    ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
    ink_std = [np.std(ink[labels == i]) for i in range(10)]

    return ink, ink_mean, ink_std, ink_scale


def other_features(digits, labels):
    ink_four = []
    for digit in digits:
        d = digit.reshape(56, 14)
        ink_f = np.array([sum(row) for row in d])
        ink_four.append(ink_f)
    ink_four = np.array(ink_four)

    return ink_four


def regularized_multinomial_logit_model(ink_mean, labels, ink_scale, ink_four):
    lr_model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=5000)
    X_train = ink_scale + ink_four

    lr_model.fit(X_train, labels)

    pred = lr_model.predict(X_train)

    precision = metrics.precision_score(labels, pred, average=None)

    recall = metrics.recall_score(labels, pred, average=None)

    f1 = metrics.f1_score(labels, pred, average=None)

    precision_result = pd.DataFrame(precision, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    precision_result.rename(columns={0: 'precision'}, inplace=True)

    recall_results = pd.DataFrame(recall, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    recall_results.rename(columns={0: 'Recall'}, inplace=True)

    f1_results = pd.DataFrame(f1, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    f1_results.rename(columns={0: 'f1'}, inplace=True)
    print(metrics.classification_report(labels, pred))


mnist_data, labels, digits = datareadin()
ink_four = other_features(digits, labels)
ink, ink_mean, ink_std, ink_scale = inkfeature(digits, labels)
regularized_multinomial_logit_model(ink_mean, labels, ink_scale, ink_four)
