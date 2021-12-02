import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
import random as rd

img_size = 28

def datareadin() :
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    # print(f"number: {labels[0]}")
    # plt.imshow(digits[0].reshape(img_size, img_size))
    # plt.show()
    return mnist_data, labels, digits

def inkfeature(digits, labels) :
    ink = np.array([sum(row) for row in digits])
    ink_scale = scale(ink)
    print(f"ink_scale shape is {ink_scale.shape}")
    ink_scale = ink_scale.reshape(-1, 1)
    print(f"ink_scale shape after reshape is {ink_scale.shape}")
    ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
    ink_std = [np.std(ink[labels == i]) for i in range(10)]
    return ink, ink_mean,ink_std,ink_scale

def regularized_multinomial_logit_model(digits, labels, ink, ink_scale):
    # may use GridSearchCV(), RandomizedSearchCV() for param select? accu=0.2269 now
    lr_model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=4000)
    lr_model.fit(ink_scale, labels)
    pred = lr_model.predict(ink_scale)
    accuracy = metrics.accuracy_score(labels, pred)
    print(accuracy)

def show_5_random_drawings_of_digit(digit, digits, labels):
    possible_indices = []
    for i in range (len(labels)):
        if labels[i] == digit:
            possible_indices.append(i)
    random_drawings = rd.sample(possible_indices, 5)
    for index in random_drawings:
        plt.imshow(digits[index].reshape(img_size, img_size))
        plt.show()



mnist_data, labels, digits = datareadin()

ink, ink_mean, ink_std, ink_scale = inkfeature(digits,labels)
regularized_multinomial_logit_model(digits, labels, ink, ink_scale)

