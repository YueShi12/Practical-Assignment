import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import random as rd

img_size = 28
training_set_size = 5000

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

# def show_5_random_drawings_of_digit(digit, digits, labels):
#     possible_indices = []
#     for i in range (len(labels)):
#         if labels[i] == digit:
#             possible_indices.append(i)
#     random_drawings = rd.sample(possible_indices, 5)
#     for index in random_drawings:
#         plt.imshow(digits[index].reshape(img_size, img_size))
#         plt.show()



def draw_random_training_sample(digits, labels, size):
    training_indices = np.random.choice(np.arange(0, digits.shape[0]), size = size, replace = False)
    training_digits, training_labels, test_digits, test_labels = [], [], [], []
    for index in range(digits.shape[0]):
        if index in training_indices:
            training_digits.append(digits[index])
            training_labels.append(labels[index])
        else:
            test_digits.append(digits[index])
            test_labels.append(labels[index])
    return training_digits, training_labels, test_digits, test_labels

def draw_useless_pixels(digits, labels):
    df = pd.DataFrame(digits)
    useless_features = df.columns[(df == 0).all()]
    print(len(useless_features.tolist()))
    illustration = list(map(lambda pixel: 1 if pixel in useless_features else 0, range(784)))
    cmap = matplotlib.colors.ListedColormap(['green', 'red'])
    plt.imshow(np.array(illustration).reshape(img_size, img_size), cmap = cmap)
    plt.show()

def analyze_class_distribution(labels):
    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for label in labels:
        if label == 0:
            zeros += 1
        elif label == 1:
            ones += 1
        elif label == 2:
            twos += 1
        elif label == 3:
            threes += 1
        elif label == 4:
            fours += 1
        elif label == 5:
            fives += 1
        elif label == 6:
            sixes += 1
        elif label == 7:
            sevens += 1
        elif label == 8:
            eights += 1
        elif label == 9:
            nines += 1
        else:
            print("Something went wrong")
    df = pd.DataFrame(data = np.array([zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]).reshape(1, -1),
    columns = ['zeros', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'sevens', 'eights', 'nines'])
    print(df.describe())




mnist_data, labels, digits = datareadin()
ink, ink_mean,ink_std,ink_scale = inkfeature(digits, labels)
print(f"Ink: {ink}")
print(f"Ink mean: {ink_mean}")
print(f"Ink std: {ink_std}")



