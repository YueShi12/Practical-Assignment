import pandas as pd
import matplotlib.pyplot as plt
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

# def inkfeature(digits, labels) :
#     ink = np.array([sum(row) for row in digits])
#     ink_scale = scale(ink)
#     print(f"ink_scale shape is {ink_scale.shape}")
#     ink_scale = ink_scale.reshape(-1, 1)
#     print(f"ink_scale shape after reshape is {ink_scale.shape}")
#     ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
#     ink_std = [np.std(ink[labels == i]) for i in range(10)]
#     return ink, ink_mean,ink_std,ink_scale

# def regularized_multinomial_logit_model(digits, labels, ink, ink_scale):
#     # may use GridSearchCV(), RandomizedSearchCV() for param select? accu=0.2269 now
#     lr_model = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=4000)
#     lr_model.fit(ink_scale, labels)
#     pred = lr_model.predict(ink_scale)
#     accuracy = metrics.accuracy_score(labels, pred)
#     print(accuracy)

# def show_5_random_drawings_of_digit(digit, digits, labels):
#     possible_indices = []
#     for i in range (len(labels)):
#         if labels[i] == digit:
#             possible_indices.append(i)
#     random_drawings = rd.sample(possible_indices, 5)
#     for index in random_drawings:
#         plt.imshow(digits[index].reshape(img_size, img_size))
#         plt.show()




# we will compare the regularized (LASSO) multinomial logit model, support vector machines,
# and feed-forward neural networks
def compare_three_methods(digits, labels):
    # get training set
    training_digits, training_labels, test_digits, test_labels = draw_random_training_sample(digits, labels, size = training_set_size)
    # scale digits
    training_digits = scale(training_digits)
    test_digits = scale(test_digits)

    # # multinomial logit model
    # clf = get_grid_multinomial_logit_model()
    # mlm_model = clf.fit(training_digits, training_labels)
    # print(f"The best parameters for the multinomial logit model are {mlm_model.best_params_}")
    # mlm_pred = mlm_model.predict(test_digits)
    # evaluate_results(model = "Multinomial Logit Model", pred = mlm_pred, actual = test_labels)

    # # support vector machine
    # clf = get_grid_svm()
    # svm = clf.fit(training_digits, training_labels)
    # print(f"The best parameters for the svm are {svm.best_params_}")
    # svm_pred = svm.predict(test_digits)
    # evaluate_results(model = "SVM", pred = svm_pred, actual = test_labels)

    # feed-forward neural network
    clf = get_grid_nn()
    nn = clf.fit(training_digits, training_labels)
    print(f"The best parameters for the nn are {nn.best_params_}")
    nn_pred = nn.predict(test_digits)
    evaluate_results(model = "Neural Network", pred = nn_pred, actual = test_labels)

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

def get_grid_multinomial_logit_model():
    param_grid = [
        {
            'penalty': ['l2'],
            'C': list(np.arange(0.1, 1.1, 0.2)),
            'solver': ['lbfgs'],
            'max_iter': [1000], # as long as theres no errors it's fine
            'multi_class': ['multinomial'],
        }
    ]
    return GridSearchCV(estimator=LogisticRegression(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3, scoring = 'accuracy')
    # return LogisticRegression(penalty = "l2", C = 0.01, solver = "lbfgs", max_iter = 10000, multi_class = "multinomial")

def get_grid_svm():
    param_grid = [
        {
            'C': [0.1,1, 10, 100], 
            # 'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
            }
    ]
    return GridSearchCV(estimator = SVC(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose =  3, scoring = 'accuracy')

def get_grid_nn():
    param_grid = [
        {
            TODO
        }
    ]
    return GridSearchCV(estimator = MLPClassifier(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose =  3, scoring = 'accuracy')

def evaluate_results(model, pred, actual):
    right, wrong = 0, 0
    for index in range(len(pred)):
        if(pred[index] == actual[index]):
            right += 1
        else:
            wrong += 1
    print(f"The accuracy of the {model} is {right / (right + wrong)}")





mnist_data, labels, digits = datareadin()

# ink, ink_mean, ink_std, ink_scale = inkfeature(digits,labels)
# regularized_multinomial_logit_model(digits, labels, ink, ink_scale)

compare_three_methods(digits, labels)