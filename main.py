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
    return mnist_data, labels, digits 

# we will compare the regularized (LASSO) multinomial logit model, support vector machines,
# and feed-forward neural networks
def compare_three_methods(digits, labels):
    # get training set
    training_digits, training_labels, test_digits, test_labels = draw_random_training_sample(digits, labels, size = training_set_size)
    # preprocessing
    training_digits, test_digits = preprocess(training_digits, test_digits)

    # multinomial logit model
    clf = get_grid_multinomial_logit_model()
    mlm_model = clf.fit(training_digits, training_labels)
    # print(f"The best parameters for the multinomial logit model are {mlm_model.best_params_}")
    mlm_pred = mlm_model.predict(test_digits)
    evaluate_results(model = "Multinomial Logit Model", pred = mlm_pred, actual = test_labels)

    # # support vector machine
    # clf = get_grid_svm()
    # svm = clf.fit(training_digits, training_labels)
    # print(f"The best parameters for the svm are {svm.best_params_}")
    # svm_pred = svm.predict(test_digits)
    # evaluate_results(model = "SVM", pred = svm_pred, actual = test_labels)

    # # feed-forward neural network
    # clf = get_grid_nn()
    # nn = clf.fit(training_digits, training_labels)
    # print(f"The best parameters for the nn are {nn.best_params_}")
    # nn_pred = nn.predict(test_digits)
    # evaluate_results(model = "Neural Network", pred = nn_pred, actual = test_labels)

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

def preprocess(training_digits, test_digits):
    # remove unnecessary features
    unnecessary_features_indices = get_useless_feature_indices(training_digits)
    training_digits = np.delete(training_digits, unnecessary_features_indices, axis=1)
    test_digits = np.delete(test_digits, unnecessary_features_indices, axis=1)
    # scale digits
    training_digits = scale(training_digits)
    test_digits = scale(test_digits)
    return training_digits, test_digits

def get_useless_feature_indices(digits):
    df = pd.DataFrame(digits)
    useless_features = df.columns[(df == 0).all()]
    return useless_features.tolist()

def get_grid_multinomial_logit_model():
    param_grid = [
        {
            'penalty': ['l2'],
            'C': [.001 ,.01, .1, 1, 10, 100, 1000],
            'solver': ['lbfgs'],
            'max_iter': [1000], # as long as theres no errors it's fine
            'multi_class': ['multinomial'],
        }
    ]
    # return GridSearchCV(estimator=LogisticRegression(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose = 3, scoring = 'accuracy')
    return LogisticRegression(penalty = "l2", C = 0.01, solver = "lbfgs", max_iter = 1000, multi_class = "multinomial")

def get_grid_svm():
    param_grid = [
        {
            'C': [.001 ,.01, .1, 1, 10, 100, 1000], 
            # 'gamma': [1,0.1,0.01,0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    ]
    return GridSearchCV(estimator = SVC(), param_grid = param_grid, n_jobs = -1, cv = 10, verbose =  3, scoring = 'accuracy')

def get_grid_nn():
    param_grid = [
        {
            "hidden_layer_sizes": [(400,), (450, 450), (480,), (500,)],
            "solver": ['adam', 'sgd', 'lbfgs'],
            "activation" :['identity', 'logistic', 'tanh', 'relu'],
            "max_iter": [200],
            "verbose": [True]
            # The best parameters for the nn are {'hidden_layer_sizes': (480,), 'max_iter': 200, 'solver': 'adam', 'activation': 'relu','verbose': True}
            # The accuracy of the Neural Network is 0.9384594594594594
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
compare_three_methods(digits, labels)

