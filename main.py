import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import random as rd

img_size = 28
training_set_size = 5000
MLM = "Regularized Multinomial Logit Model"
SVM = "Support Vector Machine"
NN = "Feed-forward Neural Network"

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
    clf = LogisticRegression(penalty = "l2", C = 0.01, solver = "lbfgs", max_iter = 1000, multi_class = "multinomial")
    mlm_model = clf.fit(training_digits, training_labels)
    mlm_pred = mlm_model.predict(test_digits)

    # support vector machine
    clf = SVC(C = 100, kernel = "poly", degree = 3)
    svm = clf.fit(training_digits, training_labels)
    svm_pred = svm.predict(test_digits)

    # feed-forward neural network
    clf = MLPClassifier(hidden_layer_sizes = (480, 480), learning_rate_init = 0.00095, max_iter = 200, solver = "adam", activation = "relu")
    nn = clf.fit(training_digits, training_labels)
    nn_pred = nn.predict(test_digits)

    # get digit specific recall (if weighted, adds up to accuracy)
    accuracies_mlm = get_digit_recall(mlm_pred, test_labels)
    accuracies_svm = get_digit_recall(svm_pred, test_labels)
    accuracies_nn = get_digit_recall(nn_pred, test_labels)

    # show digit specific metric
    show_digit_metric(accuracies_mlm, accuracies_svm, accuracies_nn)

    # get overall accuracy, weighted precision, and f1 score (weighted recall is useless)
    accuracy_mlm, precision_mlm, f1_mlm = get_basic_metrics(mlm_pred, test_labels)
    accuracy_svm, precision_svm, f1_svm = get_basic_metrics(svm_pred, test_labels)
    accuracy_nn, precision_nn, f1_nn = get_basic_metrics(nn_pred, test_labels)

    # show graph of basic metrics
    mlm_scores = [accuracy_mlm, precision_mlm, f1_mlm]
    svm_scores = [accuracy_svm, precision_svm, f1_svm]
    nn_scores = [accuracy_nn, precision_nn, f1_nn]
    show_basic_metrics(mlm_scores, svm_scores, nn_scores)
    
    # create confusion matrices
    df_mlm_single = create_confusion_matrix_single(mlm_pred, test_labels)
    show_confusion_matrix_single(MLM, df_mlm_single)
    df_svm_single = create_confusion_matrix_single(svm_pred, test_labels)
    show_confusion_matrix_single(SVM, df_svm_single)
    df_nn_single = create_confusion_matrix_single(nn_pred, test_labels)
    show_confusion_matrix_single(NN, df_nn_single)
    # reg multinomial logit model vs support vector machine
    df_mlm_svm_dual = create_confusion_matrix_dual(mlm_pred, svm_pred)
    show_confusion_matrix_dual(MLM, SVM, df_mlm_svm_dual)
    # reg multinomial logit model vs neural network
    df_mlm_nn_dual = create_confusion_matrix_dual(mlm_pred, nn_pred)
    show_confusion_matrix_dual(MLM, NN, df_mlm_nn_dual)
    # support vector machine vs neural network
    df_svm_nn_dual = create_confusion_matrix_dual(svm_pred, nn_pred)
    show_confusion_matrix_dual(SVM, NN, df_svm_nn_dual)
    # reg multinomial logit model vs support vector machine
    df_mlm_svm_comp = create_confusion_matrix_compare(mlm_pred, svm_pred, test_labels)
    show_confusion_matrix_comp(MLM, SVM, df_mlm_svm_comp)
    # reg multinomial logit model vs neural network
    df_mlm_nn_comp = create_confusion_matrix_compare(mlm_pred, nn_pred, test_labels)
    show_confusion_matrix_comp(MLM, NN, df_mlm_nn_comp)
    # support vector machine vs neural network
    df_svm_nn_comp = create_confusion_matrix_compare(svm_pred, nn_pred, test_labels)
    show_confusion_matrix_comp(SVM, NN, df_svm_nn_comp)

    

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

def get_basic_metrics(pred, actual):
    return (accuracy_score(actual, pred), 
    precision_score(actual, pred, average = 'weighted'),
    f1_score(actual, pred, average = 'weighted'))

def get_digit_recall(pred, actual):
    return recall_score(actual, pred, average = None)

def show_digit_metric(mlm_accuracies, svm_accuracies, nn_accuracies):
    print(f"mlm scores are: {mlm_accuracies}")
    print(f"svm_scores are: {svm_accuracies}")
    print(f"nn_scores are: {nn_accuracies}")
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    w = 0.2
    bar1 = np.arange(len(labels))
    bar2 = [i+w for i in bar1]
    bar3 = [i+w for i in bar2]

    mlm_bars = plt.bar(bar1, mlm_accuracies, w, label="Regularized Multinomial Logit Model")
    for bar in mlm_bars:
        bar.set_color("#308089")
    svm_bars = plt.bar(bar2, svm_accuracies, w, label="Support Vector Machine")
    for bar in svm_bars:
        bar.set_color("#D2D534")
    nn_bars = plt.bar(bar3, nn_accuracies, w, label="Feed-forward Neural Network")
    for bar in nn_bars:
        bar.set_color("#A82BD7")
    plt.xlabel("Digit")
    plt.ylabel("Recall")
    plt.xticks(bar1 + 1 * w, labels)
    plt.ylim(0.8, 1)
    plt.legend()
    plt.title("Recall for each digit")
    plt.show()

def show_basic_metrics(mlm_scores, svm_scores, nn_scores):
    print(f"mlm scores are: {mlm_scores}")
    print(f"svm_scores are: {svm_scores}")
    print(f"nn_scores are: {nn_scores}")
    labels = ['Accuracy', 'Precision', 'F1 Score']
    w = 0.2
    bar1 = np.arange(len(labels))
    bar2 = [i+w for i in bar1]
    bar3 = [i+w for i in bar2]

    mlm_bars = plt.bar(bar1, mlm_scores, w, label="Regularized Multinomial Logit Model")
    for bar in mlm_bars:
        bar.set_color("#308089")
    svm_bars = plt.bar(bar2, svm_scores, w, label="Support Vector Machine")
    for bar in svm_bars:
        bar.set_color("#D2D534")
    nn_bars = plt.bar(bar3, nn_scores, w, label="Feed-forward Neural Network")
    for bar in nn_bars:
        bar.set_color("#A82BD7")
    plt.ylabel("Score")
    plt.xticks(bar1 + 1 * w, labels)
    plt.ylim(0.85, 1)
    plt.legend()
    plt.title("Accuracy, Precision, and F1 Score")
    plt.show()

def create_confusion_matrix_single(pred, actual):
    df = pd.DataFrame(data = confusion_matrix(y_true = actual, y_pred = pred))
    return df

def create_confusion_matrix_dual(pred1, pred2):
    df = pd.DataFrame(data = confusion_matrix(y_true = pred1, y_pred = pred2))
    return df

def create_confusion_matrix_compare(pred1, pred2, actual):
    zero_data = np.zeros(shape = (2,2))
    df = pd.DataFrame(data = zero_data)
    for index in range(len(pred1)):
        if pred1[index] == actual[index]:
            row = 1
        else:
            row = 0
        if pred2[index] == actual[index]:
            column = 1
        else:
            column = 0
        df.at[row, column] = df.at[row, column] + 1
    return df

def show_confusion_matrix_single(clf: str, df):
    sn.heatmap(df, annot=True, annot_kws={"size": 10}, fmt='g')
    plt.title(clf)
    plt.show()

def show_confusion_matrix_dual(clf1: str, clf2: str, df):
    sn.heatmap(df, annot=True, annot_kws={"size": 10}, fmt='g')
    plt.xlabel(clf2)
    plt.ylabel(clf1)
    plt.show()

def show_confusion_matrix_comp(clf1: str, clf2: str, df):
    sn.heatmap(
        df, 
        annot=True, 
        annot_kws={"size": 10}, 
        fmt='g', 
        xticklabels = ["Wrong", "Right"], 
        yticklabels = ["Wrong", "Right"],
        cmap = ListedColormap(['white'])
        )
    plt.xlabel(clf2)
    plt.ylabel(clf1)
    plt.show()

mnist_data, labels, digits = datareadin()
compare_three_methods(digits, labels)
