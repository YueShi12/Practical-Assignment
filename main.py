import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
img_size = 28
def datareadin() :
    mnist_data = pd.read_csv('mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    plt.imshow(digits[2].reshape(img_size, img_size))
    plt.show()
    return mnist_data, labels, digits

def inkfeature(digits, labels) :
    ink = np.array([sum(row) for row in digits])
    ink_scale = scale(ink).reshape(-1, 1)
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



mnist_data, labels, digits = datareadin()

ink, ink_mean,ink_std,ink_scale = inkfeature(digits,labels)
regularized_multinomial_logit_model(digits, labels, ink, ink_scale)

