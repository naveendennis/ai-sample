import os

import numpy as np

from utils.read_dataset import read_digits_csv, custom_read_csv
dir_path = os.path.dirname(os.path.realpath(__file__))

feature_size = 64
feature_train, train_evaluate_digit_dt = read_digits_csv(dir_path + '/../dataset/digit_dataset/optdigits_raining.csv', feature_size)

from sklearn import tree
train_features = np.array([[]])
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=5)
clf.fit(feature_train, train_evaluate_digit_dt)

from utils.read_csv_file import *
test_dataset = custom_read_csv(dir_path+'/../dataset/digit_dataset/optdigits_test.csv', feature_size)
feature_test = test_dataset[0:test_dataset.shape[0], 0:feature_size]
label_test = test_dataset[0:test_dataset.shape[0], feature_size]
label_predict = clf.predict(feature_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(label_test, label_predict))
from sklearn.metrics import precision_score
print(precision_score(label_test, label_predict, average='weighted'))
from sklearn.metrics import recall_score
print(recall_score(label_test, label_predict, average='weighted'))

