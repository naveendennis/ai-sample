import os

import numpy as np

from utils.read_dataset import read_digits_csv

"""
    test with test set
"""


def run_with_test_set(clf):
    from utils.read_dataset import custom_read_csv
    test_dataset = custom_read_csv(dir_path + '/../dataset/digit_dataset/optdigits_test.csv', feature_size)
    feature_test = test_dataset[0:test_dataset.shape[0], 0:feature_size]
    label_test = test_dataset[0:test_dataset.shape[0], feature_size]
    label_predict = clf.predict(feature_test)
    from sklearn.metrics import precision_score
    precision = precision_score(label_test, label_predict, average='weighted')
    from sklearn.metrics import recall_score
    recall = recall_score(label_test, label_predict, average='weighted')
    f1 = 2 * (precision * recall) / (precision + recall)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(label_test, label_predict)
    print(str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(accuracy))


if __name__ == '__main__':

    """
        read dataset
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    feature_size = 64
    feature_list, label_list = read_digits_csv(dir_path + '/../dataset/digit_dataset/optdigits_raining.csv', feature_size)
    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature_list , label_list, train_size=0.95)
    train_features = np.array([[]])
    from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.neighbors import NearestNeighbors
    clf = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto',p=1)
    # clf = NearestNeighbors()
    # graph = clf.kneighbors_graph(n_neighbors=7,X=feature_train)
    clf.fit(feature_train, label_train)
    label_predict = clf.predict(feature_test)
    from sklearn.metrics import precision_score
    precision = precision_score(label_test, label_predict, average='weighted')

    from sklearn.metrics import recall_score
    recall = recall_score(label_test, label_predict, average='weighted')

    f1 = 2 * (precision * recall) / (precision + recall)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(label_test, label_predict)

    # print('estimator size:', estimator_size)
    print(str(precision)+'\t'+ str(recall)+'\t' +str(f1)+'\t'+ str(accuracy))
    print()