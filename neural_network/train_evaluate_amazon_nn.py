from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *


def get_neural_network(feature_vector, label_train, layer_size=(1000, 1000), learning_rate_init=0.001, learning_rate='constant'):

    """

    :param feature_vector:
    :param label_train:
    :param layer_size:
    :param learning_rate_init:
    :param learning_rate:
    :return:
    """
    filename = dir_path + '/../resources/neural_network_clf'
    if not os.path.exists(filename):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(activation='relu',
                            solver='adam',
                            max_iter=2000,
                            hidden_layer_sizes=layer_size,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init)

        clf.fit(feature_vector, label_train)
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

        print('pickle created for model...')

    else:
        with open(filename,'rb') as f:
            clf = pickle.load(f)

        print('pickle loaded for model...')
    return clf


def run_with_test_set(clf):
    feature_test, label_test = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_test.csv')
    feature_test = get_features(feature_test, label_test, feature_size=f_size, op_type='test')
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
    feature_train, label_train, feature_test, label_test, p_feature_train, f_size = load_data(feature_size=1000)

    clf = get_neural_network(p_feature_train, label_train)

    feature_test = get_features(feature_test, label_test, op_type='test', feature_size=f_size)
    label_predict = clf.predict(feature_test)

    from sklearn.metrics import precision_score
    precision = precision_score(label_test, label_predict, average='weighted')

    from sklearn.metrics import recall_score
    recall = recall_score(label_test, label_predict, average='weighted')

    f1 = 2 * (precision * recall) / (precision + recall)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(label_test, label_predict)

    print(str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(accuracy))

    # run_with_test_set(clf)