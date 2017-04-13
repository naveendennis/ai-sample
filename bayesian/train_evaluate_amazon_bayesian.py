from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *


def get_bayesian_classifier(feature_vector, label_train, eachC):

    """
    :param feature_vector:
    :param label_train:
    :param layer_size:
    :param learning_rate_init:
    :param learning_rate:
    :return:
    """
    filename = dir_path + '/../resources/bayesian_classifier'
    if not os.path.exists(filename):
        from sklearn.naive_bayes import GaussianNB
        # class_probabilities = [len(each_class)/len(label_train) for each_class in
        #                        [[each_value for each_value in
        #                          label_train if each_value == each_class_value] for each_class_value in set(label_train)]
        #                        ]
        # class_probabilities = [0.5841347463629925, 0.0910205416008282, 0.18054950144390564, 0.06157031547975808, 0.08272489511251567]
        class_probabilities = [0.5, 0.15,0.25,0.05,0.05]
        clf = GaussianNB(priors=class_probabilities)
        print(clf)
        clf.fit(feature_vector, label_train)
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

        print('pickle created for model...')

    else:
        with open(filename,'rb') as f:
            clf = pickle.load(f)

        print('pickle loaded for model...')
    return clf


if __name__ == '__main__':
    feature_train, label_train, feature_test, label_test, p_feature_train, f_size = load_data(feature_size=1000)

    clf = get_bayesian_classifier(p_feature_train, label_train, 6)

    test_features = get_features(feature_test, label_test, feature_size=f_size, op_type='test')
    label_predict = clf.predict(test_features)

    from sklearn.metrics import precision_score
    precision = precision_score(label_test, label_predict, average='weighted')

    from sklearn.metrics import recall_score
    recall = recall_score(label_test, label_predict, average='weighted')

    f1 = 2 * (precision * recall) / (precision + recall)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(label_test, label_predict)

    print(str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(accuracy))