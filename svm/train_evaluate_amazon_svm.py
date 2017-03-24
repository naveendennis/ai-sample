from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *


def get_svm(feature_vector, label_train, eachC):

    """

    :param feature_vector:
    :param label_train:
    :param layer_size:
    :param learning_rate_init:
    :param learning_rate:
    :return:
    """
    filename = dir_path + '/../resources/svm_clf_rbf_C'+str(eachC)
    if not os.path.exists(filename):
        from sklearn.svm import SVC
        clf = SVC(C=eachC, kernel='rbf', class_weight='balanced')
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
    feature_train, label_train, feature_test, label_test, p_feature_train, f_size = load_data()

    f_size=1000
    clf = get_svm(p_feature_train, label_train, 6)

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