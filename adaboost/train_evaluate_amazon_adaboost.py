from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *


def get_adaboost_classifier(feature_vector, label_vector):
    filename = dir_path + '/../resources/adaboost_clf'
    if not os.path.exists(filename):

        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=10000, algorithm='SAMME.R', base_estimator=None)
        clf.fit(feature_vector, label_vector)
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

        print('pickle created for model...')

    else:
        with open(filename, 'rb') as f:
            clf = pickle.load(f)

        print('pickle loaded for model...')
    return clf

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    feature_list, label_list = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_train.csv')

    filename = dir_path + '/../resources/raw_features'
    try:
        if not os.path.exists(filename):
            feature_train, feature_test, label_train, label_test = train_test_split(
                feature_list, label_list)

            with open(filename, "wb") as f:
                pickle.dump(feature_train, f)
                pickle.dump(feature_test, f)
                pickle.dump(label_train, f)
                pickle.dump(label_test,f)

            print('pickle created for raw features...')

        else:
            with open(filename, 'rb') as f:
                feature_train = pickle.load(f)
                feature_test = pickle.load(f)
                label_train = pickle.load(f)
                label_test = pickle.load(f)
            print('pickle loaded for raw features...')

    except Exception as e:
        print(e)
        silentremove(filename)
        exit(0)

    f_size = 100
    filename = dir_path+ '/../resources/rec_features_1000'
    try:
        if not os.path.exists(filename):
            p_feature_train = get_features(feature_train,label_train, feature_size=f_size)

            with open(filename, "wb") as f:
                pickle.dump(p_feature_train, f)

            print('pickle created for features in training set...')

        else:
            with open(filename,'rb') as f:
                p_feature_train = pickle.load(f)
            print('pickle loaded for training features...')
    except Exception as e:
        print(e)
        silentremove(filename)
        exit(0)

    clf = get_adaboost_classifier(p_feature_train, label_train)

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

