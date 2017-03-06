from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *

def get_decision_tree(feature_vector, label_train):
    """

    :param feature_vector:
    :param label_train:
    :param layer_size:
    :return:
    """
    filename = dir_path + '/../resources/decision_tree_clf'
    if not os.path.exists(filename):

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(min_samples_split=100, max_depth=100)

        clf.fit(p_feature_train, label_train)
        with open(filename, "wb") as f:
            pickle.dump(clf, f)

        print('pickle created for model...')

    else:
        with open(filename,'rb') as f:
            clf = pickle.load(f)

        print('pickle loaded for model...')
    return clf



if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    feature_list, label_list = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_train.csv')

    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature_list , label_list, train_size=0.90, random_state=True)

    f_size = 1000
    filename = dir_path+ '/../resources/rec_features_1000'
    try:
        if not os.path.exists(filename):
            p_feature_train = get_features(feature_train[:,1],label_train, feature_size=f_size)

            with open(filename, "wb") as f:
                pickle.dump(p_feature_train, f)

            print('pickle created for features in training set...')

        else:
            with open(filename,'rb') as f:
                p_feature_train = pickle.load(f)
            print('pickle loaded for training features...')
    except Exception as e:
        print(e.with_traceback())
        silentremove(filename)
        exit(0)

    clf = get_decision_tree(p_feature_train, label_train)

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
