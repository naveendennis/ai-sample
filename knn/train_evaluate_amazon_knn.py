from utils.read_dataset import read_amazon_csv
from utils.amazon_utils import *

def get_knn_classifier(feature_vector, label_vector):
    filename = dir_path + '/../resources/knneighbours'
    if not os.path.exists(filename):

        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=13, weights='distance')
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
    feature_train, label_train, feature_test, label_test, p_feature_train, f_size = load_data()

    clf = get_knn_classifier(p_feature_train, label_train)

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

