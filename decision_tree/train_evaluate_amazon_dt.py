import numpy as np
import nltk
from utils.read_dataset import read_amazon_csv
import re
import pickle
import os.path

REGEX = re.compile("([\w][\w']*\w)")


def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.findall(text)]


def get_features(data_list):
    l_data_list = np.array([])

    from nltk.corpus import stopwords
    for each_row in data_list:
        count = 0
        new_cell_contents = ''

        for each_cell in each_row:
            if type(each_cell) is str:
                '''
                    Changing to lower case
                '''
                words = nltk.word_tokenize(each_cell.lower())
                words = [word.lower() for word in words if word.isalpha()]

                '''
                    Stemming the words
                '''
                from nltk.stem.snowball import SnowballStemmer
                stemmer = SnowballStemmer('english')
                words = [stemmer.stem(word) for word in words]

                '''
                    Removing stop words
                '''

                cell_contents = [word for word in words if word not in stopwords.words('english')]

                for each_word in cell_contents:
                    new_cell_contents = new_cell_contents +' '+each_word

                new_cell_contents = new_cell_contents.strip()

        l_data_list = np.append(l_data_list, np.array([new_cell_contents]), axis=0)

    '''
        Selecting the features
    '''
    feature_size = 500
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(tokenizer=tokenize, max_features=feature_size)
    features = vectorizer.fit_transform(l_data_list)
    features = features.toarray()

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=feature_size)
    weightVector = vectorizer.fit_transform(l_data_list)
    weightVector = weightVector.toarray()

    return features * weightVector


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    feature_list, label_list = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_test.csv')

    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature_list , label_list, train_size=0.90)

    if not os.path.exists('features_500'):
        p_feature_train = get_features(feature_train)

        with open("features_500", "wb") as f:
            pickle.dump(p_feature_train, f)

        print('pickle created...')

    else:
        with open('features_500','rb') as f:
            p_feature_train = pickle.load(f)

        print('pickle loaded...')

    '''
        Training the decision tree
    '''
    # try:
    from sklearn import tree

    clf = tree.DecisionTreeClassifier(min_samples_split=100)

    clf.fit(p_feature_train, label_train)

    test_features = get_features(feature_test)
    label_predict = clf.predict(test_features)

    from sklearn.metrics import precision_score
    precision = precision_score(label_test, label_predict, average='weighted')

    from sklearn.metrics import recall_score
    recall = recall_score(label_test, label_predict, average='weighted')

    f1 = 2 * (precision * recall) / (precision + recall)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(label_test, label_predict)

    print(str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(accuracy))
