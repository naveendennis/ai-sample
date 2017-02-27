
import numpy as np
import nltk
from utils.read_dataset import read_amazon_csv
import re
import pickle
import os.path

REGEX = re.compile("([\w][\w']*\w)")


def tokenize(text):
    """
    It tokenizes the given text
    :param text:
    :return: a list of words in the text
    """

    return [tok.strip().lower() for tok in REGEX.findall(text)]


def next_char(c):
    return chr(ord(c) + 1)


def get_label_indices(label_list):

    r_indices = np.negative(np.ones(shape=(5, len(label_list))))
    label_val = '1'
    for index in range(0, 5):
        r_indices[index,:] = np.where(label_list == label_val)
        label_val = next_char(label_val)

    return r_indices


def build_vocabulary(data_list, r_indices, feature_size):
    v_start = 0

    for index in range(1, 5):
        word_freq = get_word_freq(data_list, r_indices[index], feature_size)
        vocabulary = [each for each in word_freq.keys()]
        v_end = len(vocabulary)
        for each_key in r_indices[index]:
            if each_key not in vocabulary:
                vocabulary.append(each_key)
        o_v_size = v_end - v_start
        n_v_size = len(vocabulary) - v_end
        if n_v_size < o_v_size:
            mod_vocabulary = vocabulary[0: v_start + n_v_size]
            mod_vocabulary[len(mod_vocabulary): len(mod_vocabulary) + n_v_size] = vocabulary[v_end: v_end + n_v_size]
            vocabulary = mod_vocabulary
        v_start = v_end

    return vocabulary


def pre_process_data():
    from nltk.corpus import stopwords
    l_data_list = np.array([])
    for each_row in l_data_list:
        cell_dup = []
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
                    if each_word not in cell_dup:
                        cell_dup.append(each_word)
                        new_cell_contents = new_cell_contents + ' ' + each_word

                new_cell_contents = new_cell_contents.strip()

        l_data_list = np.append(l_data_list, np.array([new_cell_contents]), axis=0)
    return l_data_list


def get_features(data_list, label_list, feature_size=500, op_type=''):

    """
    Returns a feature vector after feature selection
    :param data_list: contains the review text
    :param label_list: contains the classified review rating
    :param feature_size: the size of the feature vector
    :param op_type: the type of operation performed
    :return: feature vector of size feature_size
    """

    r_indices = get_label_indices(label_list)



    if os.path.exists(dir_path + '/../resources/'+op_type+ 'amazon_datalist'):
        with open(dir_path + '/../resources/'+op_type+ 'amazon_datalist', "rb") as f:
            l_data_list = pickle.load(f)
            print('data list is loaded ...')
    else:
        with open(dir_path + '/../resources/'+op_type+ 'amazon_datalist', "wb") as f:
            l_data_list = pre_process_data()

            pickle.dump(l_data_list, f)

    '''
        Selecting the features
    '''

    if op_type == '' and not os.path.exists(dir_path + '/../resources/vocabulary_1000'):
        with open(dir_path + '/../resources/vocabulary_1000', "wb") as f:

            vocabulary = build_vocabulary(l_data_list,r_indices,feature_size)
            pickle.dump(vocabulary,f)
            print('vocabulary is created ... ')
    else:
        with open(dir_path + '/../resources/vocabulary_1000', "rb") as f:
            vocabulary = pickle.load(f)
            print('vocabulary is loaded...')

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocabulary, tokenizer=tokenize)
    features = vectorizer.fit_transform(l_data_list)
    features = features.toarray()
    return features

def get_word_freq(data_list, r_indices, feature_size):

    """
    It returns the word frequencies of the text
    :param data_list: the text review
    :param r_indices: indices to be considered for calculating the word frequencies
    :param feature_size: size of the feature vector
    :return: words along with their frequencies
    """

    r_data_list = []
    for each_index in r_indices:
        r_data_list = r_data_list + re.findall(r'\w+', data_list[each_index])
    from collections import Counter
    t = Counter(r_data_list).most_common(int(feature_size))
    vocabulary = dict((x, y) for x, y in t)
    return vocabulary

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    feature_list, label_list = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_train.csv')

    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature_list , label_list, train_size=0.90)

    f_size = 1000
    if not os.path.exists(dir_path+ '/../resources/rec_features_1000'):
        p_feature_train = get_features(feature_train,label_train, feature_size=f_size)

        with open(dir_path+'/../resources/rec_features_1000', "wb") as f:
            pickle.dump(p_feature_train, f)

        print('pickle created for features in training set...')

    else:
        with open(dir_path+'/../resources/rec_features_1000','rb') as f:
            p_feature_train = pickle.load(f)

        print('pickle loaded for training features...')

    '''
        Training the decision tree
    '''
    if not os.path.exists(dir_path + dir_path + '/../resources/neural_network_clf'):
        from sklearn.neural_network import MLPClassifier

        layer_size = 1000
        clf = MLPClassifier(activation='logistic',
                            solver='adam',
                            max_iter=2000,
                            hidden_layer_sizes=(layer_size, layer_size),
                            learning_rate='constant',
                            learning_rate_init=0.001)

        clf.fit(p_feature_train, label_train)
        with open(dir_path + '/../resources/neural_network_clf', "wb") as f:
            pickle.dump(clf, f)

        print('pickle created for model...')

    else:
        with open(dir_path + '/../resources/neural_network_clf','rb') as f:
            clf = pickle.load(f)

        print('pickle loaded for model...')



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
