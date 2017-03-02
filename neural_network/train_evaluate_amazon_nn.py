
import numpy as np
import nltk
from utils.read_dataset import read_amazon_csv
import re
import pickle
import os
import errno

REGEX = re.compile("([\w][\w']*\w)")


def tokenize(text):
    """
    It tokenizes the given text
    :param text:
    :return: a list of words in the text
    """

    return [tok.strip() for tok in REGEX.findall(text)]


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def next_char(c):
    """
    Used to obtain the next character
    :param c: a character
    :return: The next character of the character c
    """
    return chr(ord(c) + 1)


def get_label_indices(label_list):

    """
    Used to get labeled indices
    :param label_list:
    :return:
    """

    r_indices = []
    label_val = '1'
    for index in range(0, 5):
        temp = np.where(label_list == label_val)
        r_indices.append(temp)
        label_val = next_char(label_val)

    return r_indices

def get_duplicate_list(vocabulary_list):
    """
    Maintaining a duplicate list
    """
    dup_keys = []
    for each_list in vocabulary_list:
        all_keys =[]
        for each_key in each_list:
            if each_key in all_keys:
                dup_keys.append(each_key)
            all_keys.append(each_key)
    return dup_keys


def get_unique_class_vocabulary(vocabulary_list):
    """

    :param vocabulary_list: Contains a list of a list of vocabulary words.
    :return: vocabulary_list is cleaned so that the same vocabulary words is not present in more than one category
                and then the minimum len of the list in each category is returned
    """

    dup_keys = get_duplicate_list(vocabulary_list)

    """
    Removing duplicate items items from the duplicate list
    """
    vocab = []
    for each_list in vocabulary_list:
        new_vocab = []
        for each_key in each_list:
            if each_key not in dup_keys:
                new_vocab.append(each_key)
        vocab.append(new_vocab)

    return vocab, min([len(each_list) for each_list in vocab])


def build_vocabulary(data_list, r_indices, feature_size):
    """
    Builds a vocabulary based on the indices of each catergory
    :param data_list:
    :param r_indices:
    :param feature_size:
    :return:
    """
    
    cur_vocabulary = []
    for index in range(0, 5):
        word_freq = get_word_freq(data_list, r_indices[index], feature_size)
        cur_vocabulary.append([each for each in word_freq.keys()])

    cur_vocabulary, feature_size = get_unique_class_vocabulary(cur_vocabulary)
    vocabulary = []
    for index in range(0,5):
        vocabulary.append(cur_vocabulary[index][0: feature_size])
    vocabulary = vocabulary[0]

    vocab_dic = {}
    for each_value, index in zip(vocabulary, range(len(vocabulary))):
        vocab_dic[each_value] = index
    return vocab_dic


def pre_process_data(data_list):
    try:
        from nltk.corpus import stopwords
        l_data_list = []
        for index in zip(range(len(data_list))):
            each_cell = data_list[index]
            if type(each_cell) is str:
                new_cell_contents = ''
                '''
                    Changing to lower case
                '''
                words = nltk.word_tokenize(each_cell.lower())
                words = [word.lower() for word in words if word.isalpha()]

                '''
                    Removing stop words
                '''

                words = [word for word in words if word not in stopwords.words('english')]

                '''
                    Lemmatize & Stemming the words
                '''
                from nltk.stem.snowball import SnowballStemmer
                from nltk.stem import WordNetLemmatizer
                stemmer = SnowballStemmer('english')
                lematizer = WordNetLemmatizer()
                words = [lematizer.lemmatize(word) for word in words]

                cell_contents = [stemmer.stem(word) for word in words]

                cell_dup = []
                for each_word in cell_contents:
                    if each_word not in cell_dup:
                        cell_dup.append(each_word)
                        new_cell_contents = new_cell_contents + ' ' + each_word
                new_cell_contents = new_cell_contents.strip()
                l_data_list.append(new_cell_contents)
            else:
                l_data_list.append(' ')
    except Exception as e:
        print(e.with_traceback())
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

    try:
        filename = dir_path + '/../resources/'+op_type+ 'amazon_datalist'
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                l_data_list = pickle.load(f)
                print('data list is loaded ...')
        else:
            with open(filename, "wb") as f:
                l_data_list = pre_process_data(data_list)
                pickle.dump(l_data_list, f)
                print('data list is created...')
    except Exception as e:
        print(e.with_traceback())
        silentremove(filename)
        exit(0)

    '''
        Selecting the features
    '''
    try:
        filename = dir_path + '/../resources/vocabulary_1000'
        if op_type == '' and not os.path.exists(filename):
            with open(filename, "wb") as f:

                vocabulary = build_vocabulary(l_data_list,r_indices,feature_size)
                pickle.dump(vocabulary,f)
                print('vocabulary is created ... ')
        else:
            with open(filename, "rb") as f:
                vocabulary = pickle.load(f)
                print('vocabulary is loaded...')
    except Exception as e:
        print(e.with_traceback())
        silentremove(filename)
        exit(0)
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
    for each_index in r_indices[0]:
        r_data_list = r_data_list + re.findall(r'\w+', data_list[each_index])
    from collections import Counter
    t = Counter(r_data_list).most_common(int(feature_size))
    vocabulary = dict((x, y) for x, y in t)
    return vocabulary


def get_neural_network(feature_vector, label_train, layer_size=(1000, 1000)):
    """

    :param feature_vector:
    :param label_train:
    :param layer_size:
    :return:
    """
    filename = dir_path + '/../resources/neural_network_clf'
    if not os.path.exists(filename):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(activation='logistic',
                            solver='adam',
                            max_iter=2000,
                            hidden_layer_sizes=layer_size,
                            learning_rate='constant',
                            learning_rate_init=0.001)

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
    dir_path = os.path.dirname(os.path.realpath(__file__))

    feature_list, label_list = read_amazon_csv(dir_path + '/../dataset/amazon_dataset/amazon_baby_train.csv')

    from sklearn.model_selection import train_test_split
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature_list , label_list, train_size=0.90)

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

    clf = get_neural_network(p_feature_train, label_train)

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
