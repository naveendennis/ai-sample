import nltk
import re
import pickle
import os
import errno

REGEX = re.compile("([\w][\w']*\w)")
dir_path = os.path.dirname(os.path.realpath(__file__))


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
        temp = [each_val[0] for each_val in enumerate(label_list) if each_val[1] == label_val]
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


def pre_process_data(data_list):
    try:
        from nltk.corpus import stopwords
        l_data_list = []
        for index in range(len(data_list)):
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

                for each_word in cell_contents:
                    new_cell_contents = new_cell_contents + ' ' + each_word
                new_cell_contents = new_cell_contents.strip()
                l_data_list.append(new_cell_contents)
            else:
                l_data_list.append(' ')
    except Exception as e:
        print(e)
    return l_data_list


def build_vocabulary(reviews, max_features=1000, model=None):
    filename = dir_path+'/../resources/bagofwords_vocabulary_10000'
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            vocab = pickle.load(f)
            print('vocabulary is loaded')
    else:
        from collections import defaultdict
        vocab_counter = defaultdict(int)
        for each_review in reviews:
            for each_word in tokenize(each_review):
                vocab_counter[each_word] += 1
        dict_tracker = dict()

        counts = list()
        for key, value in vocab_counter.items():
            dict_tracker[value] = key
            counts.append(value)

        counts = sorted(counts, reverse=True)
        if model is None:
            if len(counts) < max_features:
                raise ValueError('possibly empty vocabulary or unable to extract '+str(max_features)+ ' features')
            vocab = [dict_tracker[each_count] for each_count in counts[:max_features]]
        else:
            vocab = []
            counter = 0
            for each_count in counts:
                if counter == max_features:
                    break;
                try:
                    model[dict_tracker[each_count]]
                    vocab.append(dict_tracker[each_count])
                    counter += 1
                except KeyError:
                    pass

        with open(filename, "wb") as f:
            pickle.dump(vocab, f)
            print('vocabulary is created')

    return vocab


def get_bag_of_words_features(reviews, max_features=1000, opt='training'):

    filename = dir_path + '/../resources/'+opt+'_features'
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            features = pickle.load(f)
            print('features are loaded')
    else:
        features = []
        from collections import defaultdict
        vocabulary = build_vocabulary(reviews=reviews, max_features=max_features)
        for each_review in reviews:
            review_feature = []
            state_tracker = defaultdict(int)
            for each_word in tokenize(each_review):
                state_tracker[each_word] += 1

            for each_word in vocabulary:
                review_feature.append(state_tracker[each_word])
            features.append(review_feature)
        with open(filename, "wb") as f:
            pickle.dump(features, f)
            print(opt+' features are created')

    return features


def get_sentences(data_list):
    sentences = []
    for each_sentence in data_list:
        if type(each_sentence) is str:
            sentences.append([each_word for each_word in tokenize(each_sentence)])
        else:
            sentences.append([])
    return sentences


def get_model(sentences=[], op_type=None):
    op_type = 'training' if op_type is None else op_type
    from gensim.models import Word2Vec
    w2v_file_name=dir_path + '/../resources/w2v_model'
    if op_type is 'training' and not os.path.exists(w2v_file_name):
        sentences = get_sentences(sentences)
        model = Word2Vec(sentences=sentences)
        pickle.dump(model,open(w2v_file_name, 'wb'))
    else:
        model = pickle.load(open(w2v_file_name,'rb'))
    return model


def get_w2v_features(data_list, op_type=None, feature_size=100):
    model = get_model(data_list, op_type=op_type)
    vocabulary = build_vocabulary(data_list, max_features=feature_size, model=model)
    features = []
    cache_filename = dir_path + '/../resources/' + op_type + '_cached_counter'
    if not os.path.exists(cache_filename):
        current_index = 0
    else:
        current_index = pickle.load(open(cache_filename, 'rb'))
    revised_datalist = data_list[current_index:]
    for each_sentence in revised_datalist:
        each_feature = []
        for each_vocab in vocabulary:
            if each_vocab in each_sentence:
                try:
                    each_feature += model[each_vocab].tolist()
                except KeyError:
                    each_feature += [0]*100
            else:
                each_feature += [0]*100
        features.append(each_feature)
        current_index += 1
        pickle.dump(current_index, open(cache_filename, 'wb'))
    from sklearn.manifold import TSNE
    model = TSNE(n_components=feature_size)
    return model.fit_transform(features)


def get_features(data_list, label_list, feature_size=1000, op_type=None):

    """
    Returns a feature vector after feature selection
    :param data_list: contains the review text
    :param label_list: contains the classified review rating
    :param feature_size: the size of the feature vector
    :param op_type: the type of operation performed
    :return: feature vector of size feature_size
    """
    op_type = 'training' if op_type is None else op_type

    try:
        filename = dir_path + '/../resources/'+op_type+ '_amazon_datalist'
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                l_data_list = pickle.load(f)
                print('data list is loaded ...')
        else:
            l_data_list = pre_process_data(data_list)
            with open(filename, "wb") as f:
                pickle.dump(l_data_list, f)
                print('data list is created...')
    except Exception as e:
        print(e)
        silentremove(filename)
        exit(0)

    return get_bag_of_words_features(reviews=l_data_list, opt=op_type, max_features=feature_size)
    # return get_w2v_features(data_list=l_data_list, op_type=op_type, feature_size=feature_size)


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


def train_test_split(feature_list, label_list):
    r_indices = get_label_indices(label_list)
    feature_list = feature_list[:,1]
    from math import ceil
    class_size = ceil(min([len(each) for each in r_indices]) * 0.90)
    # class_size = 9000
    label_train = []
    label_test = []
    feature_train = []
    feature_test = []
    train_indices = []
    for each_class in r_indices:
        for index in range(class_size):
            train_indices.append(each_class[index])

    for index in train_indices:
        label_train.append(label_list[index])
        feature_train.append(feature_list[index])

    other_indices = [each for each in range(len(label_list)) if each not in train_indices]
    for index in other_indices:
        label_test.append(label_list[index])
        feature_test.append(feature_list[index])
    feature_train, label_train = shuffle(feature_train, label_train)
    feature_test, label_test = shuffle(feature_test, label_test)
    return feature_train, feature_test, label_train, label_test


def shuffle(feature, label):
    size = len(label)
    import random as rand
    indices = [x for x in range(size)]
    rand.shuffle(indices)
    new_feature=[]
    new_label = []
    for index in indices:
        new_feature.append(feature[index])
        new_label.append(label[index])
    return feature, label


def load_data(feature_size=1000):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    from utils.read_dataset import read_amazon_csv
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

    filename = dir_path+ '/../resources/rec_features_10000'
    if not os.path.exists(filename):
        p_feature_train = get_features(feature_train, label_train, feature_size=feature_size)

        with open(filename, "wb") as f:
            pickle.dump(p_feature_train, f)

        print('pickle created for features in training set...')

    else:
        with open(filename,'rb') as f:
            p_feature_train = pickle.load(f)
        print('pickle loaded for training features...')

    return feature_train, label_train, feature_test, label_test, p_feature_train, feature_size
