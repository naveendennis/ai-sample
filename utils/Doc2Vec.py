from gensim.models import Doc2Vec
import csv
import os.path
import pickle
from gensim.models.doc2vec import TaggedDocument

dir_path = os.path.dirname(os.path.realpath(__file__))
resources_path = dir_path+'/../resources/'

training_file = dir_path+'/../dataset/amazon_dataset/amazon_baby_train.csv'
reviews_file = resources_path + 'reviews_and_labels'
doc2vec_file = resources_path+ 'doc2vec_sentences'


class DocIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            # print 'creating tagged document...%d' % idx
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])


def get_model():
    reviews = []
    labels = []
    tags = []
    if not os.path.exists(doc2vec_file):
        if not os.path.exists(reviews_file):
            with open(training_file, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                next(reader)
                iterator = 0
                for each_row in reader:
                    reviews.append(each_row[1])
                    labels.append(each_row[2])
                    tags.append(str(iterator))
                    iterator += 1

            with open(reviews_file, 'wb') as f:
                pickle.dump(reviews, f)
                pickle.dump(labels, f)
                pickle.dump(tags, f)

            print('reviews and labels created...')
        else:
            with open(reviews_file, 'rb') as f:
                reviews = pickle.load(f)
                labels = pickle.load(f)
                tags = pickle.load(f)
            print('reviews and labels loaded...')

        sentences = DocIterator(reviews, tags)

        model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate
        model.build_vocab(sentences)
        for epoch in range(10):
            model.train(sentences)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        model.save(doc2vec_file)
        print('doc2vec is created...')

    else:
        model = Doc2Vec.load(doc2vec_file)
        print('doc2vec is loaded...')

    return model


if __name__ == '__main__':
    model = get_model()
    print(model)