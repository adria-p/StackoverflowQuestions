from itertools import izip
from logisiticTrainer import LogisticTrainer

__author__ = 'kosklain'

from neuralTrainer import NeuralTrainer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from csvCleaner import CsvCleaner
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import cPickle
import string
import numpy as np
from scipy.sparse import csr_matrix, vstack
import time


class Dataset(object):
    def __init__(self, stage=0, preprocessor_suffix="preprocess.pkl",
                 raw_data_file="Train.csv", start=0, end=30000,
                 calculate_preprocessors=True,
                 unbalance=(True, 50), preprocessors = None):
        self.stage = stage
        self.start = start
        self.end = end
        self.raw_data_file = raw_data_file
        self.preprocessor_suffix = preprocessor_suffix
        self.fixed_unbalance, self.unbalance_amount = unbalance
        data_prefix = "data_"
        labels_prefix = "labels_"
        self.data_preprocessor = data_prefix+preprocessor_suffix
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        self.tfidf, self.cv = self.get_preprocessors(calculate_preprocessors) if preprocessors is None else preprocessors

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=False)
        Y = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=True)
        return X, Y

    def get_preprocessors(self, calculate):
        if not calculate:
            return joblib.load(self.data_preprocessor), joblib.load(self.labels_preprocessor)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001, max_features=None)
        cv = CountVectorizer(tokenizer=string.split)
        X, Y = self.get_raw_data()
        tfidf.fit(X)
        cv.fit(Y)
        joblib.dump(tfidf, self.data_preprocessor)
        joblib.dump(cv, self.labels_preprocessor)
        return tfidf, cv

    def shuffle_sparse_matrices(self, start, stop, matrix1, matrix2):
        indices = np.arange(start, stop, dtype=np.int32)
        np.random.shuffle(indices)
        return matrix1[indices, :], matrix2[indices]

    def get_sliced_shuffled_data(self, X, Z):
        train_end = round(self.fractions[0]*X.shape[0])
        new_X, new_Z = self.shuffle_sparse_matrices(0, train_end, X, Z)
        VX, VZ = self.shuffle_sparse_matrices(train_end, X.shape[0], X, Z)
        return new_X, VX, new_Z, VZ

    def get_projection_matrix(self, in_dim, out_dim):
        # Probaiblity of  {-1,1} is 1/sqrt(in_dim)
        probability = 2*np.sqrt(in_dim)
        np.random.seed(15)
        P = np.random.randint(probability, size=(in_dim*out_dim)).reshape((in_dim, out_dim))
        P[P > 2] = 1
        P -= 1
        return P

    def get_projected_data(self, X):
        P = self.get_projection_matrix(X.shape[1], 4000)
        return X.dot(P)

    def __iter__(self):
        X, Y = self.get_raw_data()
        offset = len(self.tfidf.vocabulary_)
        random_range = len(self.cv.vocabulary_)
        for tx, ty in izip(X, Y):
            x = self.tfidf.transform([tx])
            y = self.cv.transform([ty])
            _, true_labels = y.nonzero()
            if self.fixed_unbalance:
                random_labels = np.random.randint(random_range - len(true_labels),
                                                  size=self.unbalance_amount-len(true_labels))
            else:
                random_labels = np.random.randint(random_range - len(true_labels),
                                                  size=self.unbalance_amount*len(true_labels))
            for label in true_labels:
                random_labels[random_labels >= label] += 1
            labels = np.concatenate((true_labels, random_labels))
            labels += offset
            data_to_add = np.concatenate((x.data, [1]), axis=0)
            new_data = np.repeat(data_to_add, len(labels))
            indptr_to_add = x.indptr[-1]+1
            new_indptr = np.arange((1+len(labels))*indptr_to_add, step=indptr_to_add)
            targets = np.ones((len(labels), 1))
            targets[len(true_labels):] = -1
            labels = labels.reshape((len(labels), 1))
            new_indices = np.repeat(x.indices.reshape((1, len(x.indices))), len(labels), axis=0)
            new_indices = np.concatenate((new_indices, labels), axis=1)
            new_indices = new_indices.flatten()
            yield (new_data, new_indices, new_indptr), targets




if __name__ == "__main__":
    actual_time = time.time()
    tags_per_example = 5000
    unbalance = (True, tags_per_example)
    training_dataset = Dataset(calculate_preprocessors=False, end=4000000, unbalance=unbalance)
    new_time = time.time()
    print "Time spent in building the tfidf and cv: "+str(new_time-actual_time)
    validation_dataset = Dataset(calculate_preprocessors=False, unbalance=unbalance,
                                 preprocessors=(training_dataset.tfidf, training_dataset.cv),
                                 start=4000000, end=4001000)
    lt = LogisticTrainer(training_dataset, validation_dataset,
                         len(training_dataset.tfidf.vocabulary_)+len(training_dataset.cv.vocabulary_),
                         tags_per_example)
    lt.run()
