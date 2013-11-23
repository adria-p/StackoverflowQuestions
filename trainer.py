from itertools import izip

__author__ = 'kosklain'

from neuralTrainer import NeuralTrainer
from sklearn.linear_model import SGDClassifier
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
                 proportion_true_false=1, calculate_preprocessors=True):
        self.stage = stage
        self.start = start
        self.end = end
        self.raw_data_file = raw_data_file
        self.preprocessor_suffix = preprocessor_suffix
        self.proportion = proportion_true_false
        data_prefix = "data_"
        labels_prefix = "labels_"
        self.data_preprocessor = data_prefix+preprocessor_suffix
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        self.tfidf, self.cv = self.get_preprocessors(calculate_preprocessors)

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
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000)
        cv = CountVectorizer(tokenizer=string.split)
        X, Y = self.get_raw_data()
        tfidf.fit(X)
        cv.fit(Y)
        joblib.dump(tfidf, self.data_preprocessor)
        joblib.dump(cv, self.labels_preprocessor)
        return tfidf, cv

    def get_final_testing_data(self, X, Y):
        num_tags = Y.shape[1]
        offset = X.shape[1]
        targets = np.zeros(num_tags*Y.shape[0])
        current_row = 0
        new_indices = []
        new_data = []
        new_indptr = [0]
        last_indptr = 0
        for x, y in zip(X, Y):
            _, labels = y.nonzero()
            labels += offset
            data_to_add = np.concatenate((x.data,[1]), axis=0)
            new_data.extend(np.repeat(data_to_add, len(labels)))
            indptr_to_add = x.indptr[-1]+1
            for label in range(offset, offset+num_tags):
                targets[current_row] = 1 if label in labels else -1
                new_indices.extend(np.concatenate((x.indices, [label])))
                last_indptr += indptr_to_add
                new_indptr.append(last_indptr)
                current_row += 1
        training = self.build_matrix((new_data, new_indices, new_indptr),
                                     (num_tags, Y.shape[1]+X.shape[1]))
        #targets = targets.reshape(targets.shape[0], 1)
        f = open(self.data_final_test, 'wb')
        cPickle.dump(training, f)
        f.close()
        f = open(self.target_final_test, 'wb')
        cPickle.dump(targets, f)
        f.close()
        return training, targets

    def shuffle_sparse_matrices(self, start, stop, matrix1, matrix2):
        indices = np.arange(start, stop, dtype=np.int32)
        np.random.shuffle(indices)
        return matrix1[indices, :], matrix2[indices]

    def get_sliced_shuffled_data(self, X, Z):
        train_end = round(self.fractions[0]*X.shape[0])
        new_X, new_Z = self.shuffle_sparse_matrices(0, train_end, X, Z)
        VX, VZ = self.shuffle_sparse_matrices(train_end, X.shape[0], X, Z)
        return new_X, VX, new_Z, VZ

    def build_matrix(self, info, shape):
        training_sparse = csr_matrix(info, shape=shape, dtype=np.float64)
        return training_sparse

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

    def run(self):
        print "Num examples: %d" % (self.end - self.start)
        actual_time = time.time()
        self.tfidf, self.cv = self.get_preprocessors()
        new_time = time.time()
        print "Time spent in building the preprocessors: "+str(new_time-actual_time)


    def __iter__(self):
        X, Y = self.get_raw_data()
        offset = len(self.tfidf.vocabulary_)
        random_range = len(self.cv.vocabulary_)
        for tx, ty in izip(X, Y):
            x = self.tfidf.transform([tx])
            y = self.cv.transform([ty])
            new_indices = []
            new_data = []
            new_indptr = [0]
            last_indptr = 0
            _, true_labels = y.nonzero()
            random_labels = np.random.randint(random_range - len(true_labels),
                                              size=self.proportion*len(true_labels))
            for label in true_labels:
                random_labels[random_labels >= label] += 1
            labels = np.concatenate((true_labels, random_labels))
            labels += offset
            data_to_add = np.concatenate((x.data, [1]), axis=0)
            new_data.extend(np.repeat(data_to_add, len(labels)))
            indptr_to_add = x.indptr[-1]+1
            targets = [1]*(len(true_labels))
            targets.extend([-1]*(len(true_labels)*self.proportion))
            for label in labels:
                new_indices.extend(np.concatenate((x.indices, [label])))
                last_indptr += indptr_to_add
                new_indptr.append(last_indptr)
            training = self.build_matrix((new_data, new_indices, new_indptr),
                                        (len(labels), offset+random_range))
            yield training, targets

        #f = open("model.pkl", 'wb')
        #cPickle.dump(model, f)
        #f.close()
        #TX, TZ = self.get_final_testing_data(TX, TZ)
        #TX = self.get_projected_data(TX)
        #predictions = model.predict(TX)



if __name__ == "__main__":
    training_dataset = Dataset(calculate_preprocessors=True, end=30000)
    validation_dataset = Dataset(calculate_preprocessors=False, start=30000, end=50000)
    batch_size = 500
    actual_time = time.time()
    model = SGDClassifier(loss="log", verbose=3, n_jobs=4, random_state=2, n_iter=5)
    current_batch_X = []
    current_batch_Y = []
    for i, (x, y) in enumerate(training_dataset):
        current_batch_X.append(x)
        current_batch_Y.append(y)
        if (i+1) % batch_size == 0:
            X = vstack(current_batch_X)
            Y = np.concatenate(current_batch_Y, axis=0)
            model.partial_fit(X,Y, classes=[-1, 1])
            current_batch_X = []
            current_batch_Y = []
    new_time = time.time()
    print "Time spent in training: "+str(new_time-actual_time)
    actual_time = new_time

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    current_batch_X = []
    current_batch_Y = []
    for i, (x, y) in enumerate(validation_dataset):
        if len(y) == 0:
            current_batch_X.append(current_batch_X[-1])
            current_batch_Y.append([1])
        else:
            current_batch_X.append(x)
            current_batch_Y.append(y)
        if (i+1) % batch_size == 0:
            X = vstack(current_batch_X, format="csr")
            Y = np.concatenate(current_batch_Y, axis=0)
            predictions = model.predict(X)
            for pred, gt in zip(predictions, Y):
                if gt == 1:
                    if pred == 1:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if pred == 1:
                        FP += 1
                    else:
                        TN += 1
            current_batch_X = []
            current_batch_Y = []

    new_time = time.time()
    print "Time spent in predicting: "+str(new_time-actual_time)
    print TP, FP, TN, FN
    print "Final score: %f" % ((TP+TN+0.0)/(TP+TN+FP+FN+0.0))