from sklearn.linear_model.logistic import LogisticRegression

__author__ = 'kosklain'

from neuralTrainer import NeuralTrainer
from csvCleaner import CsvCleaner
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import cPickle
import string
import numpy as np
from scipy.sparse import csr_matrix


class Trainer:
    def __init__(self, stage=0, preprocessor_suffix="preprocess.pkl",
                 raw_data_file="Train.csv", num_examples=30000,
                 sparse_data_suffix="sparse.pkl", final_train_suffix="final_train.pkl",
                 final_test_suffix="final_test.pkl", proportion_true_false=1):
        self.stage = stage
        self.num_examples = num_examples
        self.raw_data_file = raw_data_file
        self.preprocessor_suffix = preprocessor_suffix
        self.proportion = proportion_true_false
        data_prefix = "data_"
        labels_prefix = "labels_"
        targets_prefix = "targets_"
        self.data_preprocessor = data_prefix+preprocessor_suffix
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        self.data_processed = data_prefix+sparse_data_suffix
        self.labels_processed = labels_prefix+sparse_data_suffix
        self.data_final_train = data_prefix+final_train_suffix
        self.target_final_train = targets_prefix+final_train_suffix
        self.data_final_test = data_prefix+final_test_suffix
        self.target_final_test = targets_prefix+final_test_suffix
        self.fractions = [0.5, 0.75]

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, end=self.num_examples,
                       only_tags=False)
        Y = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, end=self.num_examples,
                       only_tags=True)
        return X, Y

    def get_preprocessors(self):
        if self.stage == 1:
            return joblib.load(self.data_preprocessor), joblib.load(self.labels_preprocessor)
        if self.stage > 1:
            return None, None
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1000)
        cv = CountVectorizer(tokenizer=string.split)
        X, Y = self.get_raw_data()
        tfidf.fit(X)
        cv.fit(Y)
        joblib.dump(tfidf, self.data_preprocessor)
        joblib.dump(cv, self.labels_preprocessor)
        return tfidf, cv

    def get_transfomed_data(self, tfidf, cv):
        if self.stage == 2:
            f = open(self.data_processed, "rb")
            f2 = open(self.labels_processed, "rb")
            return cPickle.load(f), cPickle.load(f2)
        if self.stage > 2:
            return None
        X, Y = self.get_raw_data()
        transformed_X = tfidf.transform(X)
        transformed_Y = cv.transform(Y)
        f = open(self.data_processed, 'wb')
        cPickle.dump(transformed_X, f)
        f.close()
        f = open(self.labels_processed, 'wb')
        cPickle.dump(transformed_Y, f)
        f.close()
        return transformed_X, transformed_Y

    def get_final_training_data(self, X, Y):
        if self.stage == 3:
            f = open(self.data_final_train, "rb")
            f2 = open(self.target_final_train, "rb")
            return cPickle.load(f), cPickle.load(f2)
        if self.stage > 3:
            return None, None
        num_tags = len(Y.data)*(1+self.proportion)
        offset = X.shape[1]
        random_range = Y.shape[1]
        targets = np.zeros(num_tags)
        current_row = 0
        new_indices = []
        new_data = []
        new_indptr = [0]
        last_indptr = 0
        np.random.seed(67)
        for x, y in zip(X, Y):
            _, true_labels = y.nonzero()
            random_labels = np.random.randint(random_range - len(true_labels),
                                              size=self.proportion*len(true_labels))
            for label in true_labels:
                random_labels[random_labels >= label] += 1
            labels = np.concatenate((true_labels, random_labels))
            labels += offset
            data_to_add = np.concatenate((x.data,[1]), axis=0)
            new_data.extend(np.repeat(data_to_add, len(labels)))
            indptr_to_add = x.indptr[-1]+1
            targets[current_row:current_row+len(true_labels)] = [1]*(len(true_labels))
            targets[current_row+len(true_labels):current_row+len(labels)] = [-1]*(len(true_labels)*self.proportion)
            for label in labels:
                new_indices.extend(np.concatenate((x.indices, [label])))
                last_indptr += indptr_to_add
                new_indptr.append(last_indptr)
                current_row += 1
        training = self.build_matrix((new_data, new_indices, new_indptr),
                                     (num_tags, Y.shape[1]+X.shape[1]))
        #targets = targets.reshape(targets.shape[0], 1)
        f = open(self.data_final_train,'wb')
        cPickle.dump(training, f)
        f.close()
        f = open(self.target_final_train,'wb')
        cPickle.dump(targets, f)
        f.close()
        return training, targets

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
        tfidf, cv = self.get_preprocessors()
        X, Z = self.get_transfomed_data(tfidf, cv)
        val_end = round(self.fractions[1]*X.shape[0])
        TX = X[val_end:]
        TZ = Z[val_end:]
        X = X[:val_end]
        Z = Z[:val_end]
        X, Z = self.get_final_training_data(X, Z)
        X = self.get_projected_data(X)
        X, VX, Z, VZ = self.get_sliced_shuffled_data(X, Z)
        model = LogisticRegression()
        model.fit(X, Z)
        predictions = model.predict(VX)
        final_score = 0
        for pred, gt in zip(predictions, VZ):
            if pred == gt:
                final_score += 1
        print (final_score+0.0)/(0.0+len(VZ))
        #f = open("model.pkl", 'wb')
        #cPickle.dump(model, f)
        #f.close()
        #TX, TZ = self.get_final_testing_data(TX, TZ)
        #TX = self.get_projected_data(TX)
        #predictions = model.predict(TX)



if __name__ == "__main__":
    trainer = Trainer(stage=0, num_examples=30000)
    trainer.run()