from itertools import izip
from multiprocessing import Pool
from trainers.logisiticTrainer import LogisticTrainer
from preprocessing.csvCleaner import CsvCleaner
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import string
import numpy as np
import time
import os

class Dataset(object):
    def __init__(self, stage=0, preprocessor_suffix="preprocess.pkl",
                 raw_data_file="Train_clean2.csv", start=0, end=30000,
                 calculate_preprocessors=True,
                 unbalance=(True, 50), preprocessors = None, class_num=0):

        models_folder = "models"
        data_folder = "data"
        self.stage = stage
        self.start = start
        self.class_num = class_num
        self.end = end
        self.raw_data_file = os.path.join(data_folder, raw_data_file)
        self.preprocessor_suffix = preprocessor_suffix
        self.fixed_unbalance, self.unbalance_amount = unbalance
        data_prefix = "data_"
        labels_prefix = "labels_"
        self.data_preprocessor = os.path.join(models_folder, data_prefix+preprocessor_suffix)
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        if preprocessors is None:
            self.tfidf, self.cv = self.get_preprocessors(calculate_preprocessors)
        else:
            self.tfidf, self.cv = preprocessors
        self.inverse_map = np.load(os.path.join(models_folder, "inverse_map.npy"))
        tags = np.array(self.cv.get_feature_names())
        self.word_to_find = tags[self.inverse_map.argsort()][self.class_num]
        self.projection_size = 4000

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
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.0005, max_features=None)
        cv = CountVectorizer(tokenizer=string.split)
        X, Y = self.get_raw_data()
        tfidf.fit(X)
        cv.fit(Y)
        joblib.dump(tfidf, self.data_preprocessor)
        joblib.dump(cv, self.labels_preprocessor)
        return tfidf, cv

    def get_projection_matrix(self, in_dim, out_dim):
        # Probaiblity of  {-1,1} is 1/sqrt(in_dim)
        np.random.seed(15)
        probability = 2*np.sqrt(in_dim)
        P = np.random.randint(probability, size=(in_dim*out_dim)).reshape((in_dim, out_dim))
        P[P > 2] = 1
        P -= 1
        return P

    def get_projected_data(self, X):
        P = self.get_projection_matrix(X.shape[1], self.projection_size)
        return X.dot(P)

    def __iter__(self):
        X, Y = self.get_raw_data()
        for tx, ty in izip(X, Y):
            x = self.tfidf.transform([tx])
            targets = np.array([[1]]) if self.word_to_find in ty.split() else np.array([[-1]])
            yield (x.data, x.indices, x.indptr), targets


def process(class_num):
    training_dataset = Dataset(calculate_preprocessors=False, end=3500000, class_num=class_num,
                   preprocessors=(tfidf, cv))
    new_time = time.time()
    print "Time spent in building the tfidf and cv: "+str(new_time-actual_time)
    print "Creating model for word... "+ training_dataset.word_to_find
    validation_dataset = Dataset(calculate_preprocessors=False,
                                 preprocessors=(tfidf, cv),
                                 start=3500000, end=3502000, class_num=class_num)
    lt = LogisticTrainer(training_dataset, validation_dataset,
                         len(tfidf.vocabulary_), class_num)
    lt.run()

if __name__ == "__main__":
    actual_time = time.time()
    current_class_num = 0
    training_dataset = Dataset(calculate_preprocessors=False, end=3500000, class_num=current_class_num)
    tfidf = training_dataset.tfidf
    cv = training_dataset.cv
    batch_size = 16
    while True:
            classes_to_process = range(current_class_num, current_class_num+batch_size)
            pool = Pool(processes=4)
            pool.map(process, classes_to_process)
            current_class_num += batch_size
