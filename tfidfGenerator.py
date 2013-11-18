

__author__ = 'kosklain'
from csvCleaner import CsvCleaner
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import cPickle
import string

class Trainer:
    def __init__(self, stage=0, preprocessor_suffix = "preprocess.pkl",
                 raw_data_file="Train.csv", num_examples=30000,
                 sparse_data_suffix = "sparse.pkl", final_train_suffix= "final_train.pkl"):
        self.stage = stage
        self.num_examples = num_examples
        self.raw_data_file = raw_data_file
        self.preprocessor_suffix = preprocessor_suffix
        data_prefix = "data_"
        labels_prefix = "labels_"
        self.data_preprocessor = data_prefix+preprocessor_suffix
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        self.data_processed = data_prefix+sparse_data_suffix
        self.labels_processed = labels_prefix+sparse_data_suffix
        self.data_final_train = data_prefix+final_train_suffix
        self.labels_final_train = data_prefix+final_train_suffix

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False, report_every=10000, end=30000, only_tags=False)
        Y = CsvCleaner(self.raw_data_file, detector_mode=False, report_every=10000, end=30000, only_tags=True)
        return X, Y

    def get_preprocessors(self):
        if self.stage > 0:
            return joblib.load(self.data_preprocessor), joblib.load(self.labels_preprocessor)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        cv = CountVectorizer(tokenizer=string.split)
        X, Y = self.get_raw_data()
        tfidf.fit(X)
        cv.fit(Y)
        joblib.dump(tfidf, self.data_preprocessor)
        joblib.dump(cv, self.labels_preprocessor)
        return tfidf, cv

    def get_data(self, tfidf, cv):
        if self.stage > 1:
            return cPickle.loads(self.data_processed), cPickle.loads(self.labels_processed)
        X, Y = self.get_raw_data()
        transformed_X = tfidf.transform(X)
        transformed_Y = cv.transform(Y)
        f = open(self.data_processed,'wb')
        cPickle.dump(transformed_X,f,-1)
        f.close()
        f = open(self.labels_processed,'wb')
        cPickle.dump(transformed_Y,f,-1)
        f.close()
        return transformed_X, transformed_Y

    def run(self):
        tfidf, cv = self.get_preprocessors()
        X, Y = self.get_data(tfidf, cv)
        X, Z = self.get_final_training_data()
        print X
        print Y


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()


