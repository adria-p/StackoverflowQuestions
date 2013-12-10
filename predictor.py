from csvCleaner import CsvCleaner
from logisiticPredictor import LogisticPredictor
from trainer import Dataset
import numpy as np
import time

__author__ = 'apuigdom'

class TestDataset(Dataset):
    def __init__(self, raw_data_file="Test.csv", preprocessors=None):
        super(TestDataset, self).__init__(raw_data_file=raw_data_file, end=-1,
                                          calculate_preprocessors=False, preprocessors=preprocessors)

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=False, testing=True)
        return X
    def __iter__(self):
        TX = self.get_raw_data()
        offset = len(self.tfidf.vocabulary_)
        num_tags = len(self.cv.vocabulary_)
        for tx in TX:

            x = self.tfidf.transform([tx])
            labels = np.arange(offset, offset+num_tags)
            data_to_add = np.concatenate((x.data, [1]), axis=0)
            new_data = np.repeat(data_to_add, len(labels))
            indptr_to_add = x.indptr[-1]+1
            new_indptr = np.arange((1+len(labels))*indptr_to_add, step=indptr_to_add)
            labels = labels.reshape((len(labels), 1))
            new_indices = np.repeat(x.indices.reshape((1, len(x.indices))), len(labels), axis=0)
            new_indices = np.concatenate((new_indices, labels), axis=1)
            new_indices = new_indices.flatten()
            yield (new_data, new_indices, new_indptr)


if __name__ == "__main__":
    actual_time = time.time()
    testing_dataset = TestDataset()
    new_time = time.time()
    print "Time spent in building the tfidf and cv: "+str(new_time-actual_time)
    lt = LogisticPredictor(testing_dataset, len(testing_dataset.tfidf.vocabulary_) + len(testing_dataset.cv.vocabulary_),
                                            len(testing_dataset.cv.vocabulary_))
    lt.run()
