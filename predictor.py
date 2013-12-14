import csv
from multiprocessing import Pool
from brummlearn.glm import GeneralizedLinearSparseModel
from scipy.sparse import csr_matrix
from csvCleaner import CsvCleaner
from trainer import Dataset
import numpy as np
import time

__author__ = 'apuigdom'


class TestDataset(Dataset):
    def __init__(self, raw_data_file="Test.csv", preprocessors=None):
        super(TestDataset, self).__init__(raw_data_file=raw_data_file, end=-1,
                                          calculate_preprocessors=False,
                                          preprocessors=preprocessors)

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=False, testing=True,
                       repeated_test=np.load("repeated_test.npy"))
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


def predict_tags(data_to_predict):
    data, indices, indptr = data_to_predict
    TX = csr_matrix((data, indices, indptr),
                shape=(num_tags, feature_size),
                dtype=np.float64)
    predictions = np.array(m.predict(TX)).flatten()
    selected_tags = tags[predictions > 0.5]
    selected_tags = " ".join(selected_tags)
    return [selected_tags]

if __name__ == "__main__":
    actual_time = time.time()
    testing_dataset = TestDataset()
    new_time = time.time()
    t = [x for x in testing_dataset]
    print "Time spent in building the tfidf and cv: "+str(new_time-actual_time)

    feature_size = len(testing_dataset.tfidf.vocabulary_) + len(testing_dataset.cv.vocabulary_)
    num_tags = len(testing_dataset.cv.vocabulary_)
    parameters_file = "params20131209-223018.npy"

    num_examples = 20
    batch_size = num_examples*num_tags
    max_iter = 3000
    actual_time = time.time()
    new_time = time.time()
    print "Time spent in transforming the training dataset: "+str(new_time-actual_time)
    actual_time = new_time
    new_time = time.time()
    print "Time spent in transforming the validation dataset: "+str(new_time-actual_time)
    optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': False} #0.01

    m = GeneralizedLinearSparseModel(feature_size, 1, out_transfer='sigmoid', loss='fmeasure',
                                     optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
                                     num_examples=num_examples)
    weight_decay = ((m.parameters.in_to_out ** 2).sum())# + (m.parameters.bias**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = 0.001
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    m.parameters.data = np.load(parameters_file)

    csv_submission = csv.writer(open("submission.csv", "w"), quoting=csv.QUOTE_NONNUMERIC)
    csv_submission.writerow(["Tags"])

    tags = np.array(testing_dataset.cv.get_feature_names())

    pool = Pool(processes=4)
    result = pool.map(predict_tags, t)
    #result = [predict_tags(x) for x in testing_dataset]
    csv_submission.writerows(result)
