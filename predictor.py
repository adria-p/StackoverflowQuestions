import csv
from brummlearn.glm import GeneralizedLinearSparseModel
from csvCleaner import CsvCleaner
from trainer import Dataset
import numpy as np
import time

__author__ = 'apuigdom'


def predict_tags(data_to_predict, csv_writer, words, models):
    predictions = np.array([m.predict(data_to_predict)[0][0] for m in models])
    selected_tags = " ".join(words[predictions > 0.5])
    csv_writer.writerow([selected_tags])


class TestDataset(Dataset):
    def __init__(self, raw_data_file="Test.csv", preprocessors=None, class_num=10):
        super(TestDataset, self).__init__(raw_data_file=raw_data_file, end=-1,
                                          calculate_preprocessors=False,
                                          preprocessors=preprocessors, class_num=class_num)
        tags = np.array(self.cv.get_feature_names())
        self.words = tags[self.inverse_map.argsort()]
	print self.words
	np.save("words", self.words)

    def get_raw_data(self):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=False, testing=True,
                       repeated_test=np.load("repeated_test.npy"))
        return X

    def __iter__(self):
        TX = self.get_raw_data()
        offset = len(self.tfidf.vocabulary_)
        for tx in TX:
            x = self.tfidf.transform([tx])
            yield x



def generate_model(class_num):
    optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': False} #0.01
    feature_size = len(testing_dataset.tfidf.vocabulary_)
    num_examples = 200
    batch_size = num_examples
    max_iter = 3000
    m = GeneralizedLinearSparseModel(feature_size, 1, out_transfer='sigmoid', loss='fmeasure',
                                     optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
                                     num_examples=num_examples)
    weight_decay = ((m.parameters.in_to_out ** 2).sum())# + (m.parameters.bias**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = 0.001
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay
    parameters_file = "class"+str(class_num)+".npy"
    m.parameters.data = np.load(parameters_file)
    return m

if __name__ == "__main__":
    classes_to_choose = 50
    actual_time = time.time()
    testing_dataset = TestDataset()
    new_time = time.time()
    print "Time spent in building the tfidf and cv: "+str(new_time-actual_time)
    models = [generate_model(class_num) for class_num in range(classes_to_choose)]
    actual_time = time.time()
    new_time = time.time()
    print "Time spent in transforming the training dataset: "+str(new_time-actual_time)
    actual_time = new_time
    new_time = time.time()
    print "Time spent in transforming the validation dataset: "+str(new_time-actual_time)
    csv_submission = csv.writer(open("submission-1.csv", "w"), quoting=csv.QUOTE_NONNUMERIC)
    csv_submission.writerow(["Tags"])
    for x in testing_dataset:
        predict_tags(x, csv_submission, testing_dataset.words, models)
