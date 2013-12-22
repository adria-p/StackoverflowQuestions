from sklearn.externals import joblib
import numpy as np
import os
from preprocessing.csvCleaner import CsvCleaner


__author__ = 'kosklain'


class DistributionCounter(object):
    """
        Gets all the indices of the tags.
    """
    def __init__(self, preprocessor_suffix="preprocess.pkl",
                 raw_data_file="Train_clean2.csv", start=0,
                 end=3500000):
        self.start = start
        self.end = end
        labels_prefix = "labels_"
        self.labels_preprocessor = labels_prefix+preprocessor_suffix
        self.raw_data_file = raw_data_file
        self.cv = joblib.load(self.labels_preprocessor)

    def get_raw_data(self):
        Y = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=10000, start=self.start,
                       end=self.end, only_tags=True)
        return Y

    def __iter__(self):
        Y = self.get_raw_data()
        for ty in Y:
            y = self.cv.transform([ty])
            yield y.indices


if __name__ == "__main__":
    models_folder = 'models'
    dc = DistributionCounter()
    final_array = [i for x in dc for i in x]
    final_array = np.array(final_array)
    final_array = np.sort(final_array)
    np.save(os.path.join(models_folder, "distribution"), final_array)
    dist_inverse = [0]
    current = final_array[0]
    for i, x in enumerate(final_array):
        if x != current:
            current = x
            dist_inverse.append(i)
    dist_inverse.append(len(final_array))
    dist_inverse = np.array(dist_inverse)
    np.save(os.path.join(models_folder, "distribution_inverse"), dist_inverse)
    dist_diff = np.diff(dist_inverse)
    sorted_diff = np.argsort(dist_diff)
    sorted_diff = sorted_diff[::-1]
    final_map = np.argsort(sorted_diff)
    np.save(os.path.join(models_folder, "inverse_map"), final_map)