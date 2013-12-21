from collections import Counter
import os
from preprocessing.csvCleaner import CsvCleaner

__author__ = 'apuigdom'


class StatCounter(object):
    def __init__(self, raw_data_file):
        self.raw_data_file = raw_data_file

    def get_raw_data(self, start, end):
        X = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=100000, start=start,
                       end=end, only_tags=False)
        Y = CsvCleaner(self.raw_data_file, detector_mode=False,
                       report_every=100000, start=start,
                       end=end, only_tags=True)
        return X, Y


    def check_stats(self, bins, training=True):
        number_to_check = 6000000 if training else 2000000
        nums_per_bin = number_to_check/bins
        checking = self.check_training_stats if training else self.check_testing_stats
        for i in range(0, number_to_check, nums_per_bin):
            checking(i, i+nums_per_bin)


    def check_training_stats(self, start, end):
        X, Y = self.get_raw_data(start, end)
        words_x = [word for x in X for word in x.split()]
        print "Result for training text from %d to %d:" % (start, end)
        print Counter(words_x).most_common(1000)
        words_y = [word for y in Y for word in y.split()]
        print "Result for training tags from %d to %d:" % (start, end)
        print Counter(words_y).most_common(1000)

    def check_testing_stats(self, start, end):
        X, _ = self.get_raw_data(start, end)
        words_x = [word for x in X for word in x.split()]
        print "Result for testing text from %d to %d:" % (start, end)

if __name__ == "__main__":
    sc = StatCounter(os.path.join('data', 'Test.csv'))
    sc.check_stats(5, training=False)
