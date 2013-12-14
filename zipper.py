__author__ = 'apuigdom'
import csv
import numpy as np


csv_writer = csv.writer(open("submission2.csv", "w"), quoting=csv.QUOTE_NONNUMERIC)
csv_reader_submission = csv.reader(open("submission.csv"))
csv_reader_train = csv.reader(open("Train.csv"))
next(csv_reader_train)
next(csv_reader_submission)
csv_writer.writerow(["Id", "Tags"])

repeated_train = np.load("repeated_train.npy")
repeated_test = np.load("repeated_test.npy")
test_index = 0
answer_num = 0
offset = 6034196

train_tags = [line[3] for line in csv_reader_train]

while True:
    if test_index != len(repeated_test) and repeated_test[test_index] == answer_num:
        csv_writer.writerow([offset+answer_num, train_tags[repeated_train[test_index]]])
        test_index += 1
    else:
        csv_writer.writerow([offset+answer_num, next(csv_reader_submission)])
    answer_num += 1