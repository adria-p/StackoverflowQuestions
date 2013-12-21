__author__ = 'apuigdom'
import csv
import numpy as np
import os

data_folder = 'data'
models_folder = 'models'
submission = os.path.join(data_folder, 'submission.csv')
presubmission = os.path.join(data_folder, 'presubmission.csv')
original_train = os.path.join(data_folder, 'Train.csv')


csv_writer = csv.writer(open(submission, 'w'), quoting=csv.QUOTE_NONNUMERIC)
csv_reader_submission = csv.reader(open(presubmission))
csv_reader_train = csv.reader(open(original_train))
next(csv_reader_train)
next(csv_reader_submission)
csv_writer.writerow(["Id", "Tags"])

repeated_train = np.load(os.path.join(models_folder, "repeated_train.npy"))
repeated_test = np.load(os.path.join(models_folder, "repeated_test.npy"))
test_index = 0
answer_num = 0
offset = 6034196
print "Reading train tags"
train_tags = [line[3] for line in csv_reader_train]
print "Read train tags"
while True:
    if test_index != len(repeated_test) and repeated_test[test_index] == answer_num:
        csv_writer.writerow([offset+answer_num, train_tags[repeated_train[test_index]]])
        test_index += 1
    else:
        csv_writer.writerow([offset+answer_num] + next(csv_reader_submission))
    answer_num += 1
