__author__ = 'apuigdom'
import csv
import re
from multiprocessing import Pool
import numpy as np

class CsvCleaner:
    def __init__(self, csv_file, start=0, end=-1, detector_mode=False, report_every=0, only_tags=False):
        self.csv = csv_file
        self.only_tags=only_tags
        self.report_every = report_every
        self.detector_mode = detector_mode
        self.start_reading = start if start is not None else 0
        self.end_reading = end if end is not None else -1
        self.p = re.compile("<p(?:\s.*?)*?>(.*?)</p>", re.S)
        self.pre = re.compile("<pre(?:\s.*?)*?>(.*?)</pre>", re.S)
        self.code = re.compile("<code>(.*?)</code>", re.S)
        self.li = re.compile("<li(?:\s.*?)*?>(.*?)</li>", re.S)
        self.stats = [0,0,0,0,0,0]
        self.characters = [0,0,0]
        self.tags_found = []
        self.tags_to_remove = [(re.compile("&lt;"), "<"), (re.compile("&gt;"), ">"), (re.compile("<.*?>"), ""),
                               (re.compile("\[.*?\]"), ""), #(re.compile("\(.*?\)"), ""),
                               (re.compile("\{.*?\}"), ""), (re.compile("[^a-zA-Z0-9 ]"), " "),
                               (re.compile(" [0-9 ]* "), self.spell_numbers)]

        self.number_spelling = [(re.compile("1"), "one "), (re.compile("2"), "two "),
                               (re.compile("3"), "three "), (re.compile("4"), "four "),
                               (re.compile("5"), "five "), (re.compile("6"), "six "),
                               (re.compile("7"), "seven "), (re.compile("8"), "eight "),
                               (re.compile("9"), "nine "), (re.compile("0"), "zero ")]

    def spell_numbers(self, matchobj):
        final_obj = matchobj.group(0)
        for number_to_spell in self.number_spelling:
            final_obj = re.sub(number_to_spell[0], number_to_spell[1], final_obj)
        return final_obj

    def get_final_string(self, string_list, trim=-1):
        if isinstance(string_list, list):
            string_list_modified = [element for element in string_list if element != '' and element != ' ']
            final_string = " ".join(string_list_modified)
            final_string = final_string.strip(' \t\n\r')
        else:
            final_string = string_list.strip(' \t\n\r')
        for remove_re in self.tags_to_remove:
            final_string = re.sub(remove_re[0], remove_re[1], final_string)

        if trim != -1:
            final_string = final_string.split()
            if len(final_string) > 2*trim:
                final_string = final_string[:trim]+ final_string[-trim:]
            final_string = " ".join(final_string)
        return final_string



    def process(self, text):
        p_list = self.p.findall(text)
        new_text = re.sub(self.p,"", text)
        li_list = self.li.findall(new_text)
        new_text = re.sub(self.li, "", new_text)
        code_list = self.code.findall(new_text)
        new_text = re.sub(self.code, "", new_text)
        pre_list = self.pre.findall(new_text)
        code_string = self.get_final_string(code_list, 20)
        pre_string = self.get_final_string(pre_list)
        li_string = self.get_final_string(li_list)
        p_string = self.get_final_string(p_list)
        final_string = p_string+" "+code_string+" "+li_string+" "+pre_string
        return final_string

    def tag_detected(self, tag, text):
        if tag in text:
            return True
        if "-" in tag:
            tags_contained = tag.split("-")
            for sub_tag in tags_contained:
                if not sub_tag in text:
                    return False
            return True
        return False


    def detect(self, title, body, processed_body, tags):
        self.stats[-1] += len(tags)
        self.characters[0] += len(processed_body)
        self.characters[1] += len(body)
        for tag in tags:
            if self.tag_detected(tag, body):
                self.stats[2] += 1
                if self.tag_detected(tag, processed_body):
                    self.stats[1] += 1

        for tag in tags:
            if self.tag_detected(tag, title):
                self.stats[0] += 1
            if self.tag_detected(tag, title+processed_body):
                self.stats[-2] += 1
            if self.tag_detected(tag, title+body):
                self.stats[-3] += 1
        print self.stats
        print self.characters

    def __iter__(self):
        csv_file = open(self.csv)
        csv_reader = csv.reader(csv_file)
        current_line = 0
        for line in csv_reader:
            current_line += 1
            if current_line == 1 or current_line < self.start_reading+1:
                continue
            if self.report_every != 0 and current_line % self.report_every == 0:
                print "Round: %d" % current_line
            if current_line > self.end_reading:
                if self.end_reading != -1:
                    break
            [_, title, body, tags] = line
            body = body.lower()
            title = self.get_final_string(title.lower())
            if not self.only_tags:
                processed_body = self.process(body)
            if self.detector_mode:
                self.detect(title, body, processed_body, tags.split(" "))
            if not self.only_tags:
                yield " ".join((title+processed_body).split())
            else:
                yield tags


class LineCounter(object):
    def __init__(self, csv_file):
        self.csv = csv_file
    def count_lines(self):
        csv_file = open(self.csv)
        csv_reader = csv.reader(csv_file)
        counter = 0
        for _ in csv_reader:
            counter += 1
        return counter

class RepeatedTitlesCounter(object):
    def __init__(self, csv_train, csv_test):
        self.csv_train = csv_train
        self.csv_test = csv_test

    def count_repeated_titles(self):
        print "Starting to store titles"
        csv_reader_train = csv.reader(open(self.csv_train))
        next(csv_reader_train)
        train_titles = [line[1] for line in csv_reader_train]
        print "Done with titles in training"
        csv_reader_test = csv.reader(open(self.csv_test))
        next(csv_reader_test)
        test_titles = [(i, line[1]) for i, line in enumerate(csv_reader_test)]
        def is_repeated(title):
            if title[0] % 10000:
                print "Round %d" % title[0]
            if title[1] in train_titles:
                return title
            return None
        pool = Pool(processes=4)
        result = pool.map(is_repeated(), test_titles)
        final_index_array = np.array([element[0] for element in result if element != None])
        print "Done, %d out of %d" % (len(final_index_array), len(result))
        np.save("repeated_test", final_index_array)


if __name__ == "__main__":
    lc = RepeatedTitlesCounter("Train.csv", "Test.csv")
    print lc.count_repeated_titles()

