__author__ = 'apuigdom'
import csv
import re

class CsvCleaner:
    def __init__(self, csv_file):
        self.csv = csv_file
        self.p = re.compile("<p(?:\s.*?)*?>(.*?)</p>", re.S)
        self.pre = re.compile("<pre(?:\s.*?)*?>(.*?)</pre>", re.S)
        self.code = re.compile("<code>(.*?)</code>", re.S)
        self.li = re.compile("<li(?:\s.*?)*?>(.*?)</li>", re.S)
        self.sentence = re.compile("(.*[\.])*?[\.!?]")
        self.any_tag = re.compile("</(.*?)>")
        self.tags_found = []
        self.tags_to_remove = re.compile("<.*?>")
        #self.tags = ['p', 'code', 'pre', 'a', 'li', 'ol', 'em', 'strong', 'blockquote']

        #self.tags_to_remove = ['h1', 'ul', 'h2', 'b', 'h3', 'strike', 'i', 'kbd', 'PRE', 'sup']
        #1996 p, 55 code
        #self.compiled_tags = [re.compile("<%s(?:\s.*?)*?>(.*?)</%s>" % (tag, tag)) for tag in self.tags]
        #self.compiled_tags_remove = [re.compile("<%s(?:\s.*?)*?>(.*?)</%s>" % (tag, tag)) for tag in self.tags_to_remove]

    def get_final_string(self, string_list):
        final_string = " ".join(string_list)
        final_string = re.sub(self.tags_to_remove,"", final_string)
        return final_string

    def process(self, text):
        """
            tags = self.any_tag.findall(text)
            for tag in tags:
                if not tag in self.tags_found:
                    self.tags_found.append(tag)
        """
        p_list = self.p.findall(text)
        new_text = re.sub(self.p,"", text)
        li_list = self.li.findall(new_text)
        new_text = re.sub(self.li, "", new_text)
        code_list = self.code.findall(new_text)
        new_text = re.sub(self.code, "", new_text)
        pre_list = self.pre.findall(new_text)
        code_string = " ".join(code_list)
        pre_string = " ".join(pre_list)
        li_string = self.get_final_string(li_list)
        p_string = self.get_final_string(p_list)
        return [code_string, p_string, p_string+code_string+li_string+pre_string, li_string, text]

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

    def clean(self, start=None, end=None):
        csv_file = open(self.csv)
        csv_reader = csv.reader(csv_file)
        start_reading = start if start is not None else 0
        end_reading = end if end is not None else -1
        current_line = 0

        a = [0,0,0,0,0,0,0,0]
        b = [0,0,0]
        for line in csv_reader:
            current_line += 1
            if current_line < start_reading:
                continue
            if current_line > end_reading:
                if end_reading != -1:
                    break
            [q_id, title, body, tags] = line
            tags = tags.split(" ")
            a[-1] += len(tags)
            body = body.lower()
            processed_body = self.process(body)
            b[0] += len(processed_body[1])
            b[1] += len(processed_body[2])
            b[2] += len(body)
            [code_list, p_list, complete, li, text] = processed_body
            for index, element in enumerate(processed_body):
                for tag in tags:
                    if self.tag_detected(tag,element):
                        a[index+1] += 1

            for tag in tags:
                if self.tag_detected(tag,processed_body[0]) and not self.tag_detected(tag,processed_body[1]):
                    print tag

            for tag in tags:
                if self.tag_detected(tag, text) and not self.tag_detected(tag, complete):
                    print text
                    print complete
                    print tag

            for tag in tags:
                if self.tag_detected(tag, title):
                    a[0] += 1
                if self.tag_detected(tag,title+processed_body[-1]):
                    a[-2] += 1
        print a
        print b
if __name__ == "__main__":
    cc = CsvCleaner("Train.csv")
    cc.clean(end=40000)
