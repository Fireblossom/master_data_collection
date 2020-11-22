import re
from nltk.corpus import stopwords
import pickle
from data_collection.py_isear import enums
import emoji

remove_list = set(stopwords.words('english'))
import string
remove_list = remove_list.union(set(string.punctuation))
remove_list = remove_list.union({'“', '”', ' ', '—', "'m", "n't", "'s", "'ll", "'ve", "'re", 'á'})
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer


class IsearSubset:
    def __init__(self,
                 labels,
                 values):
        self.labels = labels
        self.values = values


class IsearDataSet:

    def __init__(self,
                 data=IsearSubset([], []),
                 target=IsearSubset([], []),
                 text_data=[],
                 tokenize=True,
                 level=0):
        self.__data = data
        self.__target = target
        self.tokenize = tokenize
        self.level = level
        self.__text_data = self.__cleaning(text_data)
        assert len(data.values) == len(target.values) == len(text_data)

    def __cleaning(self, text_list):
        """
            Cleaning the text
        """
        clean_text_list = []
        for text in text_list:
            text = text.replace('谩 ', '')
            text = text.replace('谩', '')
            text = text.replace('á ', '')
            text = text.replace('á', '')
            text = re.sub(r"\ {2,}", ' ', text)
            text = emoji.demojize(text)
            if re.findall(r'\[.*((No)|(not)|(Never)).*\]', text):
                text = ''
            elif re.findall(r'\[.*((same)|(Same)).*\]', text):
                clean_text_list.append(clean_text_list[-1])
                continue
            text = text.replace('[', '')
            text = text.replace(']', '')
            text.lower()
            if self.tokenize:
                text = tokenizer(text)
                text = [w.text for w in text if w.text not in remove_list]
                if self.level != 0:
                    vocab = pickle.load(open('RLANS/data_collection/py_isear/vocab'+str(self.level)+'.pkl', 'rb'))
                    text = [w for w in text if w in vocab]
            #print(text)

            clean_text_list.append(text)
        return clean_text_list

    def get_freetext_content(self):
        return self.__data.values

    def get_target(self):
        return self.__target.values

    def get_data_label_at(self, i):
        return self.__data.labels[i]

    def get_target_label_at(self, i):
        return self.__target.labels[i]

    def get_data(self):
        return self.__text_data

    def set_data(self, data):
        self.__text_data = data

    def set_target(self, target):
        self.__target = IsearSubset(target, target)


class NoSuchFieldException(BaseException):

    def __init__(self, field_name):
        self.message = "No such field in dataset : " + field_name

    def get_message(self):
        return self.message


class IsearLoader:
    def __init__(self,
                 attribute_list=[],
                 target_list=[],
                 provide_text=True,
                 tokenize=True):
        # list of attributes to extract, refer to enums.py
        self.attribute_list = []
        self.set_attribute_list(attribute_list)
        # list of targets to extract
        self.target_list = []
        self.set_target_list(target_list)
        # provide the text, true by default
        self.provide_text = provide_text
        self.tokenize = tokenize

    def load_isear(self, s_isear_path, level):
        f_isear = open(s_isear_path, "r")
        '''
        The isear file extracted for the purpose of this initial
        loading is a pipe delimited csv-like file with headings
        '''

        i = 0
        entry_attributes = []
        text_data = []
        entry_target = []
        for isear_line in f_isear:
            isear_line.replace('谩|', '')
            isear_row = isear_line.split('|')[:-1]
            if i == 0:
                i = i + 1
                continue
            # print(isear_row)
            result = self.__parse_entry(isear_row,
                                        i,
                                        text_data)
            entry_attributes.append(result["attributes"])
            entry_target.append(result["target"])
            i = i + 1
        attributes_subset = IsearSubset(self.attribute_list,
                                        entry_attributes)
        target_subset = IsearSubset(self.target_list,
                                    entry_target)
        f_isear.close()
        return IsearDataSet(attributes_subset,
                            target_subset,
                            text_data,
                            self.tokenize,
                            level)

    def __parse_entry(self,
                      isear_row,  # The row of the entry
                      index,  # row number
                      text_data):  # the text data
        i_col = 0
        l_attributes = []
        l_target = []
        # start parsing the columns
        for isear_col in isear_row:
            # print(isear_col)
            # we need to know to which field we are refering
            # handling the excess columns
            if i_col >= len(enums.CONST_ISEAR_CODES):
                break

            s_cur_col = enums.CONST_ISEAR_CODES[i_col]

            # for further test this will tell whether we are in the SIT column,
            # which is a text column
            b_is_sit = bool(s_cur_col == "SIT")
            if b_is_sit:
                if self.provide_text:
                    # should be clear enough
                    text_data.append(isear_col)
            else:
                # should be an int

                if s_cur_col in self.attribute_list:
                    i_isear_col = int(isear_col)
                    l_attributes.append(i_isear_col)

                if s_cur_col in self.target_list:
                    i_isear_col = int(isear_col)
                    l_target.append(i_isear_col)
            # next column
            i_col = i_col + 1
        # we will return a pretty "free form" object
        return {"attributes": l_attributes,
                "target": l_target}

    # compares attribute existence in the Isear labels
    def __check_attr_exists(self, attribute):
        return attribute in enums.CONST_ISEAR_CODES

    def set_attribute_list(self, attrs):
        """Set a list of attributes to extract

        Args:
        attrs (list):  a list of strings refering Isear fields .

        Returns:
        self. in order to ease fluent programming (loader.set().set())
        Raises:
        NoSuchFieldException

        """
        self.attribute_list = []
        for attr in attrs:
            self.add_attribute(attr)
        return self

    def set_target_list(self, target):
        """Set a list of fields to extract as target
        Args:
        attrs (list):  a list of strings refering Isear fields .

        Returns:
        self. in order to ease fluent programming (loader.set().set())
        Raises:
        NoSuchFieldException

        """
        self.target_list = []
        for tgt in target:
            self.add_target(tgt)
        return self

    def set_provide_text(self, is_provide_text):
        """ Tell the extractor whether to load the free text field.
        Behaviour is true by default

        Args:
        is_provide_text (bool): whether to provide the text field or not
        Return
        self. For fluent API
        """
        self.provide_text = is_provide_text
        return self

    def add_attribute(self, attr):
        b_att_ex = self.__check_attr_exists(attr)
        if b_att_ex is not True:
            ex = NoSuchFieldException(attr)
            raise ex
        self.attribute_list.append(attr)
        return self

    def add_target(self, attr):
        b_att_ex = self.__check_attr_exists(attr)
        if b_att_ex is not True:
            ex = NoSuchFieldException(attr)
            raise ex
        self.target_list.append(attr)
        return self

    # def load_isear(self):


if __name__ == '__main__':
    attributes = ['SEX', 'CITY']
    target = ['EMOT']
    loader = IsearLoader(attributes, target, True)
    dataset = loader.load_isear('../isear.csv')
    print(dataset.get_data())
