import re
import html
from nltk.corpus import stopwords
remove_list = set(stopwords.words('english'))
import string
remove_list = remove_list.union(set(string.punctuation))
remove_list = remove_list.union({'“', '”', ' ', '-', "'m", "n't", "'s", "'ll", "'ve", "'re"})
import spacy
from spacy.symbols import ORTH
import pickle
import emoji
from data_collection.py_isear.isear_loader_spacy import IsearDataSet
import json

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer
tokenizer.add_special_case("[smile]", [{ORTH: "[smile]"}])
tokenizer.add_special_case("[sad]", [{ORTH: "[sad]"}])
tokenizer.add_special_case("[laugh]", [{ORTH: "[laugh]"}])
tokenizer.add_special_case("[neutral]", [{ORTH: "[neutral]"}])
tokenizer.add_special_case("[username]", [{ORTH: "[username]"}])


def cleaning(tokenize, text_list):
        """
            Cleaning the text
        """
        clean_text_list = []
        for text in text_list:
            text = re.sub(r"@[a-zA-Z0-9\_]*", '[username]', text)
            text = re.sub(r"((\^_+\^)|([:;]-?[\)pP])|(\(:))", '[smile]', text)
            text = re.sub(r"((:'?\()|(:/))", '[sad]', text)
            text = re.sub(r"((XD)|(:D)|(xd))", '[laugh]', text)
            text = re.sub(r"-_+-", '[neutral]', text)
            text = re.sub(r"(ha){2,}a*h*", 'ha', text)
            text = re.sub(r"\.{2,}", '..', text)
            text = re.sub(r"\ {2,}", ' ', text)
            text = re.sub(r"https:\/\/t\.co\/[a-zA-Z0-9]*", '', text)
            text = emoji.demojize(text)
            text = text.lower()
            if tokenize:
                text = tokenizer(text)
                text = [w.text for w in text if w.text not in remove_list]
            # print(text)
            if len(clean_text_list) >= 1:
                if text != clean_text_list[-1]:
                    clean_text_list.append(text)
            else:
                clean_text_list.append(text)
        return clean_text_list


class StoryLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

    def load_story(self, s_tec_path, isear_dataset, tokenize=True):
        f_tec = json.load(open(s_tec_path))
        text_data = []
        for line in f_tec:
            text_data.append(line)
            # print(splited[2])
        f_tec.close()
        if tokenize:
            text_data = cleaning(tokenize, text_data)
        target = [-1] * len(text_data)
        isear_dataset.set_data(isear_dataset.get_data() + text_data)
        isear_dataset.set_target(isear_dataset.get_target() + target)


class TwitterLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

    def load_twitter(self, s_tec_path, tec_dataset, tokenize=True):
        f_tec = open(s_tec_path, 'r', encoding="utf8")
        text_data = []
        for line in f_tec:
            text_data.append(line.replace('\n', ''))
            # print(splited[2])
        f_tec.close()
        if tokenize:
            text_data = cleaning(tokenize, text_data)
        target = [-1] * len(text_data)
        tec_dataset.set_data(tec_dataset.get_data() + text_data)
        tec_dataset.set_target(tec_dataset.get_target() + target)


class BlogLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

    def load_twitter(self, s_tec_path, tec_dataset, tokenize=True):
        f_tec = open(s_tec_path, 'r', encoding="utf8")
        text_data = []
        for line in f_tec:
            text_data.append(line.replace('\n', ''))
        f_tec.close()
        if tokenize:
            text_data = cleaning(tokenize, text_data)
        target = [-1] * len(text_data)
        tec_dataset.set_data(tec_dataset.get_data() + text_data)
        tec_dataset.set_target(tec_dataset.get_target() + target)


class EmoeventLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

    def load_emoevent(self, s_tec_path, isear_dataset, tokenize=True):
        f_tec = open(s_tec_path, 'r', encoding="utf8")
        text_data = []
        target = []
        for line in f_tec:
            splited = html.unescape(line).split('\t')
            text_data.append(splited[1])
            # print(splited[1])
            target.append(-1)
        f_tec.close()
        if tokenize:
            # print(text_data)
            text_data = cleaning(tokenize, text_data)
            # print(text_data)
        print('original dataset size:', len(isear_dataset.get_data()), '\nadditional dataset size:', len(text_data))
        isear_dataset.set_data(isear_dataset.get_data() + text_data)
        isear_dataset.set_target(isear_dataset.get_target() + target)


if __name__ == "__main__":
    loader = TecLoader(True)
    # print(remove_list)
    dataset = loader.load_tec('./tec.txt')
    print(dataset.get_data())