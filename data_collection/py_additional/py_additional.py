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
import json
import pandas as pd

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
            clean_text_list.append(text)
        return clean_text_list


class emoevent_Dataset():
    def __init__(self, tokenize=False):
        self.data, self.target = [], []
        df = pd.read_csv('dataset_emotions_EN.txt', sep='\t')
        self.data = cleaning(tokenize, df['tweet'].to_list())
        # print(cleaning(tokenize, df['tweet'].to_list()))
        self.target = df['emotion'].to_list()        

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


class dailydialog_Dataset():
    def __init__(self, tokenize=False):
        self.data, self.target = [], []
        text = open('datas/dialogues_text.txt')
        target = open('datas/dialogues_emotion.txt')
        for line in text:
            labels = target.readline()
            sentences = line.split('__eou__')
            label = labels.split(' ')
            if len(sentences) == len(label):
                self.data += cleaning(tokenize, sentences[:-1])
                self.target += label[:-1]
        
        print(len(self.data), len(self.target))
        text.close()
        target.close()

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


class goodnewseveryone_Dataset():
    def __init__(self, tokenize=False):
        self.data, self.target = [], []
        with open('datas/gne-release-v1.0.jsonl') as file:
            for line in file:
                dic = json.loads(line)
                self.data += cleaning(tokenize, dic['headline'])
                self.target += dic['annotations']['dominant_emotion']['gold']

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


class crowdflower_Dataset():
    def __init__(self, tokenize=False):
        self.data, self.target = [], []
        df = pd.read_csv('datas/crowdflower.csv')
        self.data = cleaning(tokenize, df['content'].to_list())
        self.target = df['sentiment'].to_list()

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


class EmoInt_Dataset():
    def __init__(self, tokenize=False):
        emotion = ['anger', 'fear', 'joy', 'sadness']
        self.data, self.target = [], []
        i = 0
        for e in emotion:
            with open('datas/'+e+'-ratings-0to1.train.txt') as file:
                text, target = [], []
                for line in file:
                    text.append(line.split('\t')[1])
                    target.append(i)
            i += 1
            self.data += cleaning(tokenize, text)
            self.target += target

    def get_data(self):
        return self.data
    
    def get_target(self):
        return self.target


if __name__ == "__main__":
    dailydialog_Dataset().get_data()
    goodnewseveryone_Dataset().get_data()
    crowdflower_Dataset().get_data()
    EmoInt_Dataset().get_data()