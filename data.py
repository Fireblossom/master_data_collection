from io import open
import torch
from torch.utils.data import Dataset
import sys
import csv
import numpy as np
import re
csv.field_size_limit(sys.maxsize)
from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet, IsearSubset
from data_collection.py_ssec.ssec_loader_spacy import SsecLoader, SsecDataset
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_additional.py_unlabeled import EmoeventLoader, TwitterLoader, BlogLoader
import json


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<pad>':0}
        self.idx2word = ['<pad>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_dic(self, idx2word):
        self.idx2word = idx2word
        i = 0
        for word in idx2word:
            self.word2idx[word] = i

    def __len__(self):
        return len(self.idx2word)


def split_by_punct(segment):
    """Splits str segment by punctuation, filters our empties and spaces."""
    return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]


class Csv_DataSet(Dataset):
    # this is used to get a csv format of action sequence with id and role
    # the data is like:
    #  id | action sequence | role sequence |
    def __init__(self, csv_file):
        self.file = csv_file
        self.tokens = []  # used to store all text information
        self.labels = []  # used to store the label information
        self.length = 0
        self.max_length = 0

    def load(self, lowercase=True, dictionary=None,train_mode=True):
        with open(self.file) as db_f:
            reader = csv.reader(db_f)
            next(reader)  # skip header
            for idx, row in enumerate(reader):
                # get actions
                content = row[1]+' '+row[2]
                content = content.strip()
                if lowercase:
                    content = content.lower()
                txt = split_by_punct(content) + ['<eos>']
                token = []
                for word in txt:
                    # Add words to the dictionary in train_mode
                    if train_mode:
                        dictionary.add_word(word)
                        # Tokenize file content
                        token.append(dictionary.word2idx[word])
                    else:
                        if word in dictionary.word2idx:
                            token.append(dictionary.word2idx[word])
                # get id
                self.labels.append(int(row[0])-1)
                self.tokens.append(token)
                if len(token) > self.max_length: self.max_length = len(token)
            self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        token_seq = np.array(self.tokens[index], dtype=int)
        is_meaningful = np.ones(len(self.tokens[index])-1)
        label = self.labels[index]
        return token_seq, label, is_meaningful


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples
    (token_seqs, role_seqs, case_ids, texts_seq).
    Seqeuences are padded to the maximum length of
    mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (token_seqs, role_seqs, case_ids, texts_seq).
            - token_seqs: np.array of shape (?); variable length.
            - role_seqs: np.array of shape (?); variable length.
            - case_id: the id of the case
            - texts_seq: List of actions
    Returns:
        token_seqs: np.array of shape (batch_size, padded_length).
        role_seqs: np.array of shape (batch_size, padded_length).
        texts_seq: same as input
        src_lengths: np.array of length (batch_size);
        case_ids: same as input
        pad_length: int length for each padded seq
    """
    def merge(sequences, pad_length):
        lengths = np.array([len(seq) for seq in sequences])
        padded_seqs = np.zeros((len(sequences), pad_length))
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # separate data, sequences
    token_seqs, labels, is_meaningful = zip(*data)

    # get the pad_length
    lengths = [len(seq) for seq in token_seqs]
    pad_length = max(lengths)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    token_seqs, seq_lengths = merge(token_seqs, pad_length)
    importance_seqs, importance_lengths = merge(is_meaningful, pad_length)
    if not (seq_lengths == importance_lengths + 1).all():
        raise ValueError("The length of token sequence is not "
                         "equal to the length of the importance sequence!")
    bitch_size = len(lengths)
    next_token_seqs = np.zeros((bitch_size, pad_length))
    next_token_seqs[:, :-1] = token_seqs[:, 1:]

    return token_seqs.astype(int), next_token_seqs.astype(int), importance_seqs, labels, seq_lengths, pad_length


class SSEC_DataSet(Dataset):
    # this is used to get a csv format of action sequence with id and role
    # the data is like:
    #  id | action sequence | role sequence |
    def __init__(self, ssec_file):
        self.file = ssec_file
        self.tokens = []  # used to store all text information
        self.labels = []  # used to store the label information
        self.length = 0
        self.max_length = 0

    def load(self, dictionary=None,train_mode=True):
        loader = SsecLoader(True)
        data = loader.load_ssec(self.file)
        for label, text in zip(data.get_target(), data.get_data()):
            # get actions
            txt = text + ['<eos>']
            token = []
            for word in txt:
                # Add words to the dictionary in train_mode
                if train_mode:
                    dictionary.add_word(word)
                    # Tokenize file content
                    token.append(dictionary.word2idx[word])
                else:
                    if word in dictionary.word2idx:
                        token.append(dictionary.word2idx[word])
            # get id
            # print(label, token)
            self.labels.append(label)
            self.tokens.append(token)
            if len(token) > self.max_length: self.max_length = len(token)
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        token_seq = np.array(self.tokens[index], dtype=int)
        is_meaningful = np.ones(len(self.tokens[index])-1)
        label = self.labels[index]
        return token_seq, label, is_meaningful


class TEC_ISEAR_DataSet(Dataset):
    # this is used to get a csv format of action sequence with id and role
    # the data is like:
    #  id | action sequence | role sequence |
    def __init__(self, tec_or_isear, number_per_class=False):
        self.file = tec_or_isear
        self.number_per_class = number_per_class
        self.tokens = []  # used to store all text information
        self.labels = []  # used to store the label information
        self.length = 0
        self.max_length = 0

    def load(self, dictionary=None, train_mode=True, tokenize=True, level=0, addition=False, label_select=False):
        if self.file == 'isear':
            attributes = []
            target = ['EMOT']
            loader = IsearLoader(attributes, target, True, tokenize)
            if train_mode is True:
                if not self.number_per_class:
                    data = loader.load_isear('isear_train.csv', level)
                else:
                    print(self.number_per_class)
                    data = loader.load_isear('isear_train_' + self.number_per_class + '.csv', level)
            else:
                data = loader.load_isear('isear_test.csv', level)
            self.split = len(data.get_data())
            if tokenize:
                if addition == 'emoevent':
                    loader = EmoeventLoader(tokenize=tokenize)
                    # print(len(data.get_data()))
                    loader.load_emoevent('dataset_emotions_EN.txt', data)
                if addition == 'self':
                    loader.load_isear('isear_train.csv', level, data)
                elif addition != False:
                    loader = TwitterLoader()
                    loader.load_twitter(addition, data)
                    """text_data = json.load(open('blog_en.json'))
                    data.set_data(data.get_data() + text_data)
                    data.set_target(data.get_target() + ([-1]*len(text_data)))
                    loader = EmoeventLoader(tokenize=tokenize)
                    # print(len(data.get_data()))
                    loader.load_emoevent('data_collection/scrape/blog_en.txt', data)"""           

        elif self.file == 'tec':
            loader = TecLoader(tokenize, augment=False)
            if train_mode is True:
                if not self.number_per_class:
                    data = loader.load_tec('tec_train.txt', level)
                else:
                    data = loader.load_tec('tec_train_' + self.number_per_class + '.txt', level)
            else:
                data = loader.load_tec('tec_test.txt', level)
            self.split = len(data.get_data())
            if tokenize:
                if addition == 'emoevent':
                    loader = EmoeventLoader(tokenize=tokenize)
                    loader.load_emoevent('dataset_emotions_EN.txt', data)
                if addition == 'self':
                    loader.load_tec('tec_train.txt', level, data)
                elif addition != False:
                    # text_data = json.load(open('twitter_data.json'))
                    # data.set_data(data.get_data() + text_data)
                    # data.set_target(data.get_target() + ([-1]*len(text_data)))
                    # print(len(data.get_data()))
                    loader = TwitterLoader()
                    loader.load_twitter(addition, data)
                
            # data.text_data = json.load(open('tec_data_da.json'))
            # data.target = json.load(open('tec_target_da.json'))

        for label, text in zip(data.get_target(), data.get_data()):
            # print(label)
            # print(label, text)
            # get actions
            if tokenize is False:
                txt = split_by_punct(text) + ['<eos>']
            else:
                if len(text) <= 50:
                    txt = text + ['<eos>']
                else:
                    # print(text)
                    txt = text[:50] + ['<eos>']
            #print(label, txt)
            token = []
            for word in txt:
                # Add words to the dictionary in train_mode
                if train_mode is True and label != -1:
                    dictionary.add_word(word)
                    # Tokenize file content
                    token.append(dictionary.word2idx[word])
                else:
                    if word in dictionary.word2idx:
                        token.append(dictionary.word2idx[word])
            # get id
            if type(label) == list:
                label = label[0]

            if label_select == False:
                self.labels.append(label-1)
                self.tokens.append(token)
                if len(token) > self.max_length: self.max_length = len(token)
            elif label_select != False and label in label_select:
                label = label_select.index(label)
                self.labels.append(label)
                self.tokens.append(token)
                if len(token) > self.max_length: self.max_length = len(token)
                self.split = len(self.labels)
            elif label == -1:
                self.labels.append(-1)
                self.tokens.append(token)
                if len(token) > self.max_length: self.max_length = len(token)
            
        # print(self.labels)
        self.length = len(self.labels)
        self.max_length = self.max_length
            

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        token_seq = np.array(self.tokens[index], dtype=int)
        #print(token_seq)
        is_meaningful = np.ones(len(self.tokens[index])-1)
        label = self.labels[index]
        return token_seq, label, is_meaningful