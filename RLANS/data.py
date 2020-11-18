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
import json


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

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
    def __init__(self, tec_or_isear):
        self.file = tec_or_isear
        self.tokens = []  # used to store all text information
        self.labels = []  # used to store the label information
        self.length = 0
        self.max_length = 0

    def load(self, dictionary=None, train_mode=True, tokenize=True, level=0):
        if self.file == 'isear':
            attributes = []
            target = ['EMOT']
            loader = IsearLoader(attributes, target, True, tokenize)
            if train_mode is True:
                data = loader.load_isear('isear_train.csv', level)
            else:
                data = loader.load_isear('isear_test.csv', level)
            if train_mode == 'lm':
                from convokit import Corpus, download
                corpus = Corpus(filename=download("friends-corpus"))
                text = []
                for utt in corpus.iter_utterances():
                    text.append(utt.text)
                # text = json.load(open('RLANS/data_collection/scrape/story.json'))
                t = IsearSubset([1]*len(text), [1]*len(text))
                data = IsearDataSet(t, t, text, tokenize)
                train_mode = True
            '''attributes = []
            target = ['EMOT']
            loader = IsearLoader(attributes, target, True)
            data = loader.load_isear('RLANS/data_collection/isear.csv')
            if tokenize is False: 
                data.set_data(json.load(open('isear_raw_data.json')))'''
                #print(data.get_data())
        elif self.file == 'tec':
            loader = TecLoader(tokenize, augment=True)
            if train_mode is True:
                data = loader.load_tec('tec_train.txt', level)
            else:
                data = loader.load_tec('tec_test.txt', level)
            # data.text_data = json.load(open('tec_data_da.json'))
            # data.target = json.load(open('tec_target_da.json'))

        for label, text in zip(data.get_target(), data.get_data()):
            #print(label)
            #print(label, text)
            # get actions
            if tokenize is False:
                txt = split_by_punct(text) + ['<eos>']
            else:
                txt = text + ['<eos>']
            #print(label, txt)
            token = []
            for word in txt:
                # Add words to the dictionary in train_mode
                if train_mode is True:
                    dictionary.add_word(word)
                    # Tokenize file content
                    token.append(dictionary.word2idx[word])
                else:
                    if word in dictionary.word2idx:
                        token.append(dictionary.word2idx[word])
            # get id
            if type(label) == list:
                label = label[0]
            self.labels.append(label-1)
            self.tokens.append(token)
            if len(token) > self.max_length: self.max_length = len(token)
        #print(self.labels)
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