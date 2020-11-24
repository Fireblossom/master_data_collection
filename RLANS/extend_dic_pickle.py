# coding: utf-8
import argparse
import os
import data
import pickle

parser = argparse.ArgumentParser(description='Extend dic')
parser.add_argument('--data', type=str, default=os.getcwd()+'/ag_news_csv/',
                    help='location of the data corpus')
parser.add_argument('--embedding', type=str, default='wiki-news-300d-1M.vec',
                    help='location of the data corpus')
args = parser.parse_args()

dic_exists = os.path.isfile(os.path.join(args.data, 'action_dictionary.pkl'))
print(dic_exists)
if dic_exists:
    with open(os.path.join(args.data, 'action_dictionary.pkl'), 'rb') as input:
        Corpus_Dic = pickle.load(input)
else:
    Corpus_Dic = data.Dictionary()
print(len(Corpus_Dic))

with open(args.embedding, encoding='utf-8') as f:
    f.readline()
    for line in f:
        word, *vector = line.split()
        Corpus_Dic.add_word(word)
print(len(Corpus_Dic))

with open(os.path.join(args.data, 'action_dictionary_full.pkl'), 'wb') as output:
    pickle.dump(Corpus_Dic, output, pickle.HIGHEST_PROTOCOL)
    print("load data and save the dictionary to '{}'".
            format(os.path.join(args.data, 'action_dictionary_full.pkl')))
            