import gensim
from gensim.similarities import MatrixSimilarity
from gensim import corpora, models, similarities
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_additional.py_unlabeled import TwitterLoader
import json
import numpy as np


loader = TecLoader(tokenize=True, augment=False)
data = loader.load_tec('tec_test.txt', level=0)

doc_list = data.get_data()
doc_len = len(doc_list)
print(doc_len)

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]

id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]

model = models.TfidfModel(BoW_corpus)

index = MatrixSimilarity(model[BoW_corpus], num_features=len(dictionary)+1)
print(model)
data = TecDataset([], [])
loader = TwitterLoader()
loader.load_twitter('data_collection/scrape/twitter_data.txt.head', data)
twitter_list = data.get_data()
twitter_BoW_corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in twitter_list]

tec_sim_twitter = []
for query in twitter_BoW_corpus:
    sims = index[model[query]]
    ind = np.argpartition(sims, -5)[-5:]
    ind = ind[np.argsort(sims[ind])]
    print(ind, sims[ind])
    tec_sim_twitter.append([np.max(sims), np.argmax(sims)])

import pandas as pd
pd.DataFrame(tec_sim_twitter).to_json('tec_sim_twitter.json.head', orient='split')