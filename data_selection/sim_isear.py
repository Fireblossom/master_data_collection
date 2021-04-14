import gensim
from gensim import corpora, models, similarities
from gensim.similarities import MatrixSimilarity
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet
from data_collection.py_additional.py_unlabeled import EmoeventLoader, TwitterLoader, StoryLoader
import json
import numpy as np

attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, True)
data = loader.load_isear('isear_train.csv', level=0)
doc_list = data.get_data()
doc_len = len(doc_list)
print(doc_len)

dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]
# print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]

model = models.TfidfModel(BoW_corpus)
index = MatrixSimilarity(model[BoW_corpus], num_features=len(dictionary)+1)
print(model)
data = IsearDataSet()
loader = TwitterLoader()
loader.load_twitter('data_collection/scrape/blog_en.txt', data)
doc_list = data.get_data()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=False) for doc in doc_list]

isear_sim_blog = []
i = 0
for query in [dictionary.doc2bow(doc, allow_update=False) for doc in data.get_data()]:
    sims = index[model[query]]
    isear_sim_blog.append([np.max(sims), np.argmax(sims)])
    i+=1
    if i >= 2000000:
        break

import pandas as pd
pd.DataFrame(isear_sim_blog).to_json('isear_sim_blog.json', orient='split')
