from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet
from data_collection.py_additional.py_unlabeled import TwitterLoader
import json
import numpy as np
import torch


attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, False)
data = loader.load_isear('isear_train.csv', level=0)


doc_list = data.get_data()
doc_len = len(doc_list)
print(doc_len)
embeddings_tec = model.encode(doc_list, convert_to_tensor=True)

data = IsearDataSet()
loader = TwitterLoader()
loader.load_twitter('data_collection/scrape/blog_en.txt.head', data, tokenize=False)
doc_list = data.get_data()
doc_len = len(doc_list)
print(doc_len)

top_score = []
top_index = []
for i in range(0, doc_len, 128):
    print(i, '/', doc_len, end="\r")
    if i+128 >= doc_len:
        embeddings_twitter = model.encode(doc_list[i:], convert_to_tensor=True)
    else:
        embeddings_twitter = model.encode(doc_list[i:i+128], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings_tec, embeddings_twitter)
    score, index = torch.topk(cosine_scores, 5, dim=0)
    top_score.append(score)
    top_index.append(index)
    if i ==0:
        torch.save(cosine_scores, 'isear_sim_matrix_head.pt')

torch.save(torch.cat(top_score, dim=1), 'isear_sim_score.pt')
torch.save(torch.cat(top_index, dim=1), 'isear_sim_index.pt')

#import pandas as pd
#pd.DataFrame(tec_sim_twitter).to_json('tec_sim_twitter.json.head', orient='split')