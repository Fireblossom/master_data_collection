import gensim
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from data_collection.py_tec.tec_loader_spacy import TecLoader
from data_collection.py_isear.isear_loader_spacy import IsearLoader
from data_collection.py_additional.py_emoevent import EmoeventLoader, TwitterLoader, StoryLoader
import json


'''loader = TecLoader(tokenize=True, augment=False)
data = loader.load_tec('tec_train.txt', level=0)
doc_list = data.get_data()
dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]
# print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
# print(id_words)
index = MatrixSimilarity(BoW_corpus, num_features=len(dictionary)+1)

loader = EmoeventLoader(tokenize=True)
loader.load_emoevent('dataset_emotions_EN.txt', data)

tec_sim_emoevent = []
for query in [dictionary.doc2bow(doc, allow_update=False) for doc in data.get_data()]:
    tec_sim_emoevent.append(index[query].tolist())
json.dump(tec_sim_emoevent, open('tec_sim_emoevent.json', 'w'), indent=2)'''


loader = TecLoader(tokenize=True, augment=False)
data = loader.load_tec('tec_train.txt', level=0)
doc_list = data.get_data()
dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]
# print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
# print(id_words)
index = MatrixSimilarity(BoW_corpus, num_features=len(dictionary)+1)

loader = TwitterLoader()
loader.load_twitter('RLANS/data_collection/scrape/anger_en.txt', data)
loader.load_twitter('RLANS/data_collection/scrape/disgust_en.txt', data)
loader.load_twitter('RLANS/data_collection/scrape/fear_en.txt', data)
loader.load_twitter('RLANS/data_collection/scrape/joy_en.txt', data)
loader.load_twitter('RLANS/data_collection/scrape/sadness_en.txt', data)
loader.load_twitter('RLANS/data_collection/scrape/surprise_en.txt', data)

tec_sim_twitter = []
for query in [dictionary.doc2bow(doc, allow_update=False) for doc in data.get_data()]:
    tec_sim_twitter.append(index[query].tolist())
json.dump(tec_sim_twitter, open('tec_sim_twitter.json', 'w'), indent=2)


attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, True)
data = loader.load_isear('isear_train.csv', level=0)
doc_list = data.get_data()
dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]
# print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
# print(id_words)
index = MatrixSimilarity(BoW_corpus, num_features=len(dictionary)+1)
loader = EmoeventLoader(tokenize=tokenize)
loader.load_emoevent('dataset_emotions_EN.txt', data)
isear_sim_emoevent = []
for query in [dictionary.doc2bow(doc, allow_update=False) for doc in data.get_data()]:
    isear_sim_emoevent.append(index[query].tolist())
json.dump(isear_sim_emoevent, open('isear_sim_emoevent.json', 'w'), indent=2)


attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, True)
data = loader.load_isear('isear_train.csv', level=0)
doc_list = data.get_data()
dictionary = corpora.Dictionary()
BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in doc_list]
# print(BoW_corpus)
id_words = [[(dictionary[id], count) for id, count in line] for line in BoW_corpus]
# print(id_words)
index = MatrixSimilarity(BoW_corpus, num_features=len(dictionary)+1)
loader = StoryLoader(tokenize=tokenize)
loader.load_story('RLANS/data_collection/scrape/story.json', data)

isear_sim_story = []
for query in [dictionary.doc2bow(doc, allow_update=False) for doc in data.get_data()]:
    isear_sim_story.append(index[query].tolist())
json.dump(isear_sim_story, open('isear_sim_story.json', 'w'), indent=2)