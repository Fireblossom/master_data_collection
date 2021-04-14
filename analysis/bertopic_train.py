import argparse
from bertopic import BERTopic
import os
import json
from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet, IsearSubset
from data_collection.py_ssec.ssec_loader_spacy import SsecLoader, SsecDataset
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_additional.py_unlabeled import TwitterLoader
import pickle

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, 
                    help='location of the data corpus')
args = parser.parse_args()

if args.data == 'isear':
    attributes = []
    target = ['EMOT']
    loader = IsearLoader(attributes, target, True, False)
    data = loader.load_isear('isear_train.csv', 0)
elif args.data == 'tec':
    loader = TecLoader(False, augment=False)
    data = loader.load_tec('tec_train.txt', 0)
else:
    data = IsearDataSet()
    loader = TwitterLoader(tokenize=False)
    loader.load_twitter(args.data, data, tokenize=False)

data = data.get_data()

#model = pickle.load(open(args.data+'_bertopic.pt', 'rb'))
#print(model.transform(data))
model = BERTopic(embedding_model="stsb-roberta-base", n_gram_range=(2, 3), stop_words="english", min_topic_size=5, nr_topics=100, verbose=True)
model.fit(data)
pickle.dump(model, open(args.data+'_bertopic.pkl', 'wb'), protocol=3)
model.save(args.data+'_bertopic.pt')