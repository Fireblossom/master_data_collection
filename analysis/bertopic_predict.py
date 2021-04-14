import argparse
from bertopic import BERTopic
import os
import json
from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet, IsearSubset
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_additional.py_unlabeled import EmoeventLoader, TwitterLoader, StoryLoader
import pickle

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, 
                    help='location of the data corpus')
parser.add_argument('--unlabel', type=str,
                    help='report interval')
args = parser.parse_args()

unlabel_files = iter(list(os.walk(args.unlabel))[0][2])
model = pickle.load(open(args.data+'_bertopic.pkl', 'rb'))

for unlabel_file in unlabel_files:
    print(unlabel_file)
    data = IsearDataSet()
    loader = TwitterLoader(tokenize=False)
    loader.load_twitter(args.unlabel+'/' + unlabel_file, data, tokenize=False)

    data = data.get_data()
    print(data[2])
    predict = model.transform(data)

    # json.dump(predict.tolist(), open(args.unlabel+'/' + unlabel_file + '.json', 'w'), indent=2)
    print(predict)
    #pickle.dump(predict, open(args.unlabel+'/' + unlabel_file + '.pt', 'wb'), protocol=3)