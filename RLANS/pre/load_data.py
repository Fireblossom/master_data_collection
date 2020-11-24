from data_collection.py_isear.isear_loader_spacy import IsearLoader
from data_collection.py_ssec.ssec_loader_spacy import SsecLoader
from data_collection.py_tec.tec_loader_spacy import TecLoader
import json
import nltk

def load_data():
    loader = TecLoader(True, augment=False)
    tec_data = loader.load_tec('../tec_train.txt', 0)
    attributes = []
    target = ['EMOT']
    loader = IsearLoader(attributes, target, True, True)
    isear_data = loader.load_isear('../isear_train.csv', 0)
    loader = SsecLoader(True)
    ssec_data = loader.load_ssec('data_collection/ssec-aggregated/train-combined-0.0.csv')
    ssec_data.target = json.load(open('../ssec_new_label.json'))
    return tec_data, isear_data, ssec_data


def get_freq_plot(data):
    if data == 'isear':
        attributes = []
        target = ['EMOT']
        loader = IsearLoader(attributes, target, True, True)

        data0 = loader.load_isear('../isear_train.csv', 0)
        all_tokens = [element for lis in data0.get_data() for element in lis]
        freq_dist_nltk_0 = nltk.FreqDist(all_tokens)
        data40 = loader.load_isear('../isear_train.csv', 40)
        all_tokens = [element for lis in data40.get_data() for element in lis]
        freq_dist_nltk_40 = nltk.FreqDist(all_tokens)
        data60 = loader.load_isear('../isear_train.csv', 60)
        all_tokens = [element for lis in data60.get_data() for element in lis]
        freq_dist_nltk_60 = nltk.FreqDist(all_tokens)
        data80 = loader.load_isear('../isear_train.csv', 80)
        all_tokens = [element for lis in data80.get_data() for element in lis]
        freq_dist_nltk_80 = nltk.FreqDist(all_tokens)

        return freq_dist_nltk_0, freq_dist_nltk_40, freq_dist_nltk_60, freq_dist_nltk_80
