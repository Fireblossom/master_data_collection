from data_collection.py_isear.isear_loader_spacy import IsearLoader
from data_collection.py_ssec.ssec_loader_spacy import SsecLoader
from data_collection.py_tec.tec_loader_spacy import TecLoader
import json


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


def get_freqc