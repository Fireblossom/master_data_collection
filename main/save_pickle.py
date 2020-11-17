from data_collection.py_tec.tec_loader_spacy import TecLoader
loader = TecLoader(tokenize=True, augment=True)
import os
print(os.getcwd())
dataset = loader.load_tec('RLANS/data_collection/tec.txt')

import json

json.dump(dataset.get_data(), open('tec_data_da.json', 'w'))
json.dump(dataset.get_target(), open('tec_target_da.json', 'w'))