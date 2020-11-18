from data_collection.py_isear.isear_loader_spacy import IsearLoader
import json

attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, False)
dataset = loader.load_isear('RLANS/data_collection/isear.csv')
json.dump(dataset.get_data(), open('isear_raw_data.json', 'w'))
json.dump(dataset.get_target(), open('isear_raw_target.json', 'w'))