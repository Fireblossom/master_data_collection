from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet
import json
import numpy as np


attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True, True)
dataset = loader.load_isear('RLANS/data_collection/isear.csv')
dataset.set_data(json.load(open('isear_raw_data.json')))
train_text, train_target = [], []
test_text, test_target = [], []
for text, target in zip(dataset.get_data(), dataset.get_target()):
    if np.random.random() > 0.9:
        test_text.append(text)
        test_target.append(target)
    else:
        train_text.append(text)
        train_target.append(target)

json.dump(train_text, open('isear_train_text_raw.json', 'w'))
json.dump(train_target, open('isear_train_target_raw.json', 'w'))
json.dump(test_text, open('isear_test_text_raw.json', 'w'))
json.dump(test_target, open('isear_test_target_raw.json', 'w'))
