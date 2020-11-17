from ssec_loader_spacy import SsecLoader
import numpy as np
import copy


loader = SsecLoader(True)
dataset0 = loader.load_ssec('RLANS/data_collection/ssec-aggregated/train-combined-0.0.csv')
dataset3 = loader.load_ssec('RLANS/data_collection/ssec-aggregated/train-combined-0.33.csv')
dataset5 = loader.load_ssec('RLANS/data_collection/ssec-aggregated/train-combined-0.5.csv')
dataset6 = loader.load_ssec('RLANS/data_collection/ssec-aggregated/train-combined-0.66.csv')
dataset9 = loader.load_ssec('RLANS/data_collection/ssec-aggregated/train-combined-0.99.csv')

labels = [
    dataset0.get_target(),
    dataset3.get_target(),
    dataset5.get_target(),
    dataset6.get_target(),
    dataset9.get_target()
]

print('total:',len(dataset0.get_target()), '\n\n')
labels_per_tweet = np.zeros((len(dataset0.get_target()), 5))
for i in range(len(dataset0.get_target())):
    for j in range(5):
        labels_per_tweet[i, j] = sum(labels[j][i])
print('Number of labels for each sample:\n', labels_per_tweet, '\n\n')
print('Average labels per level:\n', np.average(labels_per_tweet, axis=0), '\n\n')

label_loss = np.zeros((len(dataset0.get_target()), 5))
for i in range(4):
    label_loss[:,i+1] = labels_per_tweet[:,0]-labels_per_tweet[:,i+1]

print('Number of label losses:\n', label_loss, '\n\n')        
print('Average label losses per level (comapre to 0.0):\n', np.average(label_loss, axis=0)), '\n\n'

print('Generating new dataset.....')
newdataset = copy.deepcopy(dataset0)
for i in range(len(dataset0.get_target())):
    for j in range(4,-1, -1):
        if labels_per_tweet[i, j] > 0:
            # print(labels_per_tweet[i], labels_per_tweet[i, j])
            newdataset.target[i] = labels[4-j][i]
            break

#print(newdataset.get_target())
print('Average labels in new dataset:')
print(np.sum(newdataset.get_target())/len(newdataset.get_target()))

import json
json.dump(newdataset.get_data(), open('ssec_new_data.json', 'w'))
json.dump(newdataset.get_target(), open('ssec_new_label.json', 'w'))