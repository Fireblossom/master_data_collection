import numpy as np
import pandas as pd
import os
import warnings
import random
import torch 
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaConfig
from RoBERTa_train import TweetModel, TweetDataset, get_selected_text
import argparse
import data

from data_collection.py_isear.isear_loader_spacy import IsearLoader, IsearDataSet, IsearSubset
from data_collection.py_tec.tec_loader_spacy import TecLoader, TecDataset
from data_collection.py_additional.py_unlabeled import EmoeventLoader, TwitterLoader, StoryLoader


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM classification Model')
parser.add_argument('--data', type=str, default=os.getcwd()+'/ag_news_csv/',
                    help='location of the data corpus')
parser.add_argument('--label', type=str, default='', metavar='N',
                    help='vocab level')


args = parser.parse_args()

if args.data == 'isear':
    attributes = []
    target = ['EMOT']
    loader = IsearLoader(attributes, target, True, False)
    data = loader.load_isear('isear_train.csv', 0)
elif args.data == 'tec':
    loader = TecLoader(False, augment=False)
    data = loader.load_tec('tec_train.txt', 0)
elif args.data == 'blog':
    data = IsearDataSet()
    loader = TwitterLoader()
    loader.load_twitter('data_collection/scrape/blog_en.txt', data, tokenize=False)
elif args.data == 'twitter':
    data = TecDataset([],[])
    loader = TwitterLoader()
    loader.load_twitter('data_collection/scrape/twitter_data.txt', data, tokenize=False)

text = data.get_data()
if args.data == 'tec' or args.data == 'isear':
    target = data.get_target()
    label = []
    for l in target:
        if l == [1] or l == 1:
            label.append('positive')
        else:
            label.append('negative')
else:
    label = [args.label] * len(text)
df = pd.DataFrame({
    'text': text,
    'sentiment': label
})
print(df)

bitch_size = 32
test_loader = torch.utils.data.DataLoader(dataset=TweetDataset(df),
                                           batch_size=bitch_size,
                                           shuffle=False)

file = open(args.data+args.label+'_sti.txt', 'w')

model = TweetModel()
model.cuda()
fold = 0
model.load_state_dict(torch.load(f'roberta_fold{fold+1}.pth'))
model.eval()

for data in test_loader:
    # print(data)
    ids = data['ids'].cuda()
    masks = data['masks'].cuda()
    tweet = data['tweet']
    offsets = data['offsets'].numpy()

    with torch.no_grad():
        output = model(ids, masks)
        start_logits = torch.softmax(output[0], dim=1).cpu().detach().numpy()
        end_logits = torch.softmax(output[1], dim=1).cpu().detach().numpy()

    for i in range(len(ids)):    
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        file.write(pred+'\n')
        # predictions.append(pred)