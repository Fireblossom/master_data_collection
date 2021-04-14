import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import pickle
import discriminator_model as model
import data
import pandas as pd
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, 
                    help='location of the data corpus')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################
dic_path = os.path.join(args.data, 'action_dictionary.pkl')
dic_exists = os.path.isfile(dic_path)
if dic_exists:
    with open(dic_path, 'rb') as input:
        Corpus_Dic = pickle.load(input)
else:
    Corpus_Dic = data.Dictionary()

test_data_name = os.path.join(args.data, 'test.csv')

if args.data == 'ssec':
    import json
    # train_data = data.SSEC_DataSet('data_collection/ssec-aggregated/train-combined-0.33.csv') 
    test_data = data.SSEC_DataSet('data_collection/ssec-aggregated/test-combined-0.33.csv')
    # train_data.load(dictionary=Corpus_Dic)
    # train_data.labels = json.load(open('ssec_new_label.json'))
    test_data.load(dictionary=Corpus_Dic, train_mode=False)
    test_data.labels = json.load(open('ssec_new_label_test.json'))
else:
    test_data = data.TEC_ISEAR_DataSet(args.data)
    test_data.load(dictionary=Corpus_Dic, train_mode=False, tokenize=True, level=0)

bitch_size = 128

print('The size of the dictionary is', len(Corpus_Dic))

def evaluate(model, test_loader):
    # Turn on evaluate mode which disables dropout.
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        gold = []
        pred = []
        for i_batch, sample_batched in enumerate(test_loader):
            token_seqs = torch.from_numpy(np.transpose(sample_batched[0])).to(device)
            labels = torch.from_numpy(np.array(sample_batched[3])).to(device)
            seq_lengths = np.transpose(sample_batched[4])
            hidden = model.init_hidden(token_seqs.shape[1])
            output, _ = model(token_seqs, hidden, seq_lengths)
            #print(output.size(), labels.size())
            if args.data == 'ssec':
                predict_class = (output > 0).float()
                pred.append(predict_class)
                correct += ((predict_class == labels).sum(axis=1).true_divide(8)).sum().item()
            else:
                _, predict_class = torch.max(output, 1)
                pred.append(predict_class)
                correct += (predict_class == labels).sum().item()
            total += labels.size(0)
            gold.append(labels)
        print('Accuracy of the classifier on the test data is : {:5.4f}'.format(
                100 * correct / total))
        y_true = torch.cat(gold).cpu().detach().numpy()
        y_pred = torch.cat(pred).cpu().detach().numpy()
        return correct / total, f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='micro')

if  __name__ == '__main__':
    result = {
        'model': [],
        'acc': [], 
        'micro': [],
        'macro': []
    }
    for filenames in os.walk('results/'+args.data):
        if ['classifier_model.pt'] in filenames:
            test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                    batch_size=bitch_size,
                                                    shuffle=True,
                                                    collate_fn=data.collate_fn)
            model_name = filenames[0] + '/classifier_model.pt'
            try:
                model = torch.load(model_name, map_location=device)
                acc, micro, macro = evaluate(model, test_loader)
                result['model'].append(filenames[0])
                result['acc'].append(acc)
                result['micro'].append(micro)
                result['macro'].append(macro)
            except Exception as e:
                print(model_name, e)
    df = pd.DataFrame(result)
    df.to_csv(args.data+'_result.csv')
    