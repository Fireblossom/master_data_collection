import torch, pickle
import data as util
import numpy as np


def load_model(path):
    model = torch.load(path)
    return model


def evaluate(model, data, full):
    # Turn on evaluate mode which disables dropout.
    if data == 'ssec':
        raise NotImplementedError
    
    if full:
        Corpus_Dic = pickle.load(open('../'+data+'/action_dictionary_full.pkl', 'rb'))
    else:
        Corpus_Dic = pickle.load(open('../'+data+'/action_dictionary.pkl', 'rb'))
    
    test_data = util.TEC_ISEAR_DataSet(data)
    test_data.load(dictionary=Corpus_Dic, train_mode=False, tokenize=True, level=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=32,
                                          shuffle=True,
                                          collate_fn=util.collate_fn)
    correct = 0
    total = 0
    predict = []
    label = []
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            token_seqs = torch.from_numpy(np.transpose(sample_batched[0]))
            labels = torch.from_numpy(np.array(sample_batched[3]))
            seq_lengths = np.transpose(sample_batched[4])
            hidden = model.init_hidden(token_seqs.shape[1])
            output = model(token_seqs, hidden, seq_lengths)
            #print(output.size(), labels.size())
            if data == 'ssec':
                predict_class = (output > 0).float()
                correct += ((predict_class == labels).sum(axis=1).true_divide(8)).sum().item()
            else:
                _, predict_class = torch.max(output, 1)
                correct += (predict_class == labels).sum().item()
            predict += predict_class
            label += labels
            total += labels.size(0)
        print('Accuracy of the classifier on the test data is : {:5.4f}'.format(
                100 * correct / total))
        return predict, label