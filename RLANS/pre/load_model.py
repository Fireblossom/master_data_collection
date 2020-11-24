import torch


def load_model(path):
    model = torch.load(path)
    return model


def evaluate(model, data):
    # Turn on evaluate mode which disables dropout.
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            token_seqs = torch.from_numpy(np.transpose(sample_batched[0])).to(device)
            labels = torch.from_numpy(np.array(sample_batched[3])).to(device)
            seq_lengths = np.transpose(sample_batched[4])
            hidden = model.init_hidden(token_seqs.shape[1])
            output = model(token_seqs, hidden, seq_lengths)
            #print(output.size(), labels.size())
            if args.data == 'ssec':
                predict_class = (output > 0).float()
                correct += ((predict_class == labels).sum(axis=1).true_divide(8)).sum().item()
            else:
                _, predict_class = torch.max(output, 1)
                correct += (predict_class == labels).sum().item()
            total += labels.size(0)
        print('Accuracy of the classifier on the test data is : {:5.4f}'.format(
                100 * correct / total))
        return correct / total