import torch.nn as nn
import torch
import numpy as np


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index)  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, encoding='utf-8') as f:
        f.readline()
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout_em=0.5,dropout_rnn=1,dropout_out=1, tie_weights=False, bidirection=False):
        super(RNNModel, self).__init__()
        self.drop_em = nn.Dropout(dropout_em)
        self.drop_out = nn.Dropout(dropout_out)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.bidirection = bidirection
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout_rnn, bidirectional=bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout_rnn)
        n = 2 if bidirection else 1
        self.decoder = nn.Linear(nhid*n, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def load_embedding(self, filepath, Corpus_Dic, device):
        print('loading embedding......')
        self.encoder = self.encoder.from_pretrained(torch.Tensor(create_embedding_matrix(filepath, Corpus_Dic.word2idx, self.ninp))).to(device)
        print('embedding loaded.')

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop_em(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop_out(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        n = 2 if self.bidirection else 1
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*n, bsz, self.nhid),
                    weight.new_zeros(self.nlayers*n, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers*n, bsz, self.nhid)
