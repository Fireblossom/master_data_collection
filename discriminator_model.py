import torch.nn as nn
import torch
import numpy as np


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index)  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    i = 0
    with open(filepath, encoding='utf-8') as f:
        f.readline()
        for line in f:
            if line != '\n':
                word, *vector = line.split()
                if word in word_index:
                    i += 1
                    idx = word_index[word] 
                    embedding_matrix[idx] = np.array(
                        vector, dtype=np.float32)[:embedding_dim]
    print('unk tokens:', vocab_size - i)
    return embedding_matrix


def weights_init(m):
    initrange = 0.1
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-initrange, initrange)
        m.bias.data.zero_()

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nclass, 
                 dropout_em=0.5,dropout_rnn=0,dropout_out=0, tie_weights=False, n_cl_hidden=30,
                 bidirection=False):
        super(RNNModel, self).__init__()
        self.drop_em = nn.Dropout(dropout_em)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
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
        self.dis_out = nn.Sequential(
            nn.Linear(nhid*n, n_cl_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_out),
            nn.Linear(n_cl_hidden, nclass)
        )

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder_dis.weight = self.encoder.weight

        self.init_weights()
        self.dis_out.apply(weights_init)

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

    def forward(self, input, hidden, last_location, vat=False, d=None):
        if vat:
            assert d is not None
            emb = self.drop_em(self.encoder(input)+d)
        else:
            emb = self.drop_em(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        decoded = self.dis_out(output[last_location-2, range(input.size()[1])])
        return decoded, output[last_location-2, range(input.size()[1])]

    def get_emb(self, input):
        return self.encoder(input)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.bidirection:
            n = 2
        else:
            n = 1
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*n, bsz, self.nhid),
                    weight.new_zeros(self.nlayers*n, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers*n, bsz, self.nhid)