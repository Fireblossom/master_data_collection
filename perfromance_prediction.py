import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Performance Prediction')
parser.add_argument('--data', type=str, default='isear',
                    help='location of the data corpus')
parser.add_argument('--target', type=str, default='acc',
                    help='location of the data corpus')
args = parser.parse_args()


class LinearNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(LinearNet, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden)
        self.predict_layer = torch.nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        hidden_result = self.hidden_layer(x)
        relu_result = self.relu(hidden_result)
        predict_result = self.predict_layer(relu_result)
        return self.softmax(predict_result)


num_feature = 5
net = LinearNet(num_feature, 10, 1)

result = pd.read_csv(args.data + '_result.csv')
model = result['model']
features = torch.zeros(len(model), num_feature)
mapping = dict(
    vat=0,
    sssl=1,
    both=2
)
size = 6143 if args.data == 'isear' else 15526
for i, exp in enumerate(model):
    e = exp.split('/')
    if len(e) >= 6:
        features[i, 0] = mapping[e[3]] # model
        features[i, 1] = int(e[4].split('_')[1]) # balance
        if features[i, 1] > 10:
            features[i, 1] = 0
        features[i, 2] = float(e[5].split('_')[1])/size # size
        features[i, 3] = float(e[5].split('_')[2]) # sim
    else:
        features[i, 4] = 1 # baseline

labels = torch.FloatTensor(result[args.target])
dataset = Data.TensorDataset(features, labels)
# print(features, labels)
batch_size = 1
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00003)

num_epochs = 300
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        print(X,y)
        output = net(X)
        print(output, y)
        loss = loss_fn(output, y.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, loss.item()))