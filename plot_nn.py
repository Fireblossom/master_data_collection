import matplotlib
import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, 
                    help='location of the data corpus')
parser.add_argument('--balance', type=int, default=99999999,
                    help='report interval')

args = parser.parse_args()

dirs = args.data+'_'+str(args.balance)+'_data'

for file in list(os.walk(dirs))[0][2]:
    if file[-4:] == '.png':
        print('png')
        continue
    count = np.array(json.load(open(dirs+'/'+file)))
    x = np.argsort(count)[::-1][:200]
    y = np.sort(count)[::-1][:200]
    print(y.shape)
    x_ = np.arange(200)
    plt.plot(x_,y, '-',color = 'g')
    plt.xlabel("Neibhgour")
    plt.ylabel("Frequency")
    plt.savefig(dirs+'/'+file+'.png')
    
