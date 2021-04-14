import pandas as pd
import numpy as np
import argparse
import linecache
import os
import json
import torch

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language Model')
parser.add_argument('--data', type=str, default='twitter', 
                    help='location of the data corpus')
parser.add_argument('--size', nargs='+', type=int, default=15526,
                    help='report interval')
parser.add_argument('--balance', type=int, default=99999999,
                    help='report interval')
parser.add_argument('--average', nargs='+', type=float, 
                    help='report interval')

args = parser.parse_args()

dirs = args.data+'_'+str(args.balance)+'_data'
if not os.path.exists(dirs):
    os.makedirs(dirs)

"""if args.data == 'twitter':
    df = pd.read_json('tec_sim_twitter.json', orient='split')
    filename = 'data_collection/scrape/twitter_data.txt'
elif args.data == 'blog':
    df = pd.read_json('isear_sim_blog.json', orient='split')
    filename = 'data_collection/scrape/blog_en.txt'
else:
    print('error')
    exit()

sims = np.array(df[0])
nn = np.array(df[1])
ind = np.argsort(sims)"""

if args.data == 'twitter':
    sims = torch.load('tec_sim_score.pt').numpy()[0]
    nn = torch.load('tec_sim_index.pt').numpy()[0]
    ind = np.argsort(sims)
    filename = 'data_collection/scrape/twitter_data.txt'

elif args.data == 'blog':
    sims = torch.load('tec_sim_score.pt').numpy()[0]
    nn = torch.load('tec_sim_index.pt').numpy()[0]
    ind = np.argsort(sims)
    filename = 'data_collection/scrape/blog_en.txt'
else:
    print('error')
    exit()

for size in args.size:
    count = [0] * (max(nn) + 1)
    j = 0
    """
    mean = np.sum(sims[ind[0 : 0+size]])
    for i in range(1, len(ind)-size):
        mean -= sims[ind[i-1]]
        mean += sims[ind[i : i+size][-1]]
        # mean = np.sum(sims[ind[i : i+size]])
        if i = 0:
            for j in range(size):


        elif args.balance > 0 and count[nn[ind[i : i+size][-1]]] == args.balance:
            mean += sims[ind[i-1]]
            
        elif mean >= args.average[j]*size or i+size == len(ind)-1:
            j += 1
            print(mean/size)
            with open(args.data+'_'+args.balance+'_data/'+args.data+'_'+str(size)+'_'+str(mean/size)[:4]+'.txt', 'w') as file:
                for index in ind[i : i+size] + 1:
                    file.write(linecache.getline(filename, index))
    """
    
    index = []
    add = 0
    for i in range(len(ind)):
        if len(index) < size and count[nn[ind[i]]] < args.balance:
            index.append(ind[i])
            add += sims[ind[i]]
            count[nn[ind[i]]] += 1
        elif len(index) >= size and count[nn[ind[i]]] < args.balance:
            r = index.pop(0)
            count[nn[r]] -= 1
            add -= sims[r]
            add += sims[ind[i]]
            index.append(ind[i])
            count[nn[ind[i]]] += 1
        elif count[nn[ind[i]]] >= args.balance:
            tmp = np.array(index)
            try:
                k = np.where(nn[index] == nn[ind[i]])[0][0]
            except:
                print(count[nn[ind[i]]], nn[index], nn[ind[i]], np.where(nn[index] == nn[ind[i]]))
                exit()
            r = index.pop(k)
            add -= sims[r]
            add += sims[ind[i]]
            index.append(ind[i])
        
        if len(index) >= size:
            if add >= args.average[j]*size or i == len(ind)-1:
                j += 1
                print(add/size)
                with open(args.data+'_'+str(args.balance)+'_data/'+args.data+'_'+str(size)+'_'+str(add/size)[:4]+'_distribution.json', 'w') as file:
                    json.dump(count, file, indent=2)
                with open(args.data+'_'+str(args.balance)+'_data/'+args.data+'_'+str(size)+'_'+str(add/size)[:4]+'.txt', 'w') as file:
                    for c in index:
                        file.write(linecache.getline(filename, c+1))