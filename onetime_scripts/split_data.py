import numpy as np


# preclean
if False:
    isear = open('isear.csv')
    isear_train = open('isear_train.csv', 'w')
    isear_test = open('isear_test.csv', 'w')
    first_line = isear.readline()
    isear_train.write(first_line)
    isear_test.write(first_line)
    for line in isear:
        line = line.replace('á|', '')
        line = line.replace('á ', '')
        line = line.replace('á', '')
        if np.random.random() < 0.8:
            isear_train.write(line)
        else:
            isear_test.write(line)

if False:
    tec = open('tec.txt')
    tec_train = open('tec_train.txt', 'w')
    tec_test = open('tec_test.txt', 'w')
    for line in tec:
        if np.random.random() < 0.8:
            tec_train.write(line)
        else:
            tec_test.write(line)

if True:
    isear_train = open('isear_train.csv')
    isear_train_labeled = open('isear_train_labeled.csv', 'w')
    count = [0] * 7
    first_line = isear_train.readline()
    isear_train_labeled.write(first_line)
    for line in isear_train:
        if count[int(line.split('|')[11])-1] <= 500:
            isear_train_labeled.write(line)
            count[int(line.split('|')[11])-1] += 1

