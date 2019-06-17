import os
import sys

from collections import defaultdict

import numpy as np

def main(allLSTMFpath, ratio_train=0.8):
    
    dataset = defaultdict(lambda : list())

    with open(allLSTMFpath, 'r') as allLSTMF:
        for entry in allLSTMF:
            basename = '.'.join(entry.split('.')[:-1])
            gtpath = basename + '.gt.txt'

            with open(gtpath, 'r') as gtfile:
                text = gtfile.read()

            dataset[text].append(entry)

        
    #print("dataset: ", dataset)

    uniquewords = list(dataset.keys())
    n_uniquewords = len(uniquewords)
    print("Total number of unique words {}".format(n_uniquewords))

    np.random.shuffle(uniquewords)

    nTrain = int(n_uniquewords * ratio_train)

    trainwords = uniquewords[:nTrain]
    valwords = uniquewords[nTrain:]

    with open('data/list.train', 'w') as listtrain:
        for trainword in trainwords:
            for entry in dataset[trainword]:
                listtrain.write(entry)


    with open('data/list.eval', 'w') as listval:
        for valword in valwords:
            for entry in dataset[valword]:
                listval.write(entry)



if __name__ == '__main__':
    allLSTMFpath = sys.argv[1]
    ratio_train = float(sys.argv[2])

    main(allLSTMFpath, ratio_train)
