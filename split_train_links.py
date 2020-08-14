import pandas as pd
import os
import pickle
from sklearn.utils import shuffle
import numpy as np

dataset_file = 'wiki_small'

n_fold = 5
for fold_num in range(n_fold):
    # load original train links
    train_links = np.loadtxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}.txt'.format(fold_num)))

    # split different ratio training links
    train_links = shuffle(train_links)
    split0_train_links = train_links[:int(0.2*len(train_links))]
    split1_train_links = train_links[:int(0.4*len(train_links))]
    split2_train_links = train_links[:int(0.6 * len(train_links))]
    split3_train_links = train_links[:int(0.8 * len(train_links))]

    np.savetxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(fold_num, 0.2)),
               split0_train_links)
    np.savetxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(fold_num, 0.4)),
               split1_train_links)
    np.savetxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(fold_num, 0.6)),
               split2_train_links)
    np.savetxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(fold_num, 0.8)),
               split3_train_links)
    np.savetxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(fold_num, 1.0)),
               train_links)