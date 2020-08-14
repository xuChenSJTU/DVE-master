"""
Author: Xu Chen, Shanghai Jiao Tong University
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse
from DE_model import VGE_EDGE3
import File_Reader
import pickle
import utils
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = ' '
dataset_file = 'epinions_small'
ratio = 1.0
latent_d = 64
n_sample_noise = 5
use_feature = False
parser = argparse.ArgumentParser(description='')

parser.add_argument('--result_dir', dest='result_dir', default='./result', help='result of the model testing')
parser.add_argument('--dataset', dest='dataset', default=dataset_file, help='name of the training dataset')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1000, help='training batch size')
parser.add_argument('--n_hidden', dest='n_hidden', type=int, default=128, help='dimension of hidden layer')
parser.add_argument('--n_embedding', dest='n_embedding', type=int, default=64, help='dimension of embedding vector')
parser.add_argument('--dropout', dest='dropout', type=bool, default=True, help='Using dropout in training')
parser.add_argument('--keep_prob', dest='keep_prob', type=float, default=0.8, help='prob of keeping activitation nodes')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=200, help='max iteration step')

args = parser.parse_args()

if __name__ == '__main__':
    print('################### dataset: {}, ratio: {}, latent_d: {} #############'.format(dataset_file, ratio, latent_d))
    args.n_hidden_list = [2*latent_d, latent_d]  # default latent_d = 64
    args.n_gcn_layer = len(args.n_hidden_list)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    node2idx = pickle.load(open(os.path.join(os.getcwd(), 'output', '{}_node2idx.pkl'.format(dataset_file)), 'rb'))
    # construct node2idx df
    node2idx_df = pd.Series(list(node2idx.values()),
                            index=[str(int(float(ele.decode()))) for ele in list(node2idx.keys())])
    n_nodes = len(node2idx)
    n_fold = 5
    print('#########################parameter setting###########################')
    for i in range(n_fold)[:1]:
        # generate triplets training samples
        # load train links and test links
        train_links = np.loadtxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(i, ratio)))

        # transform nodeID to nodeIdx in train/test links df
        train_links_df = pd.DataFrame(data=train_links)
        train_links_df.columns = ['fromID', 'toID', 'sign']
        train_links_df['fromIdx'] = node2idx_df[list(train_links_df['fromID'].values.astype(int).astype(str))].values
        train_links_df['toIdx'] = node2idx_df[list(train_links_df['toID'].values.astype(int).astype(str))].values

        if not os.path.exists(os.path.join(os.getcwd(), 'data', '{}_train_triplets_cv{}_n_noise{}_ratio{}.pkl'.format(dataset_file, i,
                                                                                                                      n_sample_noise, ratio))):
            print('training triplet {} fold noise {} ratio {} is generating......'.format(i, n_sample_noise, ratio))
            utils.generate_training_triplets(train_links_df, n_nodes, dataset_file, i, n_sample_noise, ratio)
        else:
            print('{} fold training triplet noise {} ratio {} has generated!!!'.format(i, n_sample_noise, ratio))

    auc_sign_cv = []
    f1_sign_cv = []

    recall5_cv = []
    recall10_cv = []
    recall20_cv = []
    precision5_cv = []
    precision10_cv = []
    precision20_cv = []

    topN_list = [10, 20, 50]
    for i in range(n_fold)[:1]:
        # load train links and test links
        train_links = np.loadtxt(os.path.join(os.getcwd(), 'test', dataset_file + '_train_links_cv{}_ratio{}.txt'.format(i, ratio)))
        test_links = np.loadtxt(os.path.join(os.getcwd(), 'test', dataset_file + '_test_links_cv{}.txt'.format(i)))

        # transform nodeID to nodeIdx in train/test links df
        train_links_df = pd.DataFrame(data=train_links)
        train_links_df.columns = ['fromID', 'toID', 'sign']
        train_links_df['fromIdx'] = node2idx_df[list(train_links_df['fromID'].values.astype(int).astype(str))].values
        train_links_df['toIdx'] = node2idx_df[list(train_links_df['toID'].values.astype(int).astype(str))].values

        test_links_df = pd.DataFrame(data=test_links)
        test_links_df.columns = ['fromID', 'toID', 'sign']
        test_links_df['fromIdx'] = node2idx_df[list(test_links_df['fromID'].values.astype(int).astype(str))].values
        test_links_df['toIdx'] = node2idx_df[list(test_links_df['toID'].values.astype(int).astype(str))].values

        # if use_feature == True:
        #     # construct training node features
        #     train_node_features = utils.generate_node_features(train_links_df, n_nodes)

        # construct training data for our model
        train_sign_pos_df = train_links_df[train_links_df['sign'].values == 1]
        train_sign_neg_df = train_links_df[train_links_df['sign'].values == -1]
        args.n_pos_edges = len(train_sign_pos_df)
        args.n_neg_edges = len(train_sign_neg_df)

        train_sign_pos_row = train_sign_pos_df['fromIdx'].values
        train_sign_pos_col = train_sign_pos_df['toIdx'].values
        train_sign_pos_sp = sparse.coo_matrix(
            (np.ones(shape=len(train_sign_pos_row)), (train_sign_pos_row, train_sign_pos_col)),
            shape=(n_nodes, n_nodes))
        train_sign_pos_sp = train_sign_pos_sp.tocsr()
        train_sign_pos_sp = ((train_sign_pos_sp + np.transpose(train_sign_pos_sp)) != 0).astype(float)

        train_sign_neg_row = train_sign_neg_df['fromIdx'].values
        train_sign_neg_col = train_sign_neg_df['toIdx'].values
        train_sign_neg_sp = sparse.coo_matrix(
            (np.ones(shape=len(train_sign_neg_row)), (train_sign_neg_row, train_sign_neg_col)),
            shape=(n_nodes, n_nodes))
        train_sign_neg_sp = train_sign_neg_sp.tocsr()
        train_sign_neg_sp = ((train_sign_neg_sp + np.transpose(train_sign_neg_sp)) != 0).astype(float)

        # prepare training out nodeIdx and in nodeIdx
        f = open(os.path.join(os.getcwd(), 'data', '{}_train_triplets_cv{}_n_noise{}_ratio{}.pkl'.format(dataset_file, i, n_sample_noise, ratio)), 'rb')
        feed_data = pickle.load(f)
        print('{} fold triplets ratio {} has {} samples'.format(i, ratio, len(feed_data)))
        f.close()

        # construct tensorflow session
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            model = VGE_EDGE3(sess, n_nodes, args)
            auc_sign_list, \
            f1_sign_list, \
            recall5_list, \
            recall10_list, \
            recall20_list, \
            precision5_list, \
            precision10_list, \
            precision20_list = model.train(args, train_sign_pos_sp, train_sign_neg_sp, feed_data, train_links_df,
                                            test_links_df, topN_list)

        auc_sign_cv.append(auc_sign_list)
        f1_sign_cv.append(f1_sign_list)

        recall5_cv.append(recall5_list)
        recall10_cv.append(recall10_list)
        recall20_cv.append(recall20_list)
        precision5_cv.append(precision5_list)
        precision10_cv.append(precision10_list)
        precision20_cv.append(precision20_list)

    max_auc_sign_cv = [max(ele) for ele in auc_sign_cv]
    max_f1_sign_cv = [max(ele) for ele in f1_sign_cv]

    max_recall5_cv = [max(ele) for ele in recall5_cv]
    max_recall10_cv = [max(ele) for ele in recall10_cv]
    max_recall20_cv = [max(ele) for ele in recall20_cv]

    max_precision5_cv = [max(ele) for ele in precision5_cv]
    max_precision10_cv = [max(ele) for ele in precision10_cv]
    max_precision20_cv = [max(ele) for ele in precision20_cv]

    print('best auc_sign: {}\nbest f1_sign: {}\nbest recall{}: {}\nbest recall{}: {}\n'
          'best recall{}: {}\nbest precision{}: {}\nbest precision{}: {}\nbest precision{}: {}\n'
          .format(max_auc_sign_cv, max_f1_sign_cv, topN_list[0], max_recall5_cv, topN_list[1], max_recall10_cv,
                  topN_list[2], max_recall20_cv,
                  topN_list[0], max_precision5_cv, topN_list[1], max_precision10_cv, topN_list[2], max_precision20_cv))
    print('mean auc_sign: {}\nmean f1_sign: {}\nmean recall{}: {}\nmean recall{}: {}\n'
          'mean recall{}: {}\nmean precision{}: {}\nmean precision{}: {}\nmean precision{}: {}\n'
          .format(np.mean(max_auc_sign_cv), np.mean(max_f1_sign_cv), topN_list[0], np.mean(max_recall5_cv),
                  topN_list[1], np.mean(max_recall10_cv), topN_list[2], np.mean(max_recall20_cv),
                  topN_list[0], np.mean(max_precision5_cv), topN_list[1], np.mean(max_precision10_cv),
                  topN_list[2], np.mean(max_precision20_cv)))
    print(
        '################### dataset: {}, ratio: {}, latent_d: {} #############'.format(dataset_file, ratio, latent_d))