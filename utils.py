"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy import sparse
import pickle
import os
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

def generate_node_features(train_links_df, n_node):
    # This is a function for generating feature for each node
    return train_links_df

def generate_training_triplets(train_links_df, n_nodes, dataset_file, fold_num, n_sample_noise, choose_ratio = None):
    A_sp = sparse.coo_matrix(
        (train_links_df['sign'].values, (train_links_df['fromIdx'].values, train_links_df['toIdx'].values)),
        shape=(n_nodes, n_nodes))
    A = A_sp.todense()

    threshold = 200
    n_noise = n_sample_noise

    triplets = []

    labels = []
    for i in range(A.shape[0]):
        poss = np.where(A[i, :] == 1)[1]
        negs = np.where(A[i, :] == -1)[1]
        all_noises = np.where(A[i, :] == 0)[1]
        all_noises = shuffle(all_noises)
        sampled_noises = all_noises[:n_noise]

        if len(poss)!=0:
            # generate [i, j, k], ij=1, ik=0
            for pos_ind in poss:
                 for noise_ind in sampled_noises:
                    triplets.append([i, pos_ind, noise_ind])
                    labels.append([i, pos_ind, 1.0])

        if len(negs)!=0:
            # generate [i, j, k], ij=0, ik=-1
            for noise_ind in sampled_noises:
                for neg_ind in negs:
                    triplets.append([i, noise_ind, neg_ind])
                    labels.append([i, neg_ind, -1.0])

    train_triplets = np.concatenate([np.array(triplets), np.array(labels)], axis=1)
    np.random.shuffle(train_triplets)
    triplets = train_triplets[:,:3]
    labels = train_triplets[:,3:]
    # np.random.shuffle(triplets)
    triplets = np.asarray(triplets, dtype='int32')
    labels = np.asarray(labels, dtype='int32')

    '''
    Note that the SNEA name is only for method SNEA, for all other methods, 
    you should delete the name to generate the training data
    '''
    if choose_ratio==None:
        pickle.dump(triplets, open(os.path.join(os.getcwd(), 'data',
                                                'SNEA_{}_train_triplets_cv{}_n_noise{}.pkl'.format(dataset_file, fold_num, n_noise)), 'wb'))
        pickle.dump(labels, open(os.path.join(os.getcwd(), 'data',
                                                'SNEA_{}_train_labels_cv{}_n_noise{}.pkl'.format(dataset_file, fold_num,
                                                                                              n_noise)), 'wb'))
    else:
        pickle.dump(triplets, open(os.path.join(os.getcwd(), 'data',
                                                'SNEA_{}_train_triplets_cv{}_n_noise{}_ratio{}.pkl'.format(dataset_file, fold_num,
                                                                                              n_noise, choose_ratio)), 'wb'))
        pickle.dump(labels, open(os.path.join(os.getcwd(), 'data',
                                                'SNEA_{}_train_labels_cv{}_n_noise{}_ratio{}.pkl'.format(dataset_file,
                                                                                                      fold_num,
                                                                                                      n_noise,
                                                                                                      choose_ratio)), 'wb'))


def eva_recall(truth_dict, recom_dict, topN=5):
    recall_list = []
    for node in truth_dict:
        recall_value = len(set(truth_dict[node].tolist()) & set(recom_dict[node].tolist()[:topN]))*1.0/len(set(truth_dict[node].tolist()))
        if recall_value==1.0:
            pass
        recall_list.append(recall_value)
    return np.mean(recall_list)

def eva_precision(truth_dict, recom_dict, topN=5):
    precision_list = []
    for node in truth_dict:
        precision_value = len(set(truth_dict[node].tolist()) & set(recom_dict[node].tolist()[:topN]))*1.0/topN
        precision_list.append(precision_value)
    return np.mean(precision_list)

def node_recomm_vge_SiNE(train_links_df, test_links_df, out_z, in_z, paras_dict):
    test_links_df_pos = test_links_df[test_links_df['sign'] == 1]

    # construct ground truth for node recommendation
    truth_dict = {}
    original_nodes = test_links_df_pos['fromIdx'].values
    for node in original_nodes:
        truth_nodes_list = test_links_df_pos[test_links_df_pos['fromIdx'] == node]['toIdx'].values
        truth_dict[node] = truth_nodes_list

    # make recommendation list
    recom_dict = {}
    for node in truth_dict:
        node_embedding = np.reshape(out_z[node], newshape=[1, -1])
        tile_node_embedding = np.tile(node_embedding, [in_z.shape[0], 1])
        z1 = np.tanh(np.matmul(tile_node_embedding, paras_dict['W11']) + np.matmul(in_z, paras_dict['W12']) + paras_dict['b1'])
        z2 = np.tanh(np.matmul(z1, paras_dict['W2']) + paras_dict['b2'])
        scores = np.tanh(np.matmul(z2, paras_dict['w_vec']) + paras_dict['b_scalar'])
        scores = np.reshape(scores, newshape=[-1])
        recom_list = np.argsort(-1 * scores)

        recom_dict[node] = recom_list

    return truth_dict, recom_dict

def node_recomm_SiNE(train_links_df, test_links_df, emb, paras_dict):
    test_links_df_pos = test_links_df[test_links_df['sign'] == 1]

    # construct ground truth for node recommendation
    truth_dict = {}
    original_nodes = test_links_df_pos['fromIdx'].values
    for node in original_nodes:
        truth_nodes_list = test_links_df_pos[test_links_df_pos['fromIdx'] == node]['toIdx'].values
        truth_dict[node] = truth_nodes_list

    # make recommendation list
    recom_dict = {}
    for node in truth_dict:
        node_embedding = np.reshape(emb[node], newshape=[1, -1])
        tile_node_embedding = np.tile(node_embedding, [emb.shape[0], 1])
        z1 = np.tanh(np.matmul(tile_node_embedding, paras_dict['W11']) + np.matmul(emb, paras_dict['W12']) + paras_dict['b1'])
        z2 = np.tanh(np.matmul(z1, paras_dict['W2']) + paras_dict['b2'])
        scores = np.tanh(np.matmul(z2, paras_dict['w_vec']) + paras_dict['b_scalar'])
        scores = np.reshape(scores, newshape=[-1])
        recom_list = np.argsort(-1 * scores)

        recom_dict[node] = recom_list

    return truth_dict, recom_dict

def node_recomm_side(train_links_df, test_links_df, out_z, in_z, b_out_pos, b_in_pos):

    test_links_df_pos = test_links_df[test_links_df['sign']==1]

    # construct ground truth for node recommendation
    truth_dict = {}
    original_nodes = test_links_df_pos['fromIdx'].values
    for node in original_nodes:
        truth_nodes_list = test_links_df_pos[test_links_df_pos['fromIdx']==node]['toIdx'].values
        truth_dict[node] = truth_nodes_list

    # make recommendation list
    recom_dict = {}
    for node in truth_dict:
        node_embedding = np.reshape(out_z[node], newshape=[1, -1])
        scores = np.sum(np.tile(node_embedding, [in_z.shape[0], 1])*in_z+b_out_pos[node]+b_in_pos[node], axis=1)
        # scores = model.predict_proba(X=np.concatenate([np.tile(node_embedding, [in_z.shape[0], 1]), in_z], axis=1))[:, 1]
        recom_list = np.argsort(-1*scores)

        recom_dict[node] = recom_list

    return truth_dict, recom_dict

def node_recomm(train_links_df, test_links_df, out_z, in_z, topN=10):

    test_links_df_pos = test_links_df[test_links_df['sign']==1]

    # construct ground truth for node recommendation
    truth_dict = {}
    original_nodes = test_links_df_pos['fromIdx'].values
    for node in original_nodes:
        truth_nodes_list = test_links_df_pos[test_links_df_pos['fromIdx']==node]['toIdx'].values
        truth_dict[node] = truth_nodes_list

    # make recommendation list
    recom_dict = {}
    for node in truth_dict:
        node_embedding = np.reshape(out_z[node], newshape=[1, -1])
        scores = np.sum(np.tile(node_embedding, [in_z.shape[0], 1])*in_z, axis=1)
        # scores = model.predict_proba(X=np.concatenate([np.tile(node_embedding, [in_z.shape[0], 1]), in_z], axis=1))[:, 1]
        recom_list = np.argsort(-1*scores)

        recom_dict[node] = recom_list

    return truth_dict, recom_dict


def link_direction_prediction(train_links_df, test_links_df, out_z, in_z):
    # construct training examples for link direction prediction task
    train_fromIdx_pos = train_links_df['fromIdx'].values
    train_toIdx_pos = train_links_df['toIdx'].values

    train_from_embeddings_pos = out_z[train_fromIdx_pos]
    train_to_embeddings_pos = in_z[train_toIdx_pos]

    train_direction_pos_embeddings = np.concatenate([train_from_embeddings_pos, train_to_embeddings_pos], axis=1)

    train_from_embeddings_neg = out_z[train_toIdx_pos]
    train_to_embeddings_neg = in_z[train_fromIdx_pos]
    train_direction_neg_embeddings = np.concatenate([train_from_embeddings_neg, train_to_embeddings_neg], axis=1)

    train_direction_embeddings = np.concatenate([train_direction_pos_embeddings, train_direction_neg_embeddings], axis=0)

    train_direction_labels_pos = np.ones(shape=train_fromIdx_pos.shape[0])
    train_direction_labels_neg = np.zeros(shape=train_fromIdx_pos.shape[0])
    train_direction_labels = np.concatenate([train_direction_labels_pos, train_direction_labels_neg], axis=0)

    # construct testing examples for link direction prediction task
    test_fromIdx_pos = test_links_df['fromIdx'].values
    test_toIdx_pos = test_links_df['toIdx'].values

    test_from_embeddings_pos = out_z[test_fromIdx_pos]
    test_to_embeddings_pos = in_z[test_toIdx_pos]

    test_direction_pos_embeddings = np.concatenate([test_from_embeddings_pos, test_to_embeddings_pos], axis=1)

    test_from_embeddings_neg = out_z[test_toIdx_pos]
    test_to_embeddings_neg = in_z[test_fromIdx_pos]
    test_direction_neg_embeddings = np.concatenate([test_from_embeddings_neg, test_to_embeddings_neg], axis=1)

    test_direction_embeddings = np.concatenate([test_direction_pos_embeddings, test_direction_neg_embeddings],
                                                axis=0)

    test_direction_labels_pos = np.ones(shape=test_fromIdx_pos.shape[0])
    test_direction_labels_neg = np.zeros(shape=test_fromIdx_pos.shape[0])
    test_direction_labels = np.concatenate([test_direction_labels_pos, test_direction_labels_neg], axis=0)

    # construct logistic model and return auc, f1
    model = LogisticRegression()
    model.fit(X=train_direction_embeddings, y=train_direction_labels)

    preds_score = model.predict_proba(X=test_direction_embeddings)[:, 1]
    preds_label = model.predict(X=test_direction_embeddings)

    auc_value = roc_auc_score(test_direction_labels, preds_score)
    f1_value = f1_score(test_direction_labels, preds_label)

    return auc_value, f1_value

def link_sign_prediction(train_links_df, test_links_df, out_z, in_z):
    # link sign prediction data for training
    train_fromIdx = train_links_df['fromIdx'].values
    train_toIdx = train_links_df['toIdx'].values
    train_link_sign = train_links_df['sign'].values
    train_link_sign[train_link_sign == -1] = 0

    train_from_embeddings = out_z[train_fromIdx]
    train_to_embeddings = in_z[train_toIdx]

    # link sign prediction data for testing
    test_fromIdx = test_links_df['fromIdx'].values
    test_toIdx = test_links_df['toIdx'].values
    test_link_sign = test_links_df['sign'].values
    test_link_sign[test_link_sign == -1] = 0

    test_from_embeddings = out_z[test_fromIdx]
    test_to_embeddings = in_z[test_toIdx]

    # construct logistic model and return auc, f1
    model = LogisticRegression()
    model.fit(X=np.concatenate([train_from_embeddings, train_to_embeddings], axis=1), y=train_link_sign)

    preds_score = model.predict_proba(X=np.concatenate([test_from_embeddings, test_to_embeddings], axis=1))[:, 1]
    preds_label = model.predict(X=np.concatenate([test_from_embeddings, test_to_embeddings], axis=1))

    auc_value = roc_auc_score(test_link_sign, preds_score)
    f1_value = f1_score(test_link_sign, preds_label)
    return auc_value, f1_value

def unif_weight_init(shape, name=None, uniform_value=None):
    if uniform_value==None:
        initial = tf.random_uniform(shape, minval=-np.sqrt(6.0/(shape[0]+shape[1])), maxval=np.sqrt(6.0/(shape[0]+shape[1])), dtype=tf.float32)
    else:
        initial = tf.random_uniform(shape, minval=-uniform_value, maxval=uniform_value, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def sample_gaussian(mean, diag_cov):

    z = mean+tf.random_normal(tf.shape(diag_cov))*diag_cov

    return z


def sample_gaussian_np(mean, diag_cov):

    z = mean+np.random.normal(size=diag_cov.shape)*diag_cov

    return z


def gcn_layer_id(norm_adj_mat, h=None, W=None, b=None, h_sparse=True):
    if h is not None:
        if h_sparse:
            return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, tf.sparse_tensor_dense_matmul(h, W)), b))
        else:
            return tf.nn.relu(tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b))
    else:
        return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, W), b))


def gcn_layer(norm_adj_mat, h=None, W=None, b=None):
    if h is None:
        return tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, tf.sparse_tensor_dense_matmul(W, h)), b)
    else:
        return tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)


def sigmoid(x):

    return 1.0/(1.0+np.exp(-x))