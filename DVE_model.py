"""
Tensorflow is really really really garbage, rubbish !!!!!!!!!!!!!!!!!!!!!!!
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
FUCK YOU, TENSORFLOW
.......
"""

import os
import tensorflow as tf
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import File_Reader
import utils
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression



class VGE_EDGE(object):
    def __init__(self, sess, n_nodes, args):

        self.sess = sess
        self.result_dir = args.result_dir
        self.n_nodes = n_nodes
        self.n_gcn_layer = args.n_gcn_layer
        self.n_hidden_list = args.n_hidden_list
        self.n_hidden_list = [self.n_nodes] + self.n_hidden_list
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.max_iteration = args.max_iteration
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.shape = np.array([self.n_nodes, self.n_nodes])

        self.norm_sign_pos_sp = tf.sparse_placeholder(tf.float32, shape=self.shape, name='norm_sign_pos_sp')
        self.norm_sign_neg_sp = tf.sparse_placeholder(tf.float32, shape=self.shape, name='norm_sign_neg_sp')
        self.pos_degree = tf.placeholder(tf.float32, name='pos_degree')
        self.neg_degree = tf.placeholder(tf.float32, name='neg_degree')
        self.train_triplets = tf.placeholder(tf.int32, name='train_triplets')
        self.keep_prob = tf.placeholder(tf.float32)

        self._build_VGAE()

    def _build_VGAE(self):

        """
        1.build for out_pos graph
        """
        self.out_pos_W_mu_params = {}
        self.out_pos_b_mu_params = {}

        self.out_pos_W_sigma_params = {}
        self.out_pos_b_sigma_params = {}

        for n_layer in range(self.n_gcn_layer):
            self.out_pos_W_mu_params['out_pos_W_{}_mu'.format(n_layer)] = utils.unif_weight_init(shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer+1]])
            self.out_pos_b_mu_params['out_pos_b_{}_mu'.format(n_layer)] = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer+1]]))

            self.out_pos_W_sigma_params['out_pos_W_{}_sigma'.format(n_layer)] = utils.unif_weight_init(shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer+1]])
            self.out_pos_b_sigma_params['out_pos_b_{}_sigma'.format(n_layer)] = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer+1]]))


        self.out_pos_hidden_mu = {}
        self.out_pos_hidden_mu['out_pos_hidden_0_mu'] = tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp, W=self.out_pos_W_mu_params['out_pos_W_0_mu'],
                                   b=self.out_pos_b_mu_params['out_pos_b_0_mu']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer-2):
            self.out_pos_hidden_mu['out_pos_hidden_{}_mu'.format(n_layer+1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp, h=self.out_pos_hidden_mu['out_pos_hidden_{}_mu'.format(n_layer)],
                                                 W=self.out_pos_W_mu_params['out_pos_W_{}_mu'.format(n_layer+1)],
                                                b=self.out_pos_b_mu_params['out_pos_b_{}_mu'.format(n_layer+1)], h_sparse=False),
                                                self.keep_prob)


        self.out_pos_mu = utils.gcn_layer(norm_adj_mat=self.norm_sign_pos_sp,
                                          h=self.out_pos_hidden_mu['out_pos_hidden_{}_mu'.format(self.n_gcn_layer-2)],
                                          W=self.out_pos_W_mu_params['out_pos_W_{}_mu'.format(self.n_gcn_layer-1)],
                                          b=self.out_pos_b_mu_params['out_pos_b_{}_mu'.format(self.n_gcn_layer-1)])

        self.out_pos_hidden_log_sigma = {}
        self.out_pos_hidden_log_sigma['out_pos_hidden_0_sigma'] = tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp,
                                            W=self.out_pos_W_sigma_params['out_pos_W_0_sigma'],
                                            b=self.out_pos_b_sigma_params['out_pos_b_0_sigma']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.out_pos_hidden_log_sigma['out_pos_hidden_{}_sigma'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp,
                                              h=self.out_pos_hidden_log_sigma['out_pos_hidden_{}_sigma'.format(n_layer)],
                                              W=self.out_pos_W_sigma_params['out_pos_W_{}_sigma'.format(n_layer + 1)],
                                              b=self.out_pos_b_sigma_params['out_pos_b_{}_sigma'.format(n_layer + 1)], h_sparse=False),
                                                self.keep_prob)

        self.out_pos_log_sigma = utils.gcn_layer(norm_adj_mat=self.norm_sign_pos_sp,
                                          h=self.out_pos_hidden_log_sigma['out_pos_hidden_{}_sigma'.format(self.n_gcn_layer-2)],
                                          W=self.out_pos_W_sigma_params['out_pos_W_{}_sigma'.format(self.n_gcn_layer-1)],
                                          b=self.out_pos_b_sigma_params['out_pos_b_{}_sigma'.format(self.n_gcn_layer-1)])


        self.out_pos_sigma = tf.exp(self.out_pos_log_sigma)

        out_pos_z = utils.sample_gaussian(self.out_pos_mu, self.out_pos_sigma)
        test_out_pos_z = self.out_pos_mu

        """
        2.build for out_neg graph
        """
        self.out_neg_W_mu_params = {}
        self.out_neg_b_mu_params = {}

        self.out_neg_W_sigma_params = {}
        self.out_neg_b_sigma_params = {}

        for n_layer in range(self.n_gcn_layer):
            self.out_neg_W_mu_params['out_neg_W_{}_mu'.format(n_layer)] = utils.unif_weight_init(
                shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.out_neg_b_mu_params['out_neg_b_{}_mu'.format(n_layer)] = tf.Variable(
                tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

            self.out_neg_W_sigma_params['out_neg_W_{}_sigma'.format(n_layer)] = utils.unif_weight_init(
                    shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.out_neg_b_sigma_params['out_neg_b_{}_sigma'.format(n_layer)] = tf.Variable(
                    tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

        self.out_neg_hidden_mu = {}
        self.out_neg_hidden_mu['out_neg_hidden_0_mu'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp, W=self.out_neg_W_mu_params['out_neg_W_0_mu'],
                               b=self.out_neg_b_mu_params['out_neg_b_0_mu']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.out_neg_hidden_mu['out_neg_hidden_{}_mu'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                                                 h=self.out_neg_hidden_mu['out_neg_hidden_{}_mu'.format(n_layer)],
                                                 W=self.out_neg_W_mu_params['out_neg_W_{}_mu'.format(n_layer + 1)],
                                                 b=self.out_neg_b_mu_params['out_neg_b_{}_mu'.format(n_layer + 1)],
                                                 h_sparse=False), self.keep_prob)

        self.out_neg_mu = utils.gcn_layer(norm_adj_mat=self.norm_sign_neg_sp,
                                          h=self.out_neg_hidden_mu['out_neg_hidden_{}_mu'.format(self.n_gcn_layer - 2)],
                                          W=self.out_neg_W_mu_params['out_neg_W_{}_mu'.format(self.n_gcn_layer - 1)],
                                          b=self.out_neg_b_mu_params['out_neg_b_{}_mu'.format(self.n_gcn_layer - 1)])

        self.out_neg_hidden_log_sigma = {}
        self.out_neg_hidden_log_sigma['out_neg_hidden_0_sigma'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                               W=self.out_neg_W_sigma_params['out_neg_W_0_sigma'],
                               b=self.out_neg_b_sigma_params['out_neg_b_0_sigma']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.out_neg_hidden_log_sigma['out_neg_hidden_{}_sigma'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                                                 h=self.out_neg_hidden_log_sigma[
                                                     'out_neg_hidden_{}_sigma'.format(n_layer)],
                                                 W=self.out_neg_W_sigma_params[
                                                     'out_neg_W_{}_sigma'.format(n_layer + 1)],
                                                 b=self.out_neg_b_sigma_params[
                                                     'out_neg_b_{}_sigma'.format(n_layer + 1)], h_sparse=False), self.keep_prob)

        self.out_neg_log_sigma = utils.gcn_layer(norm_adj_mat=self.norm_sign_neg_sp,
                                                 h=self.out_neg_hidden_log_sigma[
                                                     'out_neg_hidden_{}_sigma'.format(self.n_gcn_layer - 2)],
                                                 W=self.out_neg_W_sigma_params[
                                                     'out_neg_W_{}_sigma'.format(self.n_gcn_layer - 1)],
                                                 b=self.out_neg_b_sigma_params[
                                                     'out_neg_b_{}_sigma'.format(self.n_gcn_layer - 1)])

        self.out_neg_sigma = tf.exp(self.out_neg_log_sigma)

        out_neg_z = utils.sample_gaussian(self.out_neg_mu, self.out_neg_sigma)
        test_out_neg_z = self.out_neg_mu

        """
        3.build for in_pos graph
        """
        self.in_pos_W_mu_params = {}
        self.in_pos_b_mu_params = {}

        self.in_pos_W_sigma_params = {}
        self.in_pos_b_sigma_params = {}

        for n_layer in range(self.n_gcn_layer):
            self.in_pos_W_mu_params['in_pos_W_{}_mu'.format(n_layer)] = utils.unif_weight_init(
                shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.in_pos_b_mu_params['in_pos_b_{}_mu'.format(n_layer)] = tf.Variable(
                tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

            self.in_pos_W_sigma_params['in_pos_W_{}_sigma'.format(n_layer)] = utils.unif_weight_init(
                    shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.in_pos_b_sigma_params['in_pos_b_{}_sigma'.format(n_layer)] = tf.Variable(
                    tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

        self.in_pos_hidden_mu = {}
        self.in_pos_hidden_mu['in_pos_hidden_0_mu'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp, W=self.in_pos_W_mu_params['in_pos_W_0_mu'],
                               b=self.in_pos_b_mu_params['in_pos_b_0_mu']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.in_pos_hidden_mu['in_pos_hidden_{}_mu'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp,
                                                 h=self.in_pos_hidden_mu['in_pos_hidden_{}_mu'.format(n_layer)],
                                                 W=self.in_pos_W_mu_params['in_pos_W_{}_mu'.format(n_layer + 1)],
                                                 b=self.in_pos_b_mu_params['in_pos_b_{}_mu'.format(n_layer + 1)],
                                                 h_sparse=False), self.keep_prob)

        self.in_pos_mu = utils.gcn_layer(norm_adj_mat=self.norm_sign_pos_sp,
                                          h=self.in_pos_hidden_mu['in_pos_hidden_{}_mu'.format(self.n_gcn_layer - 2)],
                                          W=self.in_pos_W_mu_params['in_pos_W_{}_mu'.format(self.n_gcn_layer - 1)],
                                          b=self.in_pos_b_mu_params['in_pos_b_{}_mu'.format(self.n_gcn_layer - 1)])

        self.in_pos_hidden_log_sigma = {}
        self.in_pos_hidden_log_sigma['in_pos_hidden_0_sigma'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp,
                               W=self.in_pos_W_sigma_params['in_pos_W_0_sigma'],
                               b=self.in_pos_b_sigma_params['in_pos_b_0_sigma']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.in_pos_hidden_log_sigma['in_pos_hidden_{}_sigma'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_pos_sp,
                                                 h=self.in_pos_hidden_log_sigma[
                                                     'in_pos_hidden_{}_sigma'.format(n_layer)],
                                                 W=self.in_pos_W_sigma_params[
                                                     'in_pos_W_{}_sigma'.format(n_layer + 1)],
                                                 b=self.in_pos_b_sigma_params[
                                                     'in_pos_b_{}_sigma'.format(n_layer + 1)], h_sparse=False), self.keep_prob)

        self.in_pos_log_sigma = utils.gcn_layer(norm_adj_mat=self.norm_sign_pos_sp,
                                                 h=self.in_pos_hidden_log_sigma[
                                                     'in_pos_hidden_{}_sigma'.format(self.n_gcn_layer - 2)],
                                                 W=self.in_pos_W_sigma_params[
                                                     'in_pos_W_{}_sigma'.format(self.n_gcn_layer - 1)],
                                                 b=self.in_pos_b_sigma_params[
                                                     'in_pos_b_{}_sigma'.format(self.n_gcn_layer - 1)])

        self.in_pos_sigma = tf.exp(self.in_pos_log_sigma)

        in_pos_z = utils.sample_gaussian(self.in_pos_mu, self.in_pos_sigma)
        test_in_pos_z = self.in_pos_mu


        """
        4.build for in_neg graph
        """
        self.in_neg_W_mu_params = {}
        self.in_neg_b_mu_params = {}

        self.in_neg_W_sigma_params = {}
        self.in_neg_b_sigma_params = {}

        for n_layer in range(self.n_gcn_layer):
            self.in_neg_W_mu_params['in_neg_W_{}_mu'.format(n_layer)] = utils.unif_weight_init(
                shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.in_neg_b_mu_params['in_neg_b_{}_mu'.format(n_layer)] = tf.Variable(
                tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

            self.in_neg_W_sigma_params['in_neg_W_{}_sigma'.format(n_layer)] = utils.unif_weight_init(
                    shape=[self.n_hidden_list[n_layer], self.n_hidden_list[n_layer + 1]])
            self.in_neg_b_sigma_params['in_neg_b_{}_sigma'.format(n_layer)] = tf.Variable(
                    tf.constant(0.01, dtype=tf.float32, shape=[self.n_hidden_list[n_layer + 1]]))

        self.in_neg_hidden_mu = {}
        self.in_neg_hidden_mu['in_neg_hidden_0_mu'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp, W=self.in_neg_W_mu_params['in_neg_W_0_mu'],
                               b=self.in_neg_b_mu_params['in_neg_b_0_mu']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.in_neg_hidden_mu['in_neg_hidden_{}_mu'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                                                 h=self.in_neg_hidden_mu['in_neg_hidden_{}_mu'.format(n_layer)],
                                                 W=self.in_neg_W_mu_params['in_neg_W_{}_mu'.format(n_layer + 1)],
                                                 b=self.in_neg_b_mu_params['in_neg_b_{}_mu'.format(n_layer + 1)],
                                                 h_sparse=False), self.keep_prob)

        self.in_neg_mu = utils.gcn_layer(norm_adj_mat=self.norm_sign_neg_sp,
                                         h=self.in_neg_hidden_mu['in_neg_hidden_{}_mu'.format(self.n_gcn_layer - 2)],
                                         W=self.in_neg_W_mu_params['in_neg_W_{}_mu'.format(self.n_gcn_layer - 1)],
                                         b=self.in_neg_b_mu_params['in_neg_b_{}_mu'.format(self.n_gcn_layer - 1)])

        self.in_neg_hidden_log_sigma = {}
        self.in_neg_hidden_log_sigma['in_neg_hidden_0_sigma'] = tf.nn.dropout(
            utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                               W=self.in_neg_W_sigma_params['in_neg_W_0_sigma'],
                               b=self.in_neg_b_sigma_params['in_neg_b_0_sigma']), self.keep_prob)

        for n_layer in range(self.n_gcn_layer - 2):
            self.in_neg_hidden_log_sigma['in_neg_hidden_{}_sigma'.format(n_layer + 1)] = \
                tf.nn.dropout(utils.gcn_layer_id(norm_adj_mat=self.norm_sign_neg_sp,
                                                 h=self.in_neg_hidden_log_sigma[
                                                     'in_neg_hidden_{}_sigma'.format(n_layer)],
                                                 W=self.in_neg_W_sigma_params[
                                                     'in_neg_W_{}_sigma'.format(n_layer + 1)],
                                                 b=self.in_neg_b_sigma_params[
                                                     'in_neg_b_{}_sigma'.format(n_layer + 1)], h_sparse=False),
                              self.keep_prob)

        self.in_neg_log_sigma = utils.gcn_layer(norm_adj_mat=self.norm_sign_neg_sp,
                                                h=self.in_neg_hidden_log_sigma[
                                                    'in_neg_hidden_{}_sigma'.format(self.n_gcn_layer - 2)],
                                                W=self.in_neg_W_sigma_params[
                                                    'in_neg_W_{}_sigma'.format(self.n_gcn_layer - 1)],
                                                b=self.in_neg_b_sigma_params[
                                                    'in_neg_b_{}_sigma'.format(self.n_gcn_layer - 1)])

        self.in_neg_sigma = tf.exp(self.in_neg_log_sigma)

        in_neg_z = utils.sample_gaussian(self.in_neg_mu, self.in_neg_sigma)
        test_in_neg_z = self.in_neg_mu


        # concat
        # construct node out embeddings and in embeddings
        self.out_z = tf.concat([out_pos_z, out_neg_z], axis=1)
        self.in_z = tf.concat([in_pos_z, in_neg_z], axis=1)

        self.test_out_z = tf.concat([test_out_pos_z, test_out_neg_z], axis=1)
        self.test_in_z = tf.concat([test_in_pos_z, test_in_neg_z], axis=1)

        # # concat+MLP
        # self.MLP_out = utils.unif_weight_init(shape=[out_pos_z.shape[1].value*2, out_pos_z.shape[1].value*2])
        # self.MLP_in = utils.unif_weight_init(shape=[in_pos_z.shape[1].value * 2, in_pos_z.shape[1].value * 2])
        #
        # self.out_z = tf.matmul(tf.concat([out_pos_z, out_neg_z], axis=1), self.MLP_out)
        # self.in_z = tf.matmul(tf.concat([in_pos_z, in_neg_z], axis=1), self.MLP_in)
        #
        # self.test_out_z = tf.matmul(tf.concat([test_out_pos_z, test_out_neg_z], axis=1), self.MLP_out)
        # self.test_in_z = tf.matmul(tf.concat([test_in_pos_z, test_in_neg_z], axis=1), self.MLP_in)

        # # inner product
        # self.out_z = out_pos_z*out_neg_z
        # self.in_z = in_pos_z*in_neg_z
        #
        # self.test_out_z = test_out_pos_z*test_out_neg_z
        # self.test_in_z = test_in_pos_z*test_in_neg_z
        #
        # # inner product+MLP
        # self.MLP_out = utils.unif_weight_init(shape=[out_pos_z.shape[1].value, out_pos_z.shape[1].value])
        # self.MLP_in = utils.unif_weight_init(shape=[in_pos_z.shape[1].value, in_pos_z.shape[1].value])
        #
        # self.out_z = tf.matmul(out_pos_z*out_neg_z, self.MLP_out)
        # self.in_z = tf.matmul(in_pos_z*in_neg_z, self.MLP_in)
        #
        # self.test_out_z = tf.matmul(test_out_pos_z*test_out_neg_z, self.MLP_out)
        # self.test_in_z = tf.matmul(test_in_pos_z*test_in_neg_z, self.MLP_in)

        # sample pos edges and neg edges for training
        self.triplets_i = tf.nn.embedding_lookup(self.out_z, self.train_triplets[:, 0])
        self.triplets_j = tf.nn.embedding_lookup(self.in_z, self.train_triplets[:, 1])
        self.triplets_k = tf.nn.embedding_lookup(self.in_z, self.train_triplets[:, 2])

        self.close_pair = tf.reduce_sum(self.triplets_i*self.triplets_j, axis=1)
        self.distant_pair = tf.reduce_sum(self.triplets_i*self.triplets_k, axis=1)

        # closer pair obtain more higher scores
        self.balance_loss = -1*tf.reduce_mean(tf.log(tf.sigmoid(self.close_pair - self.distant_pair) + 1e-12))

        # calculate corresponding loss
        self.kl_out_pos = self.reg_kl(self.out_pos_mu, self.out_pos_sigma, self.n_nodes)
        self.kl_out_neg = self.reg_kl(self.out_neg_mu, self.out_neg_sigma, self.n_nodes)
        self.kl_in_pos = self.reg_kl(self.in_pos_mu, self.in_pos_sigma, self.n_nodes)
        self.kl_in_neg = self.reg_kl(self.in_neg_mu, self.in_neg_sigma, self.n_nodes)

        self.kl_loss = self.kl_out_pos+self.kl_out_neg+self.kl_in_pos+self.kl_in_neg

        self.loss = self.balance_loss + self.kl_loss

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def reg_kl(self, mu, sigma, mean_num):
        return -(0.5 / mean_num) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(sigma + 1e-12) - tf.square(mu) - tf.square(sigma), 1))

    def train(self, args, train_sign_pos_sp, train_sign_neg_sp, feed_data, train_links_df=None, test_links_df=None, topN_list=None):

        # feed_data = shuffle(feed_data)

        norm_train_sign_pos_sp = File_Reader.normalize_adjacency(train_sign_pos_sp)
        norm_train_sign_neg_sp = File_Reader.normalize_adjacency(train_sign_neg_sp)

        pos_degree = File_Reader.get_degree(train_sign_pos_sp)
        neg_degree = File_Reader.get_degree(train_sign_neg_sp)

        last_epoch_loss = None
        auc_sign_lsit = []
        f1_sign_list = []

        recall5_list = []
        recall10_list = []
        recall20_list = []
        precision5_list = []
        precision10_list = []
        precision20_list = []
        best_auc = 0.0
        batch_num = int(np.ceil(float(len(feed_data)) / self.batch_size))
        epoch_loss_list = []

        time_list = []
        for i in range(self.max_iteration):
            shuffle(feed_data)
            start_idx, end_idx = 0, 0
            batch_loss_list = []
            balance_loss_list = []
            reg_kl_loss_list = []

            temp_time = 0.0
            for j in range(batch_num):
                start_idx = end_idx
                end_idx = min(start_idx + self.batch_size, len(feed_data))
                batch_feed_data = feed_data[start_idx:end_idx]

                feed_dict = {self.norm_sign_pos_sp: norm_train_sign_pos_sp[0:2],
                             self.norm_sign_neg_sp: norm_train_sign_neg_sp[0:2],
                             self.pos_degree: pos_degree,
                             self.neg_degree: neg_degree,
                             self.train_triplets:batch_feed_data,
                             self.keep_prob: args.keep_prob}

                time1 = time.time()
                # _, test_out_z, test_in_z = self.sess.run([self.train_step, self.test_out_z, self.test_in_z],
                #                                                                       feed_dict=feed_dict)

                _, loss, balance_loss, kl_loss, test_out_z, test_in_z = self.sess.run([self.train_step, self.loss,
                                                                                   self.balance_loss, self.kl_loss,
                                                                                   self.test_out_z, self.test_in_z],
                                                                                   feed_dict=feed_dict)
                time2 = time.time()
                temp_time += time2 - time1


                # print('loss: {}, balance loss: {}, kl_loss: {}'.format(loss, balance_loss, kl_loss))
                batch_loss_list.append(loss)
                balance_loss_list.append(balance_loss)
                reg_kl_loss_list.append(kl_loss)
                if np.isnan(loss):
                    print('Loss NAN Error!')
                    raise Exception

            time_list.append(temp_time)

            epoch_loss = np.mean(batch_loss_list)
            epoch_balance_loss = np.mean(balance_loss_list)
            epoch_loss_list.append(epoch_balance_loss)
            epoch_kl_loss = np.mean(reg_kl_loss_list)
            print('epoch: {}, loss: {:.4f}, balance loss: {:.4f}, reg-kl: {:.4f}'.format(i, loss, epoch_balance_loss, epoch_kl_loss))

            if type(train_links_df) != type(None) and type(test_links_df) != type(None):
                print('###############################################################')
                '''
                link sign prediction
                '''
                auc_sign, f1_sign = utils.link_sign_prediction(train_links_df, test_links_df, test_out_z, test_in_z)
                auc_sign_lsit.append(auc_sign)
                f1_sign_list.append(f1_sign)
                print('link sign prediction task=>>>>>>auc sign: {:.6f}, f1 sign: {:.6f}'.format(auc_sign, f1_sign))

                '''
                node recommendation task
                '''
                truth_dict, recom_dict = utils.node_recomm(train_links_df, test_links_df, test_out_z, test_in_z)
                recall5 = utils.eva_recall(truth_dict, recom_dict, topN=topN_list[0])
                precision5 = utils.eva_precision(truth_dict, recom_dict, topN=topN_list[0])

                recall10 = utils.eva_recall(truth_dict, recom_dict, topN=topN_list[1])
                precision10 = utils.eva_precision(truth_dict, recom_dict, topN=topN_list[1])

                recall20 = utils.eva_recall(truth_dict, recom_dict, topN=topN_list[2])
                precision20 = utils.eva_precision(truth_dict, recom_dict, topN=topN_list[2])

                recall5_list.append(recall5)
                recall10_list.append(recall10)
                recall20_list.append(recall20)
                precision5_list.append(precision5)
                precision10_list.append(precision10)
                precision20_list.append(precision20)


                print('node recommendation task=>>>>>>recall{}: {:.6f}, recall{}: {:.6f}, recall{}: {:.6f}, '
                      'precision{}: {:.6f}, precision{}: {:.6f}, precision{}: {:.6f}'.format(
                    topN_list[0], recall5, topN_list[1], recall10, topN_list[2], recall20,
                    topN_list[0], precision5, topN_list[1], precision10, topN_list[2], precision20))

                # # save best embedding according to the AUC on link sign prediction
                # if auc_sign > best_auc:
                #     best_auc = auc_sign
                #     np.savetxt(os.path.join(self.result_dir, 'DVE', '{}_out_embedding.txt'.format(self.dataset)), test_out_z)
                #     np.savetxt(os.path.join(self.result_dir, 'DVE', '{}_in_embedding.txt'.format(self.dataset)), test_in_z)
                # else:
                #     pass


            # if last_epoch_loss and np.abs(last_epoch_loss - epoch_loss)/last_epoch_loss <= 1e-4:
            #     break
            # else:
            #     last_epoch_loss = epoch_loss
        print('cost time: {:.4f}'.format(np.mean(time_list)))
        return auc_sign_lsit, f1_sign_list, recall5_list, recall10_list, recall20_list, precision5_list, precision10_list, precision20_list, epoch_loss_list

