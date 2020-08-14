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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import File_Reader
import utils
import time
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression



class BPWR(object):
    def __init__(self, sess, n_nodes, args):

        self.sess = sess
        self.result_dir = args.result_dir
        self.n_nodes = n_nodes
        self.n_embedding = args.n_embedding
        self.dropout = args.dropout
        self.learning_rate = args.learning_rate
        self.max_iteration = args.max_iteration
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.shape = np.array([self.n_nodes, self.n_nodes])

        self.train_triplets = tf.placeholder(tf.int32, name='train_triplets')
        self.keep_prob = tf.placeholder(tf.float32)

        self._build_VGAE()

    def _build_VGAE(self):
        # construct node out embeddings and in embeddings
        self.out_z = utils.unif_weight_init(shape=[self.n_nodes, self.n_embedding])
        self.in_z = utils.unif_weight_init(shape=[self.n_nodes, self.n_embedding])

        # sample pos edges and neg edges for training
        self.triplets_i = tf.nn.embedding_lookup(self.out_z, self.train_triplets[:, 0])
        self.triplets_j = tf.nn.embedding_lookup(self.in_z, self.train_triplets[:, 1])
        self.triplets_k = tf.nn.embedding_lookup(self.in_z, self.train_triplets[:, 2])

        self.close_pair = tf.reduce_sum(self.triplets_i*self.triplets_j, axis=1)
        self.distant_pair = tf.reduce_sum(self.triplets_i*self.triplets_k, axis=1)

        # closer pair obtain more higher scores
        self.balance_loss = -1*tf.reduce_mean(tf.log(tf.sigmoid(self.close_pair - self.distant_pair) + 1e-24))
        self.kl_loss = tf.constant(0.0)

        self.loss = self.balance_loss + self.kl_loss

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self, args, feed_data, train_links_df=None, test_links_df=None, topN_list=None):

        # feed_data = shuffle(feed_data)

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
        time_list = []
        batch_num = int(np.ceil(float(len(feed_data)) / self.batch_size))
        for i in range(self.max_iteration):
            shuffle(feed_data)
            start_idx, end_idx = 0, 0
            epoch_loss_list = []
            balance_loss_list = []
            reg_kl_loss_list = []

            temp_time = 0.0
            for j in range(batch_num):
                start_idx = end_idx
                end_idx = min(start_idx + self.batch_size, len(feed_data))
                batch_feed_data = feed_data[start_idx:end_idx]

                feed_dict = {self.train_triplets:batch_feed_data,
                             self.keep_prob: args.keep_prob}

                time1 = time.time()
                _, loss, balance_loss, kl_loss, out_z, in_z = self.sess.run([self.train_step, self.loss,
                                                                                   self.balance_loss, self.kl_loss,
                                                                                   self.out_z, self.in_z],
                                                                                   feed_dict=feed_dict)
                time2 = time.time()
                temp_time += time2-time1

                # print('loss: {}, balance loss: {}, kl_loss: {}'.format(loss, balance_loss, kl_loss))
                epoch_loss_list.append(loss)
                balance_loss_list.append(balance_loss)
                reg_kl_loss_list.append(kl_loss)
                if np.isnan(loss):
                    print('Loss NAN Error!')
                    raise Exception

            time_list.append(temp_time)
            epoch_loss = np.mean(epoch_loss_list)
            epoch_balance_loss = np.mean(balance_loss_list)
            epoch_kl_loss = np.mean(reg_kl_loss_list)
            print('epoch: {}, loss: {:.6f}, balance loss: {:.6f}, reg-kl: {:.6f}'.format(i, loss, epoch_balance_loss, epoch_kl_loss))

            if type(train_links_df) != type(None) and type(test_links_df) != type(None):
                print('###############################################################')
                '''
                link sign prediction
                '''
                auc_sign, f1_sign = utils.link_sign_prediction(train_links_df, test_links_df, out_z, in_z)
                auc_sign_lsit.append(auc_sign)
                f1_sign_list.append(f1_sign)
                print('link sign prediction task=>>>>>>auc sign: {:.6f}, f1 sign: {:.6f}'.format(auc_sign, f1_sign))

                '''
                node recommendation task
                '''
                truth_dict, recom_dict = utils.node_recomm(train_links_df, test_links_df, out_z, in_z)
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

                # save best embedding according to the AUC on link sign prediction
                if auc_sign > best_auc:
                    best_auc = auc_sign
                    np.savetxt(os.path.join(self.result_dir, 'BPWR', '{}_out_embedding.txt'.format(self.dataset)), out_z)
                    np.savetxt(os.path.join(self.result_dir, 'BPWR', '{}_in_embedding.txt'.format(self.dataset)), in_z)
                else:
                    pass

            if last_epoch_loss and np.abs(last_epoch_loss - epoch_loss)/last_epoch_loss <= 1e-4:
                # np.savetxt(os.path.join(self.result_dir, '{}_out_embedding.txt'.format(self.dataset)), out_z)
                # np.savetxt(os.path.join(self.result_dir, '{}_in_embedding.txt'.format(self.dataset)), in_z)
                break
            else:
                last_epoch_loss = epoch_loss

        print('cost time: {:.4f}'.format(np.mean(time_list)))
        return auc_sign_lsit, f1_sign_list, recall5_list, recall10_list, recall20_list, precision5_list, precision10_list, precision20_list

