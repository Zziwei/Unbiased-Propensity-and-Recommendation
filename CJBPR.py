import tensorflow as tf
import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from tqdm import tqdm
import math
from scipy.sparse import coo_matrix


class CJBPR:

    def __init__(self, sess, args, train_df, train_like, vali_like, test_user, test_interact, test_like, item_pop):
        self.sess = sess
        self.args = args

        self.num_item = args.num_item
        self.num_user = args.num_user

        self.hidden = args.hidden
        self.neg = args.neg
        self.batch_size = args.bs

        self.train_df = train_df

        self.C = args.C
        self.df_list = []
        len_train = len(train_df)
        df_len = int(len_train * 1. / self.C)
        left_idx = range(len_train)
        idx_list = []
        for i in range(self.C - 1):
            idx = np.random.choice(left_idx, int(df_len), replace=False).tolist()
            idx_list.append(idx)
            tmp_df = train_df.loc[idx]
            self.df_list.append(tmp_df)
            left_idx = list(set(left_idx) - set(idx))
        self.df_list.append(train_df.loc[left_idx])
        # idx_list.append(left_idx)
        # np.save('./' + args.data + '/CJBPR_idx_list.npy', np.array(idx_list))

        self.epoch = args.epoch

        self.lr = args.lr
        self.display = args.display
        self.reg = args.reg
        self.alpha = args.alpha
        self.beta = args.beta

        self.item_pop = item_pop.reshape((-1, 1))

        print('******************** CJBPR ********************')
        print(self.args)
        self._prepare_model()

        self.train_like = train_like
        self.vali_like = vali_like

        self.test_user = test_user
        self.test_interact = test_interact
        self.test_like = test_like

        print('********************* CJBPR Initialization Done *********************')

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(1, self.epoch + 1):
            self.train_model(epoch_itr)
            if epoch_itr % self.display == 0 and epoch_itr > 0:
                self.test_model(epoch_itr)


    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
            self.pop_input_pos = tf.placeholder(tf.float32, shape=[None, 1], name="pop_input_pos")
            self.pop_input_neg = tf.placeholder(tf.float32, shape=[None, 1], name="pop_input_neg")
            self.rel_input = tf.placeholder(tf.float32, shape=[None, 1], name="rel_input")
            self.exp_input = tf.placeholder(tf.float32, shape=[None, 1], name="exp_input")

        self.P_list = []
        self.Q_list = []
        self.c_list = []
        self.d_list = []
        self.a_list = []
        self.b_list = []
        self.e_list = []
        self.f_list = []
        self.para_rel_list = []
        self.para_exp_list = []
        for m in range(self.C):
            with tf.variable_scope('Relevance_' + str(m), reuse=tf.AUTO_REUSE):
                P = tf.get_variable(name='P_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_user, self.hidden],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                Q = tf.get_variable(name='Q_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_item, self.hidden],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
            para_rel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Relevance_' + str(m))
            self.P_list.append(P)
            self.Q_list.append(Q)
            self.para_rel_list.append(para_rel)

            with tf.variable_scope('Exposure_' + str(m), reuse=tf.AUTO_REUSE):
                c = tf.get_variable(name='c_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                d = tf.get_variable(name='d_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                a = tf.get_variable(name='a_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                b = tf.get_variable(name='b_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                e = tf.get_variable(name='e_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
                f = tf.get_variable(name='f_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.03,
                                                                    dtype=tf.float32), dtype=tf.float32)
            para_exp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Exposure_' + str(m))
            self.c_list.append(c)
            self.d_list.append(d)
            self.a_list.append(a)
            self.b_list.append(b)
            self.e_list.append(e)
            self.f_list.append(f)
            self.para_exp_list.append(para_exp)

        self.rel_cost_list = []
        self.exp_cost_list = []
        self.rel_reg_cost_list = []
        self.exp_reg_cost_list = []
        self.rel_cost_final_list = []
        self.exp_cost_final_list = []
        for m in range(self.C):
            P = self.P_list[m]
            Q = self.Q_list[m]
            c = self.c_list[m]
            d = self.d_list[m]
            a = self.a_list[m]
            b = self.b_list[m]
            e = self.e_list[m]
            f = self.f_list[m]

            p = tf.nn.embedding_lookup(P, tf.reshape(self.user_input, [-1]))
            q_pos = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_pos, [-1]))
            q_neg = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_neg, [-1]))

            rel_predict_pos = tf.reduce_sum(p * q_pos, 1, keepdims=True)

            rel_predict_neg = tf.reduce_sum(p * q_neg, 1, keepdims=True)

            w = tf.nn.sigmoid(tf.matmul(q_pos, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q_pos, c) + d) + (1 - w) * self.pop_input_pos,
                         tf.nn.sigmoid(tf.matmul(q_pos, e) + f))
            exp_predict_pos = pop
            exp_predict_pos = tf.clip_by_value(exp_predict_pos, clip_value_min=0.01, clip_value_max=0.99)

            w = tf.nn.sigmoid(tf.matmul(q_neg, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q_neg, c) + d) + (1 - w) * self.pop_input_neg,
                         tf.nn.sigmoid(tf.matmul(q_neg, e) + f))
            exp_predict_neg = pop
            exp_predict_neg = tf.clip_by_value(exp_predict_neg, clip_value_min=0.01, clip_value_max=0.99)

            rel_cost = -tf.reduce_mean(tf.log(tf.nn.sigmoid(rel_predict_pos - rel_predict_neg)) / self.exp_input)

            exp_cost = -tf.reduce_mean(tf.log(exp_predict_pos) / self.rel_input) \
                       - tf.reduce_mean(tf.log(1 - exp_predict_neg))

            rel_reg_cost = self.reg * 0.5 * (self.l2_norm(P) + self.l2_norm(Q))
            exp_reg_cost = self.alpha * self.reg * 0.5 * (self.l2_norm(c) + self.l2_norm(d)
                                                          + self.l2_norm(a) + self.l2_norm(b)
                                                          + self.l2_norm(e) + self.l2_norm(f))

            rel_cost_final = rel_cost + rel_reg_cost
            exp_cost_final = exp_cost + exp_reg_cost
            self.rel_cost_list.append(rel_cost)
            self.exp_cost_list.append(exp_cost)
            self.rel_reg_cost_list.append(rel_reg_cost)
            self.exp_reg_cost_list.append(exp_reg_cost)
            self.rel_cost_final_list.append(rel_cost_final)
            self.exp_cost_final_list.append(exp_cost_final)

        self.rP_list = []
        self.rQ_list = []
        self.rc_list = []
        self.rd_list = []
        self.ra_list = []
        self.rb_list = []
        self.re_list = []
        self.rf_list = []
        self.rpara_rel_list = []
        self.rpara_exp_list = []
        for m in range(self.C):
            with tf.variable_scope('rRelevance_' + str(m), reuse=tf.AUTO_REUSE):
                P = tf.get_variable(name='rP_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_user, self.hidden],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                Q = tf.get_variable(name='rQ_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.num_item, self.hidden],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
            rpara_rel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rRelevance_' + str(m))
            self.rP_list.append(P)
            self.rQ_list.append(Q)
            self.rpara_rel_list.append(rpara_rel)
            with tf.variable_scope('rExposure_' + str(m), reuse=tf.AUTO_REUSE):
                c = tf.get_variable(name='rc_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                d = tf.get_variable(name='rd_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                a = tf.get_variable(name='ra_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                b = tf.get_variable(name='rb_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                e = tf.get_variable(name='re_' + str(m),
                                    initializer=tf.truncated_normal(shape=[self.hidden, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
                f = tf.get_variable(name='rf_' + str(m),
                                    initializer=tf.truncated_normal(shape=[1, 1],
                                                                    mean=0, stddev=0.01,
                                                                    dtype=tf.float32), dtype=tf.float32)
            rpara_exp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rExposure_' + str(m))
            self.rc_list.append(c)
            self.rd_list.append(d)
            self.ra_list.append(a)
            self.rb_list.append(b)
            self.re_list.append(e)
            self.rf_list.append(f)
            self.rpara_exp_list.append(rpara_exp)

        self.rrel_cost_list = []
        self.rexp_cost_list = []
        self.rrel_reg_cost_list = []
        self.rexp_reg_cost_list = []
        self.rrel_cost_final_list = []
        self.rexp_cost_final_list = []
        for m in range(self.C):
            rP = self.rP_list[m]
            rQ = self.rQ_list[m]
            rc = self.rc_list[m]
            rd = self.rd_list[m]
            ra = self.ra_list[m]
            rb = self.rb_list[m]
            re = self.re_list[m]
            rf = self.rf_list[m]

            P = self.P_list[m] + rP
            Q = self.Q_list[m] + rQ
            c = self.c_list[m] + rc
            d = self.d_list[m] + rd
            a = self.a_list[m] + ra
            b = self.b_list[m] + rb
            e = self.e_list[m] + re
            f = self.f_list[m] + rf

            p = tf.nn.embedding_lookup(P, tf.reshape(self.user_input, [-1]))
            q_pos = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_pos, [-1]))
            q_neg = tf.nn.embedding_lookup(Q, tf.reshape(self.item_input_neg, [-1]))

            rel_predict_pos = tf.reduce_sum(p * q_pos, 1, keepdims=True)

            rel_predict_neg = tf.reduce_sum(p * q_neg, 1, keepdims=True)

            w = tf.nn.sigmoid(tf.matmul(q_pos, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q_pos, c) + d) + (1 - w) * self.pop_input_pos,
                         tf.nn.sigmoid(tf.matmul(q_pos, e) + f))
            exp_predict_pos = pop
            exp_predict_pos = tf.clip_by_value(exp_predict_pos, clip_value_min=0.01, clip_value_max=0.99)

            w = tf.nn.sigmoid(tf.matmul(q_neg, a) + b)
            pop = tf.pow(w * tf.nn.sigmoid(tf.matmul(q_neg, c) + d) + (1 - w) * self.pop_input_neg,
                         tf.nn.sigmoid(tf.matmul(q_neg, e) + f))
            exp_predict_neg = pop
            exp_predict_neg = tf.clip_by_value(exp_predict_neg, clip_value_min=0.01, clip_value_max=0.99)

            rrel_cost = -tf.reduce_mean(tf.log(tf.nn.sigmoid(rel_predict_pos - rel_predict_neg)) / self.exp_input)

            rexp_cost = -tf.reduce_mean(tf.log(exp_predict_pos) / self.rel_input) \
                        - tf.reduce_mean(tf.log(1 - exp_predict_neg))

            rel_reg_cost = self.beta * self.reg * 0.5 * (self.l2_norm(rP) + self.l2_norm(rQ))
            exp_reg_cost = self.beta * self.alpha * self.reg * 0.5 * (self.l2_norm(rc) + self.l2_norm(rd)
                                                                      + self.l2_norm(ra) + self.l2_norm(rb)
                                                                      + self.l2_norm(re) + self.l2_norm(rf))

            rrel_cost_final = rrel_cost + rel_reg_cost
            rexp_cost_final = rexp_cost + exp_reg_cost

            self.rrel_cost_list.append(rrel_cost)
            self.rexp_cost_list.append(rexp_cost)
            self.rrel_reg_cost_list.append(rel_reg_cost)
            self.rexp_reg_cost_list.append(exp_reg_cost)
            self.rrel_cost_final_list.append(rrel_cost_final)
            self.rexp_cost_final_list.append(rexp_cost_final)

        self.rel_optimizer_list = []
        self.exp_optimizer_list = []
        self.rrel_optimizer_list = []
        self.rexp_optimizer_list = []
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            for m in range(self.C):
                rel_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.rel_cost_final_list[m],
                                                                                       var_list=self.para_rel_list[m])
                exp_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.exp_cost_final_list[m],
                                                                                       var_list=self.para_exp_list[m])
                rrel_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr / 7.).minimize(self.rrel_cost_final_list[m],
                                                                                             var_list=self.rpara_rel_list[m])
                rexp_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr / 7.).minimize(self.rexp_cost_final_list[m],
                                                                                             var_list=self.rpara_exp_list[m])
                self.rel_optimizer_list.append(rel_optimizer)
                self.exp_optimizer_list.append(exp_optimizer)
                self.rrel_optimizer_list.append(rrel_optimizer)
                self.rexp_optimizer_list.append(rexp_optimizer)

    def generate_R_PR(self):
        self.Pr = np.zeros((self.num_user, self.num_item))
        self.R = np.zeros((self.num_user, self.num_item))
        rel_list = []
        exp_list = []
        for m in range(self.C):
            P, Q, c, d, a, b, e, f = self.sess.run([self.P_list[m], self.Q_list[m],
                                                    self.c_list[m], self.d_list[m],
                                                    self.a_list[m], self.b_list[m],
                                                    self.e_list[m], self.f_list[m]])

            rel = np.matmul(P, Q.T)
            rel = np.exp(rel)
            rel /= np.sum(rel, axis=1, keepdims=True)
            rel *= (self.num_item / 2.)

            w = utility.sigmoid(np.matmul(Q, a) + b)
            pop = np.power(w * utility.sigmoid(np.matmul(Q, c) + d) \
                           + (1 - w) * self.item_pop, utility.sigmoid(np.matmul(Q, e) + f))
            exp = np.zeros((self.num_user, self.num_item)) + pop.T

            user_ids = self.df_list[m]['userId']
            item_ids = self.df_list[m]['itemId']
            rel_list.append(rel[user_ids, item_ids])
            exp_list.append(exp[user_ids, item_ids])

            self.R += rel
            self.Pr += exp
        self.R /= self.C
        self.Pr /= self.C
        for m in range(self.C):
            user_ids = self.df_list[m]['userId']
            item_ids = self.df_list[m]['itemId']
            self.R[user_ids, item_ids] = rel_list[m]
            self.Pr[user_ids, item_ids] = exp_list[m]

        self.Pr[np.where(self.Pr < 0.01)] = 0.01
        self.Pr[np.where(self.Pr > 0.99)] = 0.99

    def train_model(self, itr):
        m_list = np.random.permutation(self.C)
        for m in m_list:
            self.generate_R_PR()
            df = self.df_list[m]
            self.user_list, self.item_list_pos, self.item_list_neg = utility.negative_sampling_BPR(self.num_user,
                                                                                                   self.num_item,
                                                                                                   df,
                                                                                                   self.neg)
            pop_list_pos = self.item_pop[self.item_list_pos.reshape(-1)].reshape((-1, 1))
            pop_list_neg = self.item_pop[self.item_list_neg.reshape(-1)].reshape((-1, 1))
            num_batch = int(len(self.user_list) / float(self.batch_size)) + 1

            start_time = time.time() * 1000.0
            epoch_rel_cost = 0.
            epoch_exp_cost = 0.
            epoch_rel_reg_cost = 0.
            epoch_exp_reg_cost = 0.

            random_idx = np.random.permutation(len(self.user_list))
            for i in tqdm(range(num_batch)):
                batch_idx = None
                if i == num_batch - 1:
                    batch_idx = random_idx[i * self.batch_size:]
                elif i < num_batch - 1:
                    batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

                user_input = self.user_list[batch_idx, :]
                item_input_pos = self.item_list_pos[batch_idx, :]
                item_input_neg = self.item_list_neg[batch_idx, :]
                pop_input_pos = pop_list_pos[batch_idx, :]
                pop_input_neg = pop_list_neg[batch_idx, :]

                rel_input = self.R[user_input.reshape(-1), item_input_pos.reshape(-1)].reshape((-1, 1))
                exp_input = self.Pr[user_input.reshape(-1), item_input_pos.reshape(-1)].reshape((-1, 1))

                for update_id in range(self.C):
                    if update_id != m:
                        rel_para_list = [self.rel_optimizer_list[update_id], self.rel_cost_list[update_id],
                                         self.rel_reg_cost_list[update_id]]
                        exp_para_list = [self.exp_optimizer_list[update_id], self.exp_cost_list[update_id],
                                         self.exp_reg_cost_list[update_id]]

                        _, tmp_rel_cost, tmp_rel_reg_cost \
                            = self.sess.run(rel_para_list,
                                            feed_dict={self.user_input: user_input,
                                                       self.item_input_pos: item_input_pos,
                                                       self.item_input_neg: item_input_neg,
                                                       self.exp_input: exp_input})
                        epoch_rel_cost += tmp_rel_cost
                        epoch_rel_reg_cost += tmp_rel_reg_cost

                        _, tmp_exp_cost, tmp_exp_reg_cost \
                            = self.sess.run(exp_para_list,
                                            feed_dict={self.user_input: user_input,
                                                       self.item_input_pos: item_input_pos,
                                                       self.item_input_neg: item_input_neg,
                                                       self.pop_input_pos: pop_input_pos,
                                                       self.pop_input_neg: pop_input_neg,
                                                       self.rel_input: rel_input})
                        epoch_exp_cost += tmp_exp_cost
                        epoch_exp_reg_cost += tmp_exp_reg_cost

            if itr % self.display == 0:
                print("Training_" + str(m + 1) + " //", "Epoch %d //" % itr,
                      " rel={:.4f}".format(epoch_rel_cost / (self.C - 1)),
                      " exp={:.4f}".format(epoch_exp_cost / (self.C - 1)),
                      " rel_reg={:.4f}".format(epoch_rel_reg_cost / (self.C - 1)),
                      " exp_reg={:.4f}".format(epoch_exp_reg_cost / (self.C - 1)),
                      "Train time : %d ms" % (time.time() * 1000.0 - start_time))

        for _ in range(1):
            self.generate_R_PR()

            self.user_list, self.item_list_pos, self.item_list_neg = utility.negative_sampling_BPR(self.num_user,
                                                                                                   self.num_item,
                                                                                                   self.train_df,
                                                                                                   self.neg)
            pop_list_pos = self.item_pop[self.item_list_pos.reshape(-1)].reshape((-1, 1))
            pop_list_neg = self.item_pop[self.item_list_neg.reshape(-1)].reshape((-1, 1))
            num_batch = int(len(self.user_list) / float(1024)) + 1

            start_time = time.time() * 1000.0
            epoch_rrel_cost = 0.
            epoch_rexp_cost = 0.
            epoch_rrel_reg_cost = 0.
            epoch_rexp_reg_cost = 0.

            random_idx = np.random.permutation(len(self.user_list))
            for i in tqdm(range(num_batch)):
                batch_idx = None
                if i == num_batch - 1:
                    batch_idx = random_idx[i * self.batch_size:]
                elif i < num_batch - 1:
                    batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

                user_input = self.user_list[batch_idx, :]
                item_input_pos = self.item_list_pos[batch_idx, :]
                item_input_neg = self.item_list_neg[batch_idx, :]
                pop_input_pos = pop_list_pos[batch_idx, :]
                pop_input_neg = pop_list_neg[batch_idx, :]

                rel_input = self.R[user_input.reshape(-1), item_input_pos.reshape(-1)].reshape((-1, 1))
                exp_input = self.Pr[user_input.reshape(-1), item_input_pos.reshape(-1)].reshape((-1, 1))

                for update_id in range(self.C):
                    _, tmp_rrel_cost, tmp_rrel_reg_cost \
                        = self.sess.run([self.rrel_optimizer_list[update_id], self.rrel_cost_list[update_id],
                                         self.rrel_reg_cost_list[update_id]],
                                        feed_dict={self.user_input: user_input,
                                                   self.item_input_pos: item_input_pos,
                                                   self.item_input_neg: item_input_neg,
                                                   self.exp_input: exp_input})
                    epoch_rrel_cost += tmp_rrel_cost
                    epoch_rrel_reg_cost += tmp_rrel_reg_cost

                    _, tmp_rexp_cost, tmp_rexp_reg_cost \
                        = self.sess.run([self.rexp_optimizer_list[update_id], self.rexp_cost_list[update_id],
                                         self.rexp_reg_cost_list[update_id]],
                                        feed_dict={self.user_input: user_input,
                                                   self.item_input_pos: item_input_pos,
                                                   self.item_input_neg: item_input_neg,
                                                   self.pop_input_pos: pop_input_pos,
                                                   self.pop_input_neg: pop_input_neg,
                                                   self.rel_input: rel_input})
                    epoch_rexp_cost += tmp_rexp_cost
                    epoch_rexp_reg_cost += tmp_rexp_reg_cost

            if itr % self.display == 0:
                print("Training //", "Epoch %d //" % itr,
                      " rrel={:.4f}".format(epoch_rrel_cost / self.C),
                      " rexp={:.4f}".format(epoch_rexp_cost / self.C),
                      " rrel_reg={:.4f}".format(epoch_rrel_reg_cost / self.C),
                      " rexp_reg={:.4f}".format(epoch_rexp_reg_cost / self.C),
                      "Train time : %d ms" % (time.time() * 1000.0 - start_time))

    def test_model(self, itr):
        start_time = time.time() * 1000.0
        R_all = np.zeros((self.num_user, self.num_item))
        Pr_all = np.zeros((self.num_user, self.num_item))
        Rec_all = np.zeros((self.num_user, self.num_item))
        for m in range(self.C):
            P, Q, c, d, a, b, e, f = self.sess.run([self.P_list[m], self.Q_list[m],
                                                    self.c_list[m], self.d_list[m],
                                                    self.a_list[m], self.b_list[m],
                                                    self.e_list[m], self.f_list[m]])
            rP, rQ, rc, rd, ra, rb, re, rf = self.sess.run([self.rP_list[m], self.rQ_list[m],
                                                            self.rc_list[m], self.rd_list[m],
                                                            self.ra_list[m], self.rb_list[m],
                                                            self.re_list[m], self.rf_list[m]])
            P += rP
            Q += rQ
            a += ra
            b += rb
            c += rc
            d += rd
            e += re
            f += rf

            R = np.matmul(P, Q.T)
            R = np.exp(R)
            R /= np.sum(R, axis=1, keepdims=True)
            R *= (self.num_item / 2.)

            w = utility.sigmoid(np.matmul(Q, a) + b)
            pop = np.power(w * utility.sigmoid(np.matmul(Q, c) + d) \
                           + (1 - w) * self.item_pop, utility.sigmoid(np.matmul(Q, e) + f))
            Pr = pop.T

            Rec = R * Pr
            R_all += R
            Pr_all += Pr
            Rec_all += Rec

        Rec = Rec_all / self.C
        R = R_all / self.C
        Pr = Pr_all / self.C

        Pr[np.where(Pr < 0.01)] = 0.01
        Pr[np.where(Pr > 0.99)] = 0.99

        print('Mean Pr = ' + str(np.mean(Pr)))
        print('Std Pr = ' + str(np.std(Pr)))
        print('Max Pr = ' + str(np.max(Pr)))
        print('Min Pr = ' + str(np.min(Pr)))

        recall = utility.MP_unbiased_vali(R, self.vali_like, self.train_like, Pr, n_workers=10, k=1)

        utility.MP_perfect_Rec_test(R, self.test_user, self.test_interact, self.test_like,
                                    np.array(self.train_like)[self.test_user], n_workers=10, k=0)

        print("Testing //", "Epoch %d //" % itr,
              "Accuracy Testing time : %d ms" % (time.time() * 1000.0 - start_time))
        print("=" * 100)

        return

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


parser = argparse.ArgumentParser(description='CJBPR')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train')
parser.add_argument('--display', type=int, default=1, help='evaluate mode every X epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--reg', type=float, default=2e-5, help='regularization')
parser.add_argument('--alpha', type=float, default=500000, help='exp regularization')
parser.add_argument('--beta', type=float, default=0.5, help='res regularization')
parser.add_argument('--C', type=int, default=6, help='C')
parser.add_argument('--hidden', type=int, default=100, help='latent dimension')
parser.add_argument('--neg', type=int, default=5, help='negative sampling rate')
parser.add_argument('--bs', type=int, default=1024, help='batch size')
parser.add_argument('--data', type=str, default='yahoo', help='path to eval in the downloaded folder')
# parser.add_argument('--data', type=str, default='coat', help='path to eval in the downloaded folder')

args = parser.parse_args()

args.bs = int(args.bs * 1. * (args.C - 1) / args.C)

with open('./' + args.data + '/info.pkl', 'rb') as f:
    info = pickle.load(f)
    args.num_user = info['num_user']
    args.num_item = info['num_item']

train_df = pd.read_csv('./' + args.data + '/train_df.csv')

item_pop = (np.load('./' + args.data + '/item_pop.npy')) + 1
user_pop = (np.load('./' + args.data + '/user_pop.npy')) + 1
prop = (item_pop / np.max(item_pop)).reshape((-1, 1))

train_like = list(np.load('./' + args.data + '/user_train_like.npy', allow_pickle=True))
vali_like = list(np.load('./' + args.data + '/user_vali_like.npy', allow_pickle=True))

test_user = list(np.load('./' + args.data + '/user_test_list.npy', allow_pickle=True))
test_interact = list(np.load('./' + args.data + '/user_test_interact.npy', allow_pickle=True))
test_like = list(np.load('./' + args.data + '/user_test_like.npy', allow_pickle=True))


print('!' * 100)

with tf.Session() as sess:
    bpr = CJBPR(sess, args,
                train_df=train_df, train_like=train_like,
                vali_like=vali_like,
                test_user=test_user, test_interact=test_interact, test_like=test_like,
                item_pop=prop)
    bpr.run()

