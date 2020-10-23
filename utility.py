
import numpy as np
from operator import itemgetter
from scipy.sparse import coo_matrix
from multiprocessing import Process, Queue, Pool, Manager


top1 = 5
top2 = 10
top3 = 20
top4 = 30
k_set = [[1, 2, 3, 4], [top1, top2, top3, top4]]

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


def negative_sampling_MF(num_user, num_item, train_df, neg_rate):
    pos_user_array = train_df['userId'].values
    pos_item_array = train_df['itemId'].values
    train_mat = coo_matrix((np.ones(len(train_df)),
                            (train_df['userId'].values, train_df['itemId'].values)),
                           shape=(num_user, num_item)).toarray()
    user_pos = pos_user_array.reshape((-1, 1))
    user_neg = np.tile(pos_user_array, neg_rate).reshape((-1, 1))
    pos = pos_item_array.reshape((-1, 1))
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    label = train_mat[user_neg, neg]
    idx = (label == 0).reshape(-1)
    user_neg = user_neg[idx, :]
    neg = neg[idx, :]
    pos_lable = np.ones(pos.shape)
    neg_lable = np.zeros(neg.shape)
    return np.concatenate([user_pos, user_neg], axis=0), np.concatenate([pos, neg], axis=0), np.concatenate([pos_lable, neg_lable], axis=0)


def negative_sampling_BPR(num_user, num_item, train_df, neg_rate):
    pos_user_array = train_df['userId'].values
    pos_item_array = train_df['itemId'].values
    train_mat = coo_matrix((np.ones(len(train_df)),
                            (train_df['userId'].values, train_df['itemId'].values)),
                           shape=(num_user, num_item)).toarray()
    user = np.tile(pos_user_array, neg_rate).reshape((pos_user_array.shape[0] * neg_rate, 1))
    pos = np.tile(pos_item_array, neg_rate).reshape((pos_user_array.shape[0] * neg_rate, 1))
    neg = np.random.choice(np.arange(num_item), size=(neg_rate * pos_user_array.shape[0]), replace=True).reshape((-1, 1))
    label = train_mat[user, neg]
    idx = (label == 0).reshape(-1)
    user = user[idx, :]
    pos = pos[idx, :]
    neg = neg[idx, :]
    idx = (pos != neg).reshape(-1)
    user = user[idx, :]
    pos = pos[idx, :]
    neg = neg[idx, :]
    return user, pos, neg


def recall_DCG_MAP(ranked_item_score, like_item, k=1):
    num_item = len(ranked_item_score)

    # compute the number of true positive items at top k
    count_1, count_2, count_3, count_4 = 0., 0., 0., 0.
    dcg_1, dcg_2, dcg_3, dcg_4 = 0., 0., 0., 0.
    map = np.zeros(k_set[k][-1])
    map_ind = np.zeros(k_set[k][-1])
    for i in range(num_item):
        item_i = int(ranked_item_score[i][0])
        if item_i in like_item:
            if i < k_set[k][0]:
                count_1 += 1.0
                dcg_1 += 1.0 / np.log2(i + 2)
            if i < k_set[k][1]:
                count_2 += 1.0
                dcg_2 += 1.0 / np.log2(i + 2)
            if i < k_set[k][2]:
                count_3 += 1.0
                dcg_3 += 1.0 / np.log2(i + 2)
            if i < k_set[k][3]:
                count_4 += 1.0
                dcg_4 += 1.0 / np.log2(i + 2)
            map[i:] += 1.
            map_ind[i] = 1

    l = len(like_item)
    if l == 0:
        l = 1

    # recall@k
    recall_1 = count_1 / l
    recall_2 = count_2 / l
    recall_3 = count_3 / l
    recall_4 = count_4 / l

    # map@k
    map = (map / np.arange(1, k_set[k][-1] + 1)) * map_ind
    map_1 = np.sum(map[:k_set[k][0]]) / np.sum(map_ind[:k_set[k][0]] + 1e-7)
    map_2 = np.sum(map[:k_set[k][1]]) / np.sum(map_ind[:k_set[k][1]] + 1e-7)
    map_3 = np.sum(map[:k_set[k][2]]) / np.sum(map_ind[:k_set[k][2]] + 1e-7)
    map_4 = np.sum(map[:k_set[k][3]]) / np.sum(map_ind[:k_set[k][3]] + 1e-7)

    return np.array([recall_1, recall_2, recall_3, recall_4]),\
           np.array([dcg_1, dcg_2, dcg_3, dcg_4]), \
           np.array([map_1, map_2, map_3, map_4])


def SNIPS_recall_DCG_MAP(ranked_item_score, like_item, prop, k=1):
    num_item = len(ranked_item_score)

    # compute the number of true positive items at top k
    count_1, count_2, count_3, count_4 = 0., 0., 0., 0.
    dcg_1, dcg_2, dcg_3, dcg_4 = 0., 0., 0., 0.
    map = np.zeros(k_set[k][-1])
    map_ind = np.zeros(k_set[k][-1])
    for i in range(num_item):
        item_i = int(ranked_item_score[i][0])
        if item_i in like_item:
            if i < k_set[k][0]:
                count_1 += 1.0 / prop[item_i]
                dcg_1 += 1.0 / np.log2(i + 2) / prop[item_i]
            if i < k_set[k][1]:
                count_2 += 1.0 / prop[item_i]
                dcg_2 += 1.0 / np.log2(i + 2) / prop[item_i]
            if i < k_set[k][2]:
                count_3 += 1.0 / prop[item_i]
                dcg_3 += 1.0 / np.log2(i + 2) / prop[item_i]
            if i < k_set[k][3]:
                count_4 += 1.0 / prop[item_i]
                dcg_4 += 1.0 / np.log2(i + 2) / prop[item_i]
            map[i:] += (1. / prop[item_i])
            map_ind[i] = 1

    SN = 0.
    for i in like_item:
        SN += 1.0 / prop[i]
    if SN == 0:
        SN = 1

    # recall@k
    recall_1 = count_1 / SN
    recall_2 = count_2 / SN
    recall_3 = count_3 / SN
    recall_4 = count_4 / SN

    # map@k
    map = (map / np.arange(1, k_set[k][-1] + 1)) * map_ind
    map_1 = np.sum(map[:k_set[k][0]]) / np.sum(map_ind[:k_set[k][0]] + 1e-7)
    map_2 = np.sum(map[:k_set[k][1]]) / np.sum(map_ind[:k_set[k][1]] + 1e-7)
    map_3 = np.sum(map[:k_set[k][2]]) / np.sum(map_ind[:k_set[k][2]] + 1e-7)
    map_4 = np.sum(map[:k_set[k][3]]) / np.sum(map_ind[:k_set[k][3]] + 1e-7)

    return np.array([recall_1, recall_2, recall_3, recall_4]),\
           np.array([dcg_1, dcg_2, dcg_3, dcg_4]), \
           np.array([map_1, map_2, map_3, map_4])


def perfect_Rec_test(num_user, Rec, train_like, test_interact, test_like, recall_queue, dcg_queue, map_queue, n_user_queue, k=0):
    recall = np.array([0.0, 0.0, 0.0, 0.0])
    dcg = np.array([0.0, 0.0, 0.0, 0.0])
    map = np.array([0.0, 0.0, 0.0, 0.0])

    user_num = num_user

    for u in range(num_user):  # iterate each user
        interact = test_interact[u]
        u_pred = Rec[u, interact]
        # u_pred = np.matmul(P[u, :], Q[interact, :].T)

        ranked_item_score = np.argpartition(u_pred, -k_set[k][-1])[-k_set[k][-1]:]
        ranked_item_score = (np.array([ranked_item_score, u_pred[ranked_item_score]])).T
        ranked_item_score = sorted(ranked_item_score, key=itemgetter(1), reverse=True)

        # ranked_item_score = (np.array([range(len(u_pred)), u_pred])).T
        # ranked_item_score = sorted(ranked_item_score, key=itemgetter(1), reverse=True)

        # calculate the metrics
        if len(test_like[u]) > 0 and len(train_like[u]) > 0:
            recall_u, dcg_u, map_u = recall_DCG_MAP(ranked_item_score, test_like[u], k=k)
            recall += recall_u
            dcg += dcg_u
            map += map_u
        else:
            user_num -= 1
    recall_queue.put(recall)
    dcg_queue.put(dcg)
    map_queue.put(map)
    n_user_queue.put(user_num)


def MP_perfect_Rec_test(Rec, user_list, user_test_interact, user_test_like, user_train_like, n_workers=10, k=0):
    num_user = len(user_list)
    m = Manager()
    recall_queue = m.Queue(maxsize=n_workers)
    dcg_queue = m.Queue(maxsize=n_workers)
    map_queue = m.Queue(maxsize=n_workers)
    n_user_queue = m.Queue(maxsize=n_workers)
    processors = []

    num_user_each = int(num_user / n_workers)
    for i in range(n_workers):
        if i < n_workers - 1:
            num_batch = num_user_each
            start = num_user_each * i
            end = num_user_each * (i + 1)
        else:
            num_batch = num_user - num_user_each * i
            start = num_user_each * i
            end = num_user

        p = Process(target=perfect_Rec_test, args=(num_batch,
                                                   Rec[user_list[start: end], :],
                                                   user_train_like[start: end],
                                                   user_test_interact[start: end],
                                                   user_test_like[start: end],
                                                   recall_queue, dcg_queue, map_queue,
                                                   n_user_queue,
                                                   k))
        processors.append(p)
        p.start()
    print('!!!!!!!!!!!!!!!!! perfect test start !!!!!!!!!!!!!!!!!!')

    for p in processors:
        p.join()
    recall = recall_queue.get()
    while not recall_queue.empty():
        tmp = recall_queue.get()
        recall += tmp
    dcg = dcg_queue.get()
    while not dcg_queue.empty():
        tmp = dcg_queue.get()
        dcg += tmp
    map = map_queue.get()
    while not map_queue.empty():
        tmp = map_queue.get()
        map += tmp
    n_user = n_user_queue.get()
    while not n_user_queue.empty():
        tmp = n_user_queue.get()
        n_user += tmp

    # compute the average over all users
    recall /= n_user
    dcg /= n_user
    map /= n_user

    print('recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f]'
          % (k_set[k][0], recall[0], k_set[k][1], recall[1], k_set[k][2], recall[2], k_set[k][3], recall[3]))
    print('DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f]'
          % (k_set[k][0], dcg[0], k_set[k][1], dcg[1], k_set[k][2], dcg[2], k_set[k][3], dcg[3]))
    print('MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f]'
          % (k_set[k][0], map[0], k_set[k][1], map[1], k_set[k][2], map[2], k_set[k][3], map[3]))
    return np.array([recall[2], dcg[2], map[2]])


def unbiased_vali(num_user, Rec, train_like, vali_like, prop, vs_list, n_user_queue, k=1):
    for i in range(num_user):
        Rec[i, train_like[i]] = -100000.0

    recall_vs = np.array([0.0, 0.0, 0.0, 0.0])
    dcg_vs = np.array([0.0, 0.0, 0.0, 0.0])
    map_vs = np.array([0.0, 0.0, 0.0, 0.0])

    for u in range(num_user):  # iterate each user
        u_pred = Rec[u, :]

        ranked_item_score = np.argpartition(u_pred, -k_set[k][-1])[-k_set[k][-1]:]
        ranked_item_score = (np.array([ranked_item_score, u_pred[ranked_item_score]])).T
        ranked_item_score = sorted(ranked_item_score, key=itemgetter(1), reverse=True)

        u_prop = prop[u, :]

        # calculate the metrics
        if len(vali_like[u]) > 0 and len(train_like[u]) > 0:
            recall_u, dcg_u, map_u = SNIPS_recall_DCG_MAP(ranked_item_score, vali_like[u], u_prop, k=k)
            recall_vs += recall_u
            dcg_vs += dcg_u
            map_vs += map_u
        else:
            num_user -= 1

    vs_list[0].put(recall_vs)
    vs_list[1].put(dcg_vs)
    vs_list[2].put(map_vs)

    n_user_queue.put(num_user)


def MP_unbiased_vali(Rec, vali_like, train_like, prop, n_workers=10, k=1):
    num_user = Rec.shape[0]

    m = Manager()
    recall_vs_queue = m.Queue(maxsize=n_workers)
    dcg_vs_queue = m.Queue(maxsize=n_workers)
    map_vs_queue = m.Queue(maxsize=n_workers)
    vs_queue_list = [recall_vs_queue, dcg_vs_queue, map_vs_queue]
    n_user_queue = m.Queue(maxsize=n_workers)
    processors = []

    num_user_each = int(num_user / n_workers)
    for i in range(n_workers):
        if i < n_workers - 1:
            num_batch = num_user_each
            start = num_user_each * i
            end = num_user_each * (i + 1)
        else:
            num_batch = num_user - num_user_each * i
            start = num_user_each * i
            end = num_user

        p = Process(target=unbiased_vali, args=(num_batch,
                                                Rec[start: end],
                                                train_like[start: end],
                                                vali_like[start: end],
                                                prop[start: end],
                                                vs_queue_list,
                                                n_user_queue,
                                                k))
        processors.append(p)
        p.start()

    for p in processors:
        p.join()

    recall_vs = vs_queue_list[0].get()
    while not vs_queue_list[0].empty():
        tmp = vs_queue_list[0].get()
        recall_vs += tmp
    dcg_vs = vs_queue_list[1].get()
    while not vs_queue_list[1].empty():
        tmp = vs_queue_list[1].get()
        dcg_vs += tmp
    map_vs = vs_queue_list[2].get()
    while not vs_queue_list[2].empty():
        tmp = vs_queue_list[2].get()
        map_vs += tmp

    n_user = n_user_queue.get()
    while not n_user_queue.empty():
        tmp = n_user_queue.get()
        n_user += tmp

    print('!!!!!!!!!!!!!!!!! validation SNIPS test !!!!!!!!!!!!!!!!!!')
    recall_vs /= n_user
    dcg_vs /= n_user
    map_vs /= n_user
    print('recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f],\t||\t recall_%d   \t[%.7f]'
          % (k_set[k][0], recall_vs[0], k_set[k][1], recall_vs[1], k_set[k][2], recall_vs[2], k_set[k][3], recall_vs[3]))
    print('DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f],\t||\t DCG_%d      \t[%.7f]'
          % (k_set[k][0], dcg_vs[0], k_set[k][1], dcg_vs[1], k_set[k][2], dcg_vs[2], k_set[k][3], dcg_vs[3]))
    print('MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f],\t||\t MAP_%d      \t[%.7f]'
          % (k_set[k][0], map_vs[0], k_set[k][1], map_vs[1], k_set[k][2], map_vs[2], k_set[k][3], map_vs[3]))

    # return np.mean(recall_vs[:3])
    return recall_vs[3]