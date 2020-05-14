#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, case = 1):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 600
    if case == 1:
        # 每个 user 只有 1 类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs)
    elif case == 2:
        # 每个 user 中只有 2 类
        return noniid_label_2(dataset, num_users, 200, 300)
    elif case == 3:
        # 每个 user 中，80% 属于 1 类，20% 属于其他类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio = 0.8)
    elif case == 4:
        # 每个 user 中，50% 属于 1 类，50% 属于其他类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio = 0.5)
    else:
         exit('Error: unrecognized noniid case')


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users, case = 1):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 100, 500
    if case == 1:
        # 每个 user 只有 1 类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs)
    elif case == 2:
        # 每个 user 中只有 2 类
        return noniid_label_2(dataset, num_users, 200, 250)
    elif case == 3:
        # 每个 user 中，80% 属于 1 类，20% 属于其他类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio = 0.8)
    elif case == 4:
        # 每个 user 中，50% 属于 1 类，50% 属于其他类
        return noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio = 0.5)
    else:
         exit('Error: unrecognized noniid case')


# 每个 user 中，[ratio] 比例属于一类
def noniid_ratio_r_label_1(dataset, num_users, num_shards, num_imgs, ratio = 1):
    # num_shards, num_imgs = 100, 500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 1, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:int((rand+ratio)*num_imgs)]), axis=0)

    if ratio < 1:
        rest_idxs = np.array([], dtype='int64')
        idx_shard = [i for i in range(num_shards)]
        for i in idx_shard:
            rest_idxs = np.concatenate((rest_idxs, idxs[int((i+ratio)*num_imgs):(i+1)*num_imgs]), axis=0)
        num_items = int(len(dataset)/num_users*(1-ratio))
        for i in range(num_users):
            rest_to_add = set(np.random.choice(rest_idxs, num_items, replace=False))
            dict_users[i] = np.concatenate((dict_users[i], list(rest_to_add)), axis=0)
            rest_idxs = list(set(rest_idxs) - rest_to_add)

    return dict_users


# 每个 user 中只有 2 类
# 未来可以做成可调参的
def noniid_label_2(dataset, num_users, num_shards, num_imgs):
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    for i in range(num_users):
        len_idx_shard = len(idx_shard)
        rand1 = np.random.choice(idx_shard[0:int(len_idx_shard/2)], 1, replace=False)[0]
        rand2 = np.random.choice(idx_shard[int(len_idx_shard/2):len_idx_shard], 1, replace=False)[0]
        idx_shard = list(set(idx_shard) - set([rand1, rand2]))
        for rand in [rand1, rand2]:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:int((rand+1)*num_imgs)]), axis=0)

    return dict_users


if __name__ == '__main__':
    # 如果需要验证 non-iid 的处理是否正确，跑一下可以直观看到
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    num = 100
    d = mnist_noniid(dataset_train, num, 3)
    for user_idx in d:
        print(user_idx)
        print([dataset_train[img_idx][1] for img_idx in d[user_idx]])
