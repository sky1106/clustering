#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms, models
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage import color, transform   # lib: scikit-image
import keras
from threading import Thread
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, fashion_mnist_iid, fashion_mnist_noniid
from utils.options import args_parser
from utils.lsh import LSHAlgo
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFashionMnist
from models.Fed import FedAvg
from models.test import test_img

# cluster、lsh-cluster 共用
user_feats = []


def load_dataset(args):
    # load dataset and split users
    print('Loading dataset...')
    print(args.verbose)
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('../data/fashion-mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST('../data/fashion-mnist/', train=False, download=True, transform=trans)
        if args.iid:
            dict_users = fashion_mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, case=args.noniid_case)
    # elif args.dataset == 'svhn':
    #     trans = transforms.Compose([transforms.ToTensor()])
    #     dataset_train = datasets.SVHN('../data/svhn/', split='train', download=True, transform=trans)
    #     dataset_test = datasets.SVHN('../data/svhn/', split='test', download=True, transform=trans)
    #     print(len(dataset_train))
    #     if args.iid:
    #         dict_users = svhn_iid(dataset_train, args.num_users)
    #     else:
    #         dict_users = svhn_noniid(dataset_train, args.num_users, case=args.noniid_case)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users


# type of experiment:
# - base:        general fed
# - cluster:     base with cluster
# - lsh-cluster: base with cluster & lsh
def run_fed(args, dataset_train, dataset_test, dict_users, num_clusters, code_dim, type_exp = 'base'):
    print('Current experiment type: ', type_exp)

    img_size = dataset_train[0][0].shape

    dict_clusters = {}
    if type_exp == 'cluster' or type_exp == 'lsh-cluster':
        # feature map
        print('Featuring...')
        input_shape = (max(img_size[1], 32), max(img_size[2], 32), max(img_size[0], 3))

        if args.feat_map == 'resnet50':
            model1 = keras.applications.resnet.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        elif args.feat_map == 'vgg19':
            model1 = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=input_shape)
        else:
            exit('Error: unrecognized keras application')

        if len(user_feats):
            pass
        else:
            for idx_user in dict_users:
                print('User', idx_user, 'featuring...')
                user_images = []
                for idx in dict_users[idx_user]:
                    image = dataset_train[idx][0].numpy()
                    if args.dataset in ['mnist', 'fashion-mnist']:
                        image = color.gray2rgb(image)[0]
                        image = transform.resize(image, (32, 32))
                    user_images.append(image)

                if args.dataset not in ['mnist', 'fashion-mnist']:
                    user_images = np.swapaxes(user_images, 1, 2)
                    user_images = np.swapaxes(user_images, 2, 3)
                    
                pred = model1.predict([user_images])
                feats = np.mean([data[0][0] for data in pred], axis=0)
                user_feats.append(feats)

        if type_exp == 'lsh-cluster':
            # 局部敏感哈希
            print('LSH...')
            lsh = LSHAlgo(feat_dim=len(user_feats[0]), code_dim=code_dim) # code_dim: 输出维度
            user_feats1 = lsh.run(user_feats)
        else:
            # 普通降维
            print('PCA...')
            pca = PCA(n_components=args.pca_comps, random_state=728)
            user_feats1 = pca.fit_transform(user_feats)

        # 聚类 users
        print('Clustering...')
        kmeans = KMeans(n_clusters=num_clusters, random_state=728)
        kmeans.fit(user_feats1)

        for idx_user, label in enumerate(kmeans.labels_):
            if label in dict_clusters:
                dict_clusters[label].append(idx_user)
            else:
                dict_clusters[label] = [idx_user]
        print('Clustering finished.')
        print('Dict of cluster - users: ', dict_clusters)


    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fashion-mnist':
        net_glob = CNNFashionMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()


    # batch training
    if args.target_acc == 0:
        return batch_train(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args)
    return batch_train_with_target_acc(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args)


def batch_train(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args):
    loss_train_batch = []
    acc_test_batch = []

    for big_iter in range(args.iterations):
        print('Iteration ', big_iter)

        # copy weights
        net_glob_copy = copy.deepcopy(net_glob)

        # training
        loss_train = []
        acc_test = []

        for iter in range(args.epochs):
            one_loss_train, one_acc_test = train_one_round(iter, type_exp, net_glob_copy, dataset_train, dataset_test, dict_users, dict_clusters, args)
            loss_train.append(one_loss_train)
            acc_test.append(one_acc_test)

        loss_train_batch.append(loss_train)
        acc_test_batch.append(acc_test)

    loss_train_avg = np.mean(loss_train_batch, axis=0)
    acc_test_avg = np.mean(acc_test_batch, axis=0)

    loss_train_std = np.std(loss_train_batch, axis=0)
    acc_test_std = np.std(acc_test_batch, axis=0)

    return loss_train_avg, acc_test_avg, loss_train_std, acc_test_std


def batch_train_with_target_acc(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args):
    round_batch = []
    
    for big_iter in range(args.iterations):
        print('Iteration ', big_iter)

        # copy weights
        net_glob_copy = copy.deepcopy(net_glob)

        # training
        acc_test = float('-inf')
        round = 0
        while acc_test < args.target_acc:
            loss_train, acc_test = train_one_round(round, type_exp, net_glob_copy, dataset_train, dataset_test, dict_users, dict_clusters, args)
            round += 1
        round_batch.append(round)

    round_avg = np.mean(round_batch)

    return round_avg, round_batch


def train_one_round(iter, type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args):
    w_locals, loss_locals = [], []

    if type_exp == 'cluster' or type_exp == 'lsh-cluster':
        # 预先聚类的情况
        idxs_users = []
        for idx_cluster in dict_clusters:
            idxs_users += list(np.random.choice(list(dict_clusters[idx_cluster]), 1, replace=False))
    else:
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    threads = [Thread(target=train_single, args = (args, dataset_train, dict_users, idx, net_glob, w_locals, loss_locals)) for idx in idxs_users]
    [t.start() for t in threads]
    [t.join() for t in threads]


    # for idx in idxs_users:
    #     train_single(args, dataset_train, dict_users, idx, net_glob, w_locals, loss_locals)



    # update global weights
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob_copy
    net_glob.load_state_dict(w_glob)

    # print loss & acc
    loss_avg = sum(loss_locals) / len(loss_locals)
    one_acc_test, one_loss_test = test_img(net_glob, dataset_test, args)
    print('Round {:3d}, Average loss {:.3f}, Test accuracy {:.3f}'.format(iter, loss_avg, one_acc_test))

    return loss_avg, one_acc_test

def train_single(args, dataset_train, dict_users, idx, net_glob, w_locals, loss_locals):
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    w_locals.append(copy.deepcopy(w))
    loss_locals.append(copy.deepcopy(loss))


def plot(data, data_std, ylabel, args):
    print("##############into plot#############")
    args = args_parser()
    plt.figure()    
    # colour = ['darkblue','darkred','darkgreen','black','darkmagenta','darkorange','darkcyan']
    # ecolour = ['cornflowerblue','lightcoral','lightgreen','gray','magenta','bisque','cyan']
    # i = 0
    for label in data:
        if args.plot_std:
            plt.errorbar(range(len(data[label])), data[label], yerr=data_std[label], label=label, elinewidth=1)
        else:
            plt.plot(range(len(data[label])), data[label], label=label)
        # i = i + 1
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./output/fed_{}_{}_{}_{}_{}_{}_{}_iid{}_{}.pdf'.format(args.type_exp, ylabel, args.dataset, args.model, args.noniid_case, args.iterations, args.epochs, args.iid, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

def save_result(data, ylabel, args):
    with open(r'./test/{}_{}_{}_{}_{}_{}_{}_iid{}_{}.txt'.format(args.type_exp, ylabel, args.dataset, args.model, args.noniid_case, args.iterations, args.epochs, args.iid, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'a') as f:
        for label in data:
            f.write(label)
            f.write(' ')
            for item in data[label]:
                item1 = str(item)
                f.write(item1)
                f.write(' ')
            f.write('\n')
    print('save finished')
    f.close()


if __name__ == '__main__':
    # parse args
    print('begin time: ', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    if args.multi_plot_case == 0:
        # Case 0:
        # 默认作图

         # load dataset and split users
        dataset_train, dataset_test, dict_users = load_dataset(args)

        labels = ['base', 'cluster', 'lsh-cluster'] if args.type_exp == 'all' else [args.type_exp]

        if args.target_acc == 0:
            dict_train_loss = {}
            dict_acc_test = {}
            dict_std_train_loss = {}
            dict_std_acc_test = {}
            for label in labels:
                dict_train_loss[label], dict_acc_test[label], dict_std_train_loss[label], dict_std_acc_test[label] = run_fed(
                    args, dataset_train, dataset_test, dict_users, args.num_clusters, args.code_dim, type_exp = label
                )

            print(dict_train_loss, dict_acc_test)

            save_result(dict_train_loss, 'train_loss', args)
            save_result(dict_acc_test, 'test_acc', args)
            plot(dict_train_loss, dict_std_train_loss, 'train_loss', args)
            plot(dict_acc_test, dict_std_acc_test, 'test_acc', args)
        else:
            for label in labels:
                round_avg, round_batch = run_fed(args, dataset_train, dataset_test, dict_users, args.num_clusters, args.code_dim, label)
                print('{}, average round: {}'.format(label, round_avg))
                print(round_batch)
    

    elif args.multi_plot_case == 1:
        # Case 1:
        # - base
        # - 几种 num_clusters 的 lsh-cluster
        # - 几种 num_clusters 的 cluster
        dataset_train, dataset_test, dict_users = load_dataset(args)

        list_num_clusters = [10, 20, 50]  # <- 改这里

        plot_list = [
            { 'type_exp': 'base'}
        ] + [
            { 'type_exp': 'lsh-cluster', 'num_clusters': num } for num in list_num_clusters
        ] + [
            { 'type_exp': 'cluster', 'num_clusters': num } for num in list_num_clusters
        ]

        dict_train_loss = {}
        dict_acc_test = {}
        dict_std_train_loss = {}
        dict_std_acc_test = {}

        for p in plot_list:
            if 'num_clusters' in p:
                label = '{}: num_clusters={}'.format(p['type_exp'], p['num_clusters'])
            else:
                label = p['type_exp']
            dict_train_loss[label], dict_acc_test[label], dict_std_train_loss[label], dict_std_acc_test[label] = run_fed(
                args,
                dataset_train,
                dataset_test,
                dict_users,
                p['num_clusters'] if 'num_clusters' in p else 0,
                args.code_dim,
                type_exp = p['type_exp']
            )
        print(dict_acc_test, dict_std_acc_test)
        plot(dict_acc_test, dict_std_acc_test, 'test_acc', args)


    elif args.multi_plot_case == 2:
        # Case 2:
        # - base
        # - 几种 code_dim 的 lsh-cluster
        dataset_train, dataset_test, dict_users = load_dataset(args)

        list_code_dim = [100, 200, 512]  # <- 改这里

        plot_list = [
            { 'type_exp': 'base' }
        ] + [
            { 'type_exp': 'lsh-cluster', 'code_dim': dim } for dim in list_code_dim
        ]

        dict_train_loss = {}
        dict_acc_test = {}
        dict_std_train_loss = {}
        dict_std_acc_test = {}

        for p in plot_list:
            if 'code_dim' in p:
                label = '{}: code_dim={}'.format(p['type_exp'], p['code_dim'])
            else:
                label = p['type_exp']
            dict_train_loss[label], dict_acc_test[label], dict_std_train_loss[label], dict_std_acc_test[label] = run_fed(
                args,
                dataset_train,
                dataset_test,
                dict_users,
                args.num_clusters,
                p['code_dim'] if 'code_dim' in p else 0,
                type_exp = p['type_exp']
            )
        print(dict_acc_test, dict_std_acc_test)
        plot(dict_acc_test, dict_std_acc_test, 'test_acc', args)


    elif args.multi_plot_case == 3:
        # Case 2:
        # - 几种 num_users 的 base
        # - 几种 num_users 的 cluster
        # - 几种 num_users 的 lsh-cluster
        pass

    print('end time: ', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
