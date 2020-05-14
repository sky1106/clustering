#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from utils.lsh import LSHAlgo
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# keras.backend.set_image_data_format("channels_first")

# cluster、lsh-cluster 共用
user_feats = []


def load_dataset(args):
    # load dataset and split users
    print('Loading dataset...')
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, case=args.noniid_case)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users


# type of experiment:
# - base:        general fed
# - cluster:     base with cluster
# - lsh-cluster: base with cluster & lsh
def run_fed(args, dataset_train, dataset_test, dict_users, type_exp = 'base'):
    print('Current experiment type: ', type_exp)

    img_size = dataset_train[0][0].shape

    dict_clusters = {}
    if type_exp == 'cluster' or type_exp == 'lsh-cluster':
        # feature map
        print('Featuring...')
        input_shape = (img_size[1], img_size[2], img_size[0])

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
                user_images = [dataset_train[idx][0].numpy().swapaxes(0, 1).swapaxes(1, 2) for idx in dict_users[idx_user]]
                
                pred = model1.predict([user_images])
                feats = np.mean([data[0][0] for data in pred], axis=0)
                user_feats.append(feats)

        if type_exp == 'lsh-cluster':
            # 局部敏感哈希
            print('LSH...')
            lsh = LSHAlgo(feat_dim=len(user_feats[0]), code_dim=512) # code_dim: 输出维度
            user_feats1 = lsh.run(user_feats)
        else:
            # 普通降维
            print('PCA...')
            pca = PCA(n_components=args.pca_comps, random_state=728)
            user_feats1 = pca.fit_transform(user_feats)

        # 聚类 users
        print('Clustering...')
        kmeans = KMeans(n_clusters=args.num_clusters, random_state=728)
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
    loss_train_list, acc_test_list = batch_train(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args)


    return loss_train_list, acc_test_list


def batch_train(type_exp, net_glob, dataset_train, dataset_test, dict_users, dict_clusters, args):
    loss_train_batch = []
    acc_test_batch = []

    for big_iter in range(args.iterations):
        print('Iteration ', big_iter)

        # copy weights
        net_glob_copy = copy.deepcopy(net_glob)
        w_glob = net_glob_copy.state_dict()

        # training
        loss_train = []
        acc_test = []

        for iter in range(args.epochs):
            w_locals, loss_locals = [], []

            if type_exp == 'cluster' or type_exp == 'lsh-cluster':
                # 预先聚类的情况
                idxs_users = []
                for idx_cluster in dict_clusters:
                    idxs_users += list(np.random.choice(list(dict_clusters[idx_cluster]), 1, replace=False))
            else:
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob_copy).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = FedAvg(w_locals)

            # copy weight to net_glob_copy
            net_glob_copy.load_state_dict(w_glob)

            # print loss & acc
            loss_avg = sum(loss_locals) / len(loss_locals)
            one_acc_test, one_loss_test = test_img(net_glob_copy, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test accuracy {:.3f}'.format(iter, loss_avg, one_acc_test))
            loss_train.append(loss_avg)
            acc_test.append(one_acc_test)

        loss_train_batch.append(loss_train)
        acc_test_batch.append(acc_test)

    loss_train_avg = np.mean(loss_train_batch, axis=0)
    acc_test_avg = np.mean(acc_test_batch, axis=0)
    return loss_train_avg, acc_test_avg


def plot(data, ylabel, args):
    args = args_parser()
    plt.figure()    
    for label in data:
        plt.plot(range(len(data[label])), data[label], label=label)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./save/fed_{}_{}_{}_{}_{}_{}_iid{}_{}.png'.format(args.type_exp, ylabel, args.dataset, args.model, args.iterations, args.epochs, args.iid, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    dataset_train, dataset_test, dict_users = load_dataset(args)

    labels = ['base', 'cluster', 'lsh-cluster'] if args.type_exp == 'all' else [args.type_exp]
    # labels = ['cluster', 'lsh-cluster'] if args.type_exp == 'all' else [args.type_exp]
    dict_train_loss = {}
    dict_acc_test = {}
    for label in labels:
        dict_train_loss[label], dict_acc_test[label] = run_fed(args, dataset_train, dataset_test, dict_users, label)

    print(dict_train_loss, dict_acc_test)

    plot(dict_train_loss, 'train_loss', args)
    plot(dict_acc_test, 'test_acc', args)
