import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import itertools
from numpy import genfromtxt
sns.set(font_scale=1.0, style='whitegrid', rc={"grid.linewidth": 1.})

plt.rc('font', size = 15)
plt.rc('figure', titlesize=16)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=13)

sns.set_palette("colorblind")
#Accuracy
queries2 = [0, 50, 100, 250, 500, 750, 1000, 4000, 8000]
queries = [50, 100, 250, 500, 750, 1000, 4000, 8000]
datasets = ['cifar10', 'mnist', 'svhn']
victimacc = {'mnist':99.44, 'cifar10':95.54, 'svhn':96.17}
for dataset in datasets:
    acc = []
    acc.append(10)
    for i in queries:
        fname = f'{dataset}@{str(i)}new/log.txt'
        f = open(
            os.path.join('/ssd003/home/akaleem/capc-learning-main/MixMatch-pytorch',
                         fname), 'r')
        f.readline()
        done = False
        temp = []
        temp2 = []
        while done == False:
            a = f.readline()
            b = a.split()
            if len(b) == 0:
                done = True
                break
            if dataset == "svhn":
                temp2.append(float(b[-1]))
                temp.append(float(b[-3]))
            else:
                temp.append(float(b[-1]))
                temp2.append(float(b[-3]))
        #acc.append(max(temp))
        acc.append(temp[temp2.index(max(temp2))]) # test accuracy where validation acc is the highest
    plt.plot(queries2, acc, label = "mixmatch")
    plt.plot(queries2, [victimacc[dataset]]*len(queries2), label = "victim accuracy")
    plt.xlabel("Queries")
    if dataset == "mnist":
        plt.ylabel("Accuracy")
    else:
        plt.ylabel("")
    plt.title(f'{dataset.upper()}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        f'/ssd003/home/akaleem/capc-learning-main/overleafgraphs/mixmatch/{dataset}/acc.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

#
# #
# #Entropy vs Accuracy
for dataset in datasets:
    acc = []
    ent = []
    acc.append(10)
    ent.append(0)
    for i in queries:
        fname = f'{dataset}@{str(i)}new/log.txt'
        f = open(
            os.path.join('/ssd003/home/akaleem/capc-learning-main/MixMatch-pytorch',
                         fname), 'r')
        f.readline()
        done = False
        temp = []
        temp2 = []
        while done == False:
            a = f.readline()
            b = a.split()
            if len(b) == 0:
                done = True
                break
            temp.append(float(b[-1]))
            temp2.append(float(b[-3]))
        #acc.append(max(temp))
        acc.append(temp[temp2.index(max(temp2))]) # test accuracy where validation acc is the highest
        f.close()
        fname = f'{dataset}@{str(i)}new/stats.txt'
        f = open(
            os.path.join(
                '/ssd003/home/akaleem/capc-learning-main/MixMatch-pytorch',
                fname), 'r')
        a = f.readline()
        b = a.find(":")
        ent.append(float(a[b+1:]))
        f.close()
    plt.plot(ent, acc)
    plt.xlabel("Entropy")
    if dataset == "mnist":
        plt.ylabel("Accuracy")
    else:
        plt.ylabel("")
    plt.title(f"{dataset.upper()}")
    plt.savefig(
        f'/ssd003/home/akaleem/capc-learning-main/overleafgraphs/mixmatch/{dataset}/entacc.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

#Entropy
for dataset in datasets:
    ent = []
    ent.append(0)
    for i in queries:
        fname = f'{dataset}@{str(i)}new/stats.txt'
        f = open(
            os.path.join(
                '/ssd003/home/akaleem/capc-learning-main/MixMatch-pytorch',
                fname), 'r')
        a = f.readline()
        b = a.find(":")
        ent.append(float(a[b+1:]))
        f.close()
    plt.plot(queries2, ent, label = "MixMatch")
    fname = 'random/log_raw_entropy_random.txt'
    if dataset == "mnist":
        fname = os.path.join(f'/ssd003/home/akaleem/capc-learning-main/adaptive-model/mnist/MnistNetPate/1-models/', fname)
    elif dataset == "cifar10":
        fname = 'random/log_raw_entropy_random.txt'
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/cifar10/ResNet34/1-models/', fname)
    else:
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/svhn/ResNet34/1-models/',fname)
    #f = open(fname, 'r')
    df = pd.read_csv(fname, sep=',' )#delimiter="\t")
    # print(df.info())
    # print(df['accuracy'])
    ax = sns.lineplot(data=df, x='queries',
                      y='entropy', label = "Random")
    plt.xlabel("Queries")
    if dataset == "mnist":
        plt.ylabel("Entropy")
    else:
        plt.ylabel("")
    plt.xlim([0,8000])
    if dataset == "cifar10":
        plt.ylim([0,250])
    plt.title(f'{dataset.upper()}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        f'/ssd003/home/akaleem/capc-learning-main/overleafgraphs/mixmatch/{dataset}/ent.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

#
# #Gap
for dataset in datasets:
    gap = []
    gap.append(0)
    for i in queries:
        fname = f'{dataset}@{str(i)}new/stats.txt'
        f = open(
            os.path.join(
                '/ssd003/home/akaleem/capc-learning-main/MixMatch-pytorch',
                fname), 'r')
        f.readline()
        a = f.readline()
        b = a.find(":")
        gap.append(float(a[b+1:]))
        f.close()
    plt.plot(queries2, gap, label = "MixMatch")
    fname = 'random/log_raw_gap_random.txt'
    if dataset == "mnist":
        fname = os.path.join(f'/ssd003/home/akaleem/capc-learning-main/adaptive-model/mnist/MnistNetPate/1-models/', fname)
    elif dataset == "cifar10":
        fname = 'random/log_raw_gap_random.txt'
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/cifar10/ResNet34/1-models/', fname)
    else:
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/svhn/ResNet34/1-models/',fname)
    df = pd.read_csv(fname, sep=',' )#delimiter="\t")
    # print(df.info())
    # print(df['accuracy'])
    ax = sns.lineplot(data=df, x='queries',
                      y='gap', label = "Random")
    plt.xlabel("Queries")
    plt.xlim([0,8000])
    if dataset == "mnist":
        plt.ylabel("Gap")
    else:
        plt.ylabel("")
    if dataset == "cifar10":
        plt.ylim([0, 250])
    plt.title(f'{dataset.upper()}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        f'/ssd003/home/akaleem/capc-learning-main/overleafgraphs/mixmatch/{dataset}/gap.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
#
#
# # pkNN
queries3= [0, 50, 100, 250, 500, 1000, 4000, 8000]
entropym =  [0, 0.25, 3.838, 5.30, 5.3, 6.45, 10.65, 13.96]
entropyc = [0, 2.33, 2.34, 3.32, 3.32, 5.30, 15.44, 18.44]
entropys =  [0,1.642, 2.78, 2.800, 6.527, 8.68, 18.77, 29.68]
for dataset in datasets:
    fname = 'random/log_raw_pkNN_cost_random.txt'
    if dataset == "mnist":
        fname = os.path.join(f'/ssd003/home/akaleem/capc-learning-main/adaptive-model/mnist/MnistNetPate/1-models/', fname)
    elif dataset == "cifar10":
        fname = 'random/log_raw_pkNN_cost_random.txt'
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/cifar10/ResNet34/1-models/', fname)
    else:
        fname = os.path.join('/ssd003/home/akaleem/capc-learning-main/adaptive-model/svhn/ResNet34/1-models/',fname)
    df = pd.read_csv(fname, sep=',' )#delimiter="\t")
    print(df)
    ax = sns.lineplot(data=df, x='queries',
                      y='pknn', label = "Random")
    if dataset == "mnist":
        plt.plot(queries3, entropym, label = "MixMatch")
    elif dataset == "cifar10":
        plt.plot(queries3, entropyc, label = "MixMatch")
    elif dataset == "svhn":
        plt.plot(queries3, entropys, label = "MixMatch")
    plt.xlabel("Queries")
    plt.xlim([0,8000])
    if dataset == "mnist":
        plt.ylabel("Privacy")
    else:
        plt.ylabel("")
    plt.title(f'{dataset.upper()}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(
        f'/ssd003/home/akaleem/capc-learning-main/overleafgraphs/mixmatch/{dataset}/pknn.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()
