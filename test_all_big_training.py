# date:2020.6
# author:tianke

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn import mixture
from scipy.spatial.distance import cdist
from scipy.stats import norm
import scipy.io as sio

EPOCH_MAX = 30
batch = 309


def calculation_auc(ty_pe, ID, batch):
    p_auc = []
    auc = []
    llh1_llh1_std = []
    print(ty_pe)
    print(ID)
    for epoch in range(EPOCH_MAX):
        print(epoch)
        gmm = mixture.GaussianMixture()
        # all_traning_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID + '.npy')
        # all_val_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        # all_test_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID + '.npy')
        all_traning_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID + '.npy')
        all_val_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        all_test_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID + '.npy')
        all_traning_latent = all_traning_latent.reshape(-1, 30)
        all_val_latent = all_val_latent.reshape(-1, 30)
        all_test_latent = all_test_latent.reshape(-1, 30)
        # train1_train2_latent.append(np.mean(cdist(all_traning_latent, all_traning_latent, metric='euclidean')))
        gmm.fit(all_traning_latent)
        llh1 = gmm.score_samples(all_traning_latent)
        llh2 = gmm.score_samples(all_val_latent)
        llh3 = gmm.score_samples(all_test_latent)
        llh1 = llh1.reshape(batch, -1)
        llh2 = llh2.reshape(batch, -1)
        llh3 = llh3.reshape(batch, -1)

        llh1_llh2 = np.mean(llh1)-np.mean(llh2, axis=0)
        llh1_llh3 = np.mean(llh1)-np.mean(llh3, axis=0)

        y_true = [0]*(all_val_latent.shape[0]//batch) + [1]*(all_test_latent.shape[0]//batch)
        y_pred = np.concatenate((llh1_llh2, llh1_llh3), axis=0)
        y_pred = np.array(y_pred)
        p_auc.append(roc_auc_score(y_true, y_pred, max_fpr=0.1))
        auc.append(roc_auc_score(y_true, y_pred))
        llh1_llh1_std.append(np.mean(np.std(llh1, axis=0)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax2 = ax.twinx()
    lns1 = ax.plot(p_auc, label='p_auc', color='b')
    lns2 = ax.plot(auc, label='auc', color='y')
    lns3 = ax2.plot(llh1_llh1_std, color='g', linestyle='--', label=r'std(likelihood)')
    ax.set_xlabel('Epoch', font2)
    ax.set_ylabel('AUC/pAUC', font2)
    ax2.set_ylabel('std(likelihood)', font3)
    plt.title(ty_pe + '_' + ID, font2)
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, prop=font1)
    ax.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    plt.tight_layout()
    # if os.path.exists('small_sample/' + ty_pe):
    #     pass
    # else:
    #     os.mkdir('small_sample/' + ty_pe)
    # plt.savefig('small_sample/' + ty_pe + "/_" + ID + ".png")
    if os.path.exists('big_sample/' + ty_pe):
        pass
    else:
        os.mkdir('big_sample/' + ty_pe)
    plt.savefig('big_sample/' + ty_pe + "/_" + ID + ".png")
    plt.close()


# #################################### main ####################################################
for ty_pe in ['fan', 'slider', 'pump', 'valve', 'ToyCar', 'ToyConveyor']:
    if ty_pe == 'ToyCar':
        batch = 340
        for ID in ['id_01', 'id_02', 'id_03', 'id_04']:
            calculation_auc(ty_pe, ID, batch)
    elif ty_pe == 'ToyConveyor':
        batch = 309
        for ID in ['id_01', 'id_02', 'id_03']:
            calculation_auc(ty_pe, ID, batch)
    else:
        batch = 309
        for ID in ['id_00', 'id_02', 'id_04', 'id_06']:
            calculation_auc(ty_pe, ID, batch)

print('Good Luck')













