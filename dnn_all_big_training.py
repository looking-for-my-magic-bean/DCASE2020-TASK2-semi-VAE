import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from data import *
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# 变量设置
task = 'Task 2'
EPOCH_MAX = 30
block = 'LSTM'  # GRU , LSTM 可选
optimizer = 'Adam'  # SGD , Adam 可选
dropout = 0
latent_length = 30
batch_size = 309  # 309  340
input_size = 640
hidden1 = 128
hidden2 = 128
hidden3 = 64
hidden4 = 64
learning_rate = 0.00001

if torch.cuda.is_available:
    device = "cuda:0"
else:
    device = "cpu"
# device = "cpu"
print('device:' + device)
print('EPOCH_MAX:' + str(EPOCH_MAX))

data_path_dev = '/home/share/dataset/DCASE2020/Dcase2020_task2/data/dev/'
data_path_additional = '/home/share/dataset/DCASE2020/Dcase2020_task2/data/additional/'
data_path_eval = '/home/share/dataset/DCASE2020/Dcase2020_task2/data/eval/'


def init_layer(layer, nonlinearity='leaky_relu'):
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


def init_rnnLayers(rLayer):
    for param in rLayer.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.orthogonal_(param.data)
        else:
            torch.nn.init.normal_(param.data)


class Encoder(nn.Module):
    def __init__(self, input_size=input_size, hidden1=hidden1, hidden2=hidden2,
                 hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Encoder, self).__init__()

        # 定义属性
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.input_to_hidden1 = nn.Linear(self.input_size, self.hidden1)
        self.hidden1_to_hidden2 = nn.Linear(self.hidden1, self.hidden2)
        self.hidden2_to_hidden3 = nn.Linear(self.hidden2, self.hidden3)
        self.hidden3_to_hidden4 = nn.Linear(self.hidden3, self.hidden4)
        # self.hidden4_to_latent = nn.Linear(self.hidden4, self.latent_length)
        self.hidden4_to_mean = nn.Linear(self.hidden4, self.latent_length)
        self.hidden4_to_logvar = nn.Linear(self.hidden4, self.latent_length)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.hidden4_to_mean.weight)  # 为了通过网络层时，输入和输出的方差相同 服从均匀分布
        nn.init.xavier_uniform_(self.hidden4_to_logvar.weight)  # 为了通过网络层时，输入和输出的方差相同

        # self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, x):
        # img_seq_in = x.view(-1, x.size(0), x.size(1))  # [seq,bach,num_cha]=[154,16,8]
        # 线性变换
        hidden1 = self.ReLU(self.input_to_hidden1(x))
        hidden2 = self.ReLU(self.hidden1_to_hidden2(hidden1))
        hidden3 = self.ReLU(self.hidden2_to_hidden3(hidden2))
        hidden4 = self.ReLU(self.hidden3_to_hidden4(hidden3))
        self.latent_mean = self.hidden4_to_mean(hidden4)
        self.latent_logvar = self.hidden4_to_logvar(hidden4)
        std = torch.exp(0.5 * self.latent_logvar)
        eps = torch.randn_like(std)  # 定义一个和std一样大小的服从标准正太分布的张量
        latent = torch.mul(eps, std) + self.latent_mean  # 标准正太分布乘以标准差后加上均值 latent.shape(batch,latent_length)

        return latent, self.latent_mean, self.latent_logvar  # x.shape(sqe,batch,input)


class Decoder(nn.Module):
    def __init__(self, output_size=input_size, hidden1=hidden1,
                 hidden2=hidden2, hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Decoder, self).__init__()

        # 定义属性
        self.output_size = output_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.latent_to_hidden4 = nn.Linear(self.latent_length, self.hidden4)
        self.hidden4_to_hidden3 = nn.Linear(self.hidden4, self.hidden3)
        self.hidden3_to_hidden2 = nn.Linear(self.hidden3, self.hidden2)
        self.hidden2_to_hidden1 = nn.Linear(self.hidden2, self.hidden1)
        self.hidden1_to_output = nn.Linear(self.hidden1, self.output_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # self.init_weights()

    def init_weights(self):
        init_layer(self.deconv1)
        init_layer(self.deconv2)
        init_layer(self.deconv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, latent):
        # 反RNN+线性变换
        hidden4 = self.ReLU(self.latent_to_hidden4(latent))
        hidden3 = self.ReLU(self.hidden4_to_hidden3(hidden4))
        hidden2 = self.ReLU(self.hidden3_to_hidden2(hidden3))
        hidden1 = self.ReLU(self.hidden2_to_hidden1(hidden2))
        output = self.hidden1_to_output(hidden1)

        return output


class Autoencoder(nn.Module):
    def __init__(self, input_size=input_size, hidden1=hidden1,
                 hidden2=hidden2, hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
        self.decoder = Decoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)

    def forward(self, x):
        latent, latent_mean, latent_logvar = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent, latent_mean, latent_logvar


def calculation_latent(ty_pe, ID, data_path_train, data_path_test):
    print(ty_pe, ID)
    global optimizer, block, EPOCH_MAX, batch_size, learning_rate,\
        input_size, hidden1, hidden2, hidden3, hidden4, latent_length, device

    X_train = dnn_data_train(datadir=data_path_train, type=ty_pe, ID=ID)  # 训练数据
    # X_val = dnn_data_test(datadir=data_path, type=type, ID=ID)  # 单测试数据
    X_val, X_val_a = dnn_data_2test(datadir=data_path_test, type=ty_pe, ID=ID)  # 双测试数据

    autoencoder = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    if optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # 学习率，权重衰减
    else:
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)

    train_dataset = data.TensorDataset(torch.from_numpy(X_train))
    val_dataset = data.TensorDataset(torch.from_numpy(X_val))
    test_dataset = data.TensorDataset(torch.from_numpy(X_val_a))

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    validation_loader = data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}
    # data_loaders = {"train": train_loader, "test": test_loader}

    all_running_loss = []
    all_val_loss = []
    all_test_loss = []
    all_kl_loss = []
    all_traning_latent = []
    all_val_latent = []
    all_test_latent = []
    llh1_llh1_std = []
    optimizer.zero_grad()
    gmm = mixture.GaussianMixture()

    for epoch in range(EPOCH_MAX):
        for phase in ['train', 'val', 'test']:
            # for phase in ['train', 'test']:
            if phase == 'train':
                autoencoder.train()
            else:
                autoencoder.eval()

            for step, data_sample in enumerate(data_loaders[phase]):
                inputs = data_sample
                inputs = inputs[0]
                inputs = inputs.to(device)

                outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())
                latent_com = latent
                latent_out = latent_com.detach()
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                all_kl_loss.append(kl_loss.item())
                loss = criterion(inputs, outputs) + kl_loss

                if phase == 'train':
                    all_running_loss.append(loss.item())  # 统计所有训练loss
                    all_traning_latent.append(latent_out.cpu().numpy())  # 统计所有训练latent

                    loss.backward()
                    optimizer.step()

                elif phase == 'val':
                    all_val_loss.append(loss.item())  # 统计所有测试正常loss
                    all_val_latent.append(latent_out.cpu().numpy())  # 统计所有测试正常latent

                else:
                    all_test_loss.append(loss.item())  # 统计所有测试异常loss
                    all_test_latent.append(latent_out.cpu().numpy())  # 统计所有测试异常latent
                optimizer.zero_grad()

        running_loss = np.mean(all_running_loss)
        # val_loss = np.mean(all_val_loss)
        test_loss = np.mean(all_test_loss)
        kl_loss = np.mean(all_kl_loss)

        if os.path.exists('clustering/' + ty_pe + '/epoch/' + str(epoch)):
            pass
        else:
            # os.mkdir('clustering/' + ty_pe + '/epoch/' + str(epoch))  # 逐级创建
            os.makedirs('clustering/' + ty_pe + '/epoch/' + str(epoch))  # 一次创建到底

        np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID, all_traning_latent)
        all_traning_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) +
                                     '/all_traning_latent' + ID + '.npy')
        print('all_traning_latent.shape:', all_traning_latent.shape)

        np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID, all_val_latent)
        all_val_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        print('all_val_latent.shape:', all_val_latent.shape)

        np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID, all_test_latent)
        all_test_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) +
                                  '/all_test_latent' + ID + '.npy')
        print('all_test_latent.shape:', all_test_latent.shape)

        all_traning_latent = all_traning_latent.reshape(-1, 30)
        gmm.fit(all_traning_latent)
        llh1 = gmm.score_samples(all_traning_latent)
        llh1 = llh1.reshape(batch_size, -1)
        llh1_llh1_std.append(np.mean(np.std(llh1, axis=0)))

        f = open('clustering/' + ty_pe + '/llh1_llh1_std' + ID + '.txt', 'w')
        for ip in llh1_llh1_std:
            f.write(str(ip))
            f.write('\n')
        f.close()

        # 除最后一轮外每个epoch清零
        if epoch == EPOCH_MAX - 1:
            pass
        else:
            all_running_loss = []
            all_val_loss = []
            all_test_loss = []
            all_kl_loss = []
            all_traning_latent = []
            all_val_latent = []
            all_test_latent = []

        print('\n[Epoch: %3d] Train loss: %.5g Test loss: %.5g kl_loss: %.5g'
              % (epoch, float(running_loss), float(test_loss), float(kl_loss)))


# ############################################### main ################################################
for ty_pe in ['fan', 'slider', 'pump', 'valve', 'ToyCar', 'ToyConveyor']:
    if ty_pe == 'ToyCar':
        batch_size = 340
        for ID in ['id_01', 'id_02', 'id_03', 'id_04']:
            calculation_latent(ty_pe, ID,data_path_dev, data_path_dev)
    elif ty_pe == 'ToyConveyor':
        batch_size = 309
        for ID in ['id_01', 'id_02', 'id_03']:
            calculation_latent(ty_pe, ID, data_path_dev, data_path_dev)
    else:
        batch_size = 309
        for ID in ['id_00', 'id_02', 'id_04', 'id_06']:
            calculation_latent(ty_pe, ID, data_path_dev, data_path_dev)

print('Good Luck')









