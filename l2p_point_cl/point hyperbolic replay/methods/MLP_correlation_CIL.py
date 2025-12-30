import copy
import random

import ot
from geomloss import SamplesLoss
import models
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from types import MethodType
# from utils import utils
import utils.optim
import numpy as np
from geotorch import orthogonal
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
# import pymanopt.solvers
import time
import os, sys
from utils import wasserstein_fusion
from geodesic_algorithms.GFK_distill_normalise import GFK
from utils import orthogonal_loss
from tqdm import tqdm
from .poincareball import PoincareBall
from hyptorch.pmath import logmap0

gfk = GFK()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


setup_seed(114514)

cur_points = torch.empty(0)
ref_points = torch.empty(0)

hyper_feature1 = torch.empty(0)
hyper_feature2 = torch.empty(0)
hyper_feature3 = torch.empty(0)

cur_features = torch.empty(0)
cur_num = 0
ref_features = torch.empty(0)
ref_num = 0
# point_dict = {'table': 0, 'chair': 1, 'airplane': 2, 'car': 3, 'file': 4, 'stove': 5, 'rocket': 6, 'sofa': 7,
#               'rifle': 8, 'pillow': 9, 'guitar': 10, 'bag': 11, 'lamp': 12, 'telephone': 13, 'earphone': 14,
#               'clock': 15, 'cabinet': 16, 'pistol': 17, 'cap': 18, 'vessel': 19, 'tower': 20, 'laptop': 21, 'bed': 22,
#               'bottle': 23, 'train': 24, 'bookshelf': 25, 'keyboard': 26, 'monitor': 27, 'microphone': 28, 'washer': 29,
#               'bus': 30, 'jar': 31}
point_dict = {'chair': 0,
              'sofa': 1,
              'airplane': 2,
              'bookshelf': 3,
              'bed': 4,
              'vase': 5,
              'monitor': 6,
              'table': 7,
              'toilet': 8,
              'bottle': 9,
              'mantel': 10,
              'tv_stand': 11,
              'plant': 12,
              'piano': 13,
              'car': 14,
              'desk': 15,
              'dresser': 16,
              'night_stand': 17,
              'glass_box': 18,
              'guitar': 19,
              'range_hood': 20,
              'bench': 21,
              'cone': 22,
              'tent': 23,
              'flower_pot': 24,
              'laptop': 25,
              'keyboard': 26,
              'curtain': 27,
              'bathtub': 28,
              'sink': 29,
              'lamp': 30,
              'stairs': 31,
              'door': 32,
              'radio': 33,
              'xbox': 34,
              'stool': 35,
              'person': 36,
              'wardrobe': 37,
              'cup': 38,
              'bowl': 39}

pos_features = torch.empty(0)  # 存储正样本的特征
now_features = torch.empty(0)  # 存储当前样本的特征


def get_ref_feature(self, inputs, outputs):
    global ref_features, ref_num
    ref_features = inputs[0]
    # ref_features = outputs
    # ref_features = F.normalize(outputs, dim = 1)
    # print(len(inputs))
    # ref_num += inputs.shape[0]


def get_hyper_feature1(self, inputs, outputs):
    global hyper_feature1
    hyper_feature1 = inputs[0]


def get_hyper_feature2(self, inputs, outputs):
    global hyper_feature2
    hyper_feature2 = inputs[0]


def get_hyper_feature3(self, inputs, outputs):
    global hyper_feature3
    hyper_feature3 = inputs[0]


def get_cur_feature(self, inputs, outputs):
    global cur_features, cur_num
    cur_features = inputs[0]


def get_ref_points(self, inputs, outputs):
    global ref_points
    ref_points = inputs[0]


def get_cur_points(self, inputs, outputs):
    global cur_points
    cur_points = inputs[0]


def get_pos_features(self, inputs, outputs):
    global pos_features
    pos_features = inputs[0]


def get_now_features(self, inputs, outputs):
    global now_features
    now_features = inputs[0]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def hype_triplet_losses(parent_mu, pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim=256, one_child=False,
                        opposite_hier=False):
    if one_child:
        neg_child_mu = torch.flip(parent_mu, [0])
    else:
        neg_child_mu = torch.flip(pos_child_mu, [0])

    ball = PoincareBall(c=1.0, dim=ball_dim)
    parent_norm = torch.norm(parent_mu, dim=1)  # dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
    pos_norm = torch.norm(pos_child_mu, dim=1)

    par_norm = ball.dist0(parent_mu)
    p_norm = ball.dist0(pos_child_mu)  # .pow(2).sum(1)

    distance_positive = ball.dist(parent_mu, pos_child_mu)  # , keepdim=True).pow(2).sum(1)
    distance_negative = ball.dist(parent_mu, neg_child_mu)  # , keepdim=True).pow(2).sum(1)

    distance = distance_positive - distance_negative + contr_margin
    # distance = -distance_positive + distance_negative + mar_dist
    if opposite_hier:
        norm = par_norm - p_norm + hier_margin
    else:
        norm = -par_norm + p_norm + hier_margin  # Standard version wich improves the baseline
    # norm = par_norm-p_norm + margin                        #Possible change: parent becomes the mean of its children --EXperiment 07--

    triplet = torch.mean(
        torch.max(distance, torch.zeros_like(distance)))  # F.relu(distance_positive - distance_negative + 0.5 )
    hierarch = torch.mean(torch.max(norm, torch.zeros_like(norm)))  # F.relu(-par_norm+p_norm + margin)

    return parent_norm.mean(), pos_norm.mean(), distance_positive.mean(), distance_negative.mean(), triplet, hierarch


class MLP_correlation(pl.LightningModule):
    def __init__(self, args, outdim):
        super().__init__()
        self.args = args
        self.model = self.create_model(outdim=outdim)
        self.criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.episodic_mem_size = args.mem_size
        self.mem_data = torch.zeros(self.episodic_mem_size, 2048, 3).to(device)
        self.mem_label = torch.zeros(self.episodic_mem_size).long().to(device)
        self.mem_used = 0
        self.data_seen = 0
        self.eps_mem_batch = args.mem_batch
        self.diameter = list()
        self.idxMax = None
        # self.model.summary()
        # print(self.model.last.weight.shape)

    def create_model(self, outdim):
        model = models.mlp.HyperPointNet(outdim)
        # model.weights_init()
        return model

    def forward(self, x):
        return self.model.forward(x)

    def weight_init(self):
        for param in self.parameters():
            if len(param.shape) == 2:
                param /= param.norm(dim=1, keepdim=True)
            elif len(param.shape) == 1:
                param /= param.norm()

    def configure_optimizers(self):
        # print(self.named_parameters())
        # 标准
        return torch.optim.Adam(self.parameters(), lr=0.001)
        # return torch.optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9)
        # return utils.optim.ObliqueOpt(self.parameters(), lr = 0.1, momentum = 0.9)
        # return utils.optim.ObliqueOpt(self.parameters(), lr = 0.01)
        # return torch.optim.Adam(self.parameters(), lr = 1e-3)
        # 乘法实验
        # return torch.optim.SGD(self.parameters(), lr = 0.05 / (self.args.multiple ** 2))

        # 除法实验
        # return torch.optim.SGD(self.parameters(), lr = 0.1 * self.args.multiple)

        # return torch.optim.SGD(self.parameters(), lr = 0.003)
        # return torch.optim.Adadelta(self.parameters(),lr= 0.3)
        # return torch.optim.Adam(self.parameters(), lr = 0.1 / self.args.multiple)

    def training_step(self, batch, batch_idx):
        # print(torch.diag(self.model.linear[0].weight @ self.model.linear[0].weight.T))
        x, y, task = batch
        unique_task = torch.unique(task)
        loss = 0
        for i in range(unique_task.shape[0]):
            ttask = unique_task[i]
            tx = x[task == ttask]
            ty = y[task == ttask]
            tt = task[task == ttask]
            y_hat = self.model(tx)
            loss += self.criterion(y_hat, ty)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, task = batch
        unique_task = torch.unique(task)
        cnt, acc, loss = 0, 0, 0
        for i in range(unique_task.shape[0]):
            ttask = unique_task[i]
            tx = x[task == ttask]
            ty = y[task == ttask]
            tt = task[task == ttask]
            y_hat = self.model(tx, tt)
            loss += self.criterion(y_hat, ty)
            acc += (y_hat.argmax(dim=1) == ty).float().sum().item()
            cnt += y_hat.shape[0]
        acc = acc / cnt
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def accuracy(self, output, target):
        return (output.argmax(dim=1) == target).float().mean().item()

    def cosine_similarity(self, x, y):
        return torch.dot(x.view(-1), y.view(-1)) / (x.norm() * y.norm())

    def similarity(self, ref_model):
        tot_similarity = 0
        param_dict = dict(ref_model.named_parameters())
        for name, param in self.model.named_parameters():
            # view parameter matrix as a tensor, better performance
            tot_similarity += 1 - self.cosine_similarity(param, param_dict[name])
            # norm_1 = param.norm(dim = 1, keepdim = True)
            # norm_2 = param_dict[name].norm(dim = 1, keepdim = True)
            # tot_similarity += 1 - torch.diag(param @ param_dict[name].T / (norm_1 @ norm_2.T)).mean()
        return tot_similarity

    def class_incremental(self, cl_number):
        '''

        :param cl_number: The number of incremental class
        :return:
        '''
        in_features = self.model.last.in_features
        out_features = self.model.last.out_features
        new_out_features = out_features + cl_number

        # new classifier
        new_last = nn.Linear(in_features, new_out_features)
        nn.init.constant(new_last.weight, 0)
        new_last.weight.data[:out_features, :] = self.model.last.weight.data
        return new_last

    def get_grad_projector(self, ref_model) -> dict:
        # param_dict = dict(ref_model.named_parameters())
        projector_dict = dict()
        for name, p in ref_model.named_parameters():
            if len(p.shape) == 2:
                U, S, Vh = torch.linalg.svd(p.T, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                # r = np.sum(np.cumsum(sval_ratio) < threshold[i])  # +1
                r = torch.sum(torch.cumsum(sval_ratio, 0) < 0.99)  # +1
                projector_dict[name] = U[:, 0:r]
                projector_dict[name] = projector_dict[name] @ projector_dict[name].T
                # feature_list.append(U[:, 0:r])
        return projector_dict

    def curvature(self, points, k):

        # 将点云数据转换为合适的形状
        x = points.permute(0, 2, 1)  # 形状变为 (n, num_points, num_features)

        # 计算距离矩阵
        dist = torch.cdist(x, x)

        # 排除自身距离，然后排序并选取前k个最近邻距离
        sorted_distances, _ = dist.sort(dim=-1)
        distances = sorted_distances[:, :, 1:k + 1]  # 排除自身，保留最近的k个距离

        # 计算曲率信息
        curvature_info = 1.0 / (1e-8 + distances.mean(dim=-1, keepdim=True))

        return curvature_info

    def get_children_np(self, data, starting=2047, kmin=100, kmax=500):
        # torch.seed(10)

        k = random.randint(kmin, kmax)
        idknn = self.knn(data[:, :3, :], k)
        # print(idknn.shape)
        pos_child = data[:, :, :k]

        for id in range(data.shape[0]):
            starting_point = random.randint(0, starting)
            pos_child[id, :, :] = data[id, :, idknn[id, starting_point, :]]

        mar = 1000. / k
        return mar, pos_child, k - 1

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def train(self, train_loader, val_loader, val_original, task_num=0, ref_model=None):

        global pos_features
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        if task_num > 0:  # 冻结之前的head
            pre_out = task_num * 2
            # pre_out = (task_num - 1) * 2 + 4
            self.model.freeze_layer(self.model.output_layer[0: pre_out])

        self.model.train()  # 将model设置为训练模式
        opt = self.configure_optimizers()  # 设置为config里的优化器，默认为SGD（随机梯度下降）
        # print('<================== Task {} ===================>'.format(task_num))

        # if ref_model is not None:
        #     projector = self.get_grad_projector(ref_model)

        hyper_hook1 = self.model.shared_layer[0].register_forward_hook(get_hyper_feature1)
        # hyper_hook2 = self.model.shared_layer[3].register_forward_hook(get_hyper_feature2)
        # hyper_hook3 = self.model.output_layer[0].register_forward_hook(get_hyper_feature3)

        if ref_model is not None:
            ref_model.eval()
            cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)
            ref_hook = ref_model.output_layer[0].register_forward_hook(get_ref_feature)
            # cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)
            # ref_points_hook = ref_model.conv_fuse[2].register_forward_hook(get_ref_points)
            # cur_points_hook = self.model.conv_fuse[2].register_forward_hook(get_cur_points)
            ref_points_hook = ref_model.feature_layer3[0].register_forward_hook(get_ref_points)
            cur_points_hook = self.model.feature_layer3[0].register_forward_hook(get_cur_points)

        best_Acc = 0

        for e in range(30):
            tot_loss = 0
            tot_size = 0
            end = time.time()
            for batch in tqdm(train_loader):
                # x = batch['pointcloud'].to(device)
                x = batch[0].to(device)
                if x.shape[0] == 1:
                    continue
                # y_tmp = batch['cate']
                # y = torch.zeros(len(y_tmp)).long()
                # for i in range(len(y_tmp)):
                #     y[i] = point_dict[y_tmp[i]]
                # y = y.to(device)
                y = batch[1].squeeze(-1).long().to(device)
                y_hat = self.model(x)

                # 任务增量------------------------------------------------------------------------------------------------
                # if task_num > 0:
                #     y_hat = y_hat[:, task_num * 2: task_num * 2 + 2].to(device)
                #     y = y - task_num * 2
                loss = self.criterion(y_hat, y)  # 计算损失函数，默认为CrossEntropyLoss
                y_hat.to('cpu')
                # loss = self.criterion(y_hat, y)  # 计算损失函数，默认为CrossEntropyLoss

                # 获取patch
                x_trans = x.permute(0, 2, 1)
                mar_par, par_data, n_points = self.get_children_np(x_trans, kmin=1600,
                                                                   kmax=2048)  # 随机获取1600 - 2048个点的patch
                mar, pos_data, _ = self.get_children_np(par_data, starting=n_points, kmin=400, kmax=1000)
                par_data = par_data.permute(0, 2, 1)
                pos_data = pos_data.permute(0, 2, 1)


                self.model(par_data)
                hyper_par1 = hyper_feature1[:, :]
                # hyper_par2 = hyper_feature2[:, :]
                # hyper_par3 = hyper_feature3[:, :]

                self.model(pos_data)
                hyper_pos1 = hyper_feature1[:, :]
                # hyper_pos2 = hyper_feature2[:, :]
                # hyper_pos3 = hyper_feature3[:, :]

                criterion2 = hype_triplet_losses
                pn, posn, pd, nd, t_loss, h_loss = criterion2(hyper_par1, hyper_pos1, hier_margin=mar, contr_margin=4,
                                                              ball_dim=256)

                loss += 0.1 * t_loss + 0.1 * h_loss

                # pn, posn, pd, nd, t_loss, h_loss = criterion2(hyper_par2, hyper_pos2, hier_margin=mar, contr_margin=4,
                #                                               ball_dim=256)
                #
                # loss += 0.1 * t_loss + 0.1 * h_loss
                #
                # pn, posn, pd, nd, t_loss, h_loss = criterion2(hyper_par3, hyper_pos3, hier_margin=mar, contr_margin=4,
                #                                               ball_dim=256)
                #
                # loss += 0.1 * t_loss + 0.1 * h_loss



                if ref_model is not None:
                    # # 对比学习1
                    # pos_x = x[-1:]
                    # pos_y = y[-1]
                    # pos_hook = self.model.output_layer[0].register_forward_hook(get_pos_features)
                    # pos_y_hat = self.model(pos_x)
                    # pos_hook.remove()
                    # contrastive_loss = 0
                    #
                    # total_features = torch.empty(0).to(device)
                    # now_hook = self.model.output_layer[0].register_forward_hook(get_now_features)
                    # for i in range(50):
                    #     tmp = self.mem_data[i * 10: (i + 1) * 10]
                    #     with torch.no_grad():
                    #         self.model(tmp)
                    #     total_features = torch.cat((total_features, now_features))
                    # now_hook.remove()
                    # pos_features = F.normalize(pos_features, p=2, dim=1)
                    # total_features = F.normalize(total_features, p=2, dim=1)
                    # distances = torch.norm(pos_features - total_features, dim=1)
                    #
                    # for i in range(500):
                    #     now_y = self.mem_label[i]
                    #     if pos_y == now_y:
                    #         contrastive_loss += distances[i]
                    #     else:
                    #         contrastive_loss -= distances[i]
                    #
                    # contrastive_loss /= self.mem_used
                    # contrastive_loss = max(contrastive_loss + 10, 0)
                    # loss += contrastive_loss

                    # # 对比学习2
                    # contrastive_loss = 0
                    # total_features = torch.empty(0).to(device)
                    # now_hook = self.model.output_layer[0].register_forward_hook(get_now_features)
                    # for i in range(50):
                    #     tmp = self.mem_data[i * 10: (i + 1) * 10]
                    #     with torch.no_grad():
                    #         self.model(tmp)
                    #     total_features = torch.cat((total_features, now_features))
                    # now_hook.remove()
                    # total_features = F.normalize(total_features, p=2, dim=1)
                    # weight1 = F.normalize(self.model.output_layer[task_num * 2 + 2].weight, p=2, dim=1)
                    # weight2 = F.normalize(self.model.output_layer[task_num * 2 + 3].weight, p=2, dim=1)
                    # mul1 = torch.matmul(total_features, weight1.t())
                    # mul2 = torch.matmul(total_features, weight2.t())
                    #
                    # for i in range(500):
                    #     contrastive_loss += (mul1[i] + mul2[i])
                    #
                    # contrastive_loss /= self.mem_used
                    # contrastive_loss = max(contrastive_loss + 10, 0)
                    # loss += torch.squeeze(contrastive_loss)

                    ref_model = ref_model.to(device)
                    mm_loss = self.similarity(ref_model)  # 计算余弦相似度？
                    loss += mm_loss
                    # fusing feature manifold in replay
                    er_mem_indices = np.random.choice(self.mem_used, min(self.mem_used, self.eps_mem_batch),
                                                      replace=False)  # 生成一个以eps_mem_batch（默认为16）大小的一维数组，其中数值为0-mem_used的随机数
                    er_mem_indices = torch.from_numpy(er_mem_indices).to(x.device).long()
                    old_x = self.mem_data[er_mem_indices]
                    old_y = self.mem_label[er_mem_indices]
                    with torch.no_grad():
                        old_y_ref_hat = ref_model(old_x)
                    old_y_hat = self.model(old_x)

                    # 关系蒸馏
                    # cur_feature_result = torch.matmul(cur_features, cur_features.T)
                    # ref_feature_result = torch.matmul(ref_features, ref_features.T)
                    # feature_mse_loss = F.mse_loss(cur_feature_result, ref_feature_result) / 1600000
                    # loss += feature_mse_loss

                    # 曲率蒸馏
                    # cur_curve = self.curvature(cur_points, 20).squeeze(-1)  # 移除最后一维尺寸为1的部分
                    # ref_curve = self.curvature(ref_points, 20).squeeze(-1)
                    # curvature_loss = self.feature_matching_loss(cur_curve, ref_curve)
                    # # loss += 10 * curvature_loss
                    # loss += curvature_loss

                    # w距离蒸馏
                    w_loss = self.compute_wasserstein_distance_torch(cur_points, ref_points).mean()
                    loss += w_loss

                    feature_loss = self.feature_matching_loss(cur_features, ref_features)  # 原始的蒸馏方式  16*256
                    loss += feature_loss

                    # lamda = 10
                    # loss += gfk.fit(ref_features.detach(), cur_features) * lamda  # grassmann方式

                    # 任务增量------------------------------------------------------------------------------------------------
                    old_y_hat_alter = torch.zeros(old_y_hat.shape[0], 2).to("cuda")
                    for k in range(old_y.shape[0]):
                        now = old_y[k] // 2
                        old_y_hat_alter[k] = old_y_hat[k, now * 2: now * 2 + 2]
                        old_y[k] = old_y[k] - now * 2
                    replay_loss = self.criterion(old_y_hat_alter, old_y)  # 计算旧模型预测的loss
                    old_y_hat_alter.cpu()

                    # replay_loss = self.criterion(old_y_hat, old_y)  # 计算旧模型预测的loss

                    loss += replay_loss

                    margin_loss = orthogonal_loss.margin_loss(self.model.output_layer[0: pre_out],
                                                              self.model.output_layer[pre_out: pre_out + 2])
                    loss += margin_loss  # 添加margin损失

                    # print(f"mm_loss = {mm_loss}, feature_matching_loss = {feature_loss}, replay_loss = {replay_loss}, margin_loss = {margin_loss}")

                opt.zero_grad()
                loss.backward()
                tot_loss += loss.item()
                tot_size += 1
                opt.step()
                # if ref_model is not None:
                #     print(f'similarity: {self.similarity(ref_model)}')

            Acc = self.validation(val_loader, task_num)  # 计算当前轮次验证集的准确率（从开始到现在所有类别）
            Acc_original = self.validation_original(val_original, task_num)

            time_cost = time.time() - end
            if ref_model is not None:
                print(
                    f'<=== epoch: {e}    loss: {tot_loss / tot_size}   acc: {Acc}   acc_original:{Acc_original}   time: {time_cost}    best_acc: {best_Acc}  similarity: {self.similarity(ref_model)}===>')
                # print(f'<=== epoch: {e}    loss: {tot_loss / tot_size}   time: {time_cost}  matching_loss: {self.manifold_matching_loss(ref_model)} ===>')
            else:
                print(
                    f'<=== epoch: {e}    loss: {tot_loss / tot_size}    acc: {Acc}    time: {time_cost}    best_acc: {best_Acc} ===>')
            if Acc > best_Acc or Acc == best_Acc and (Acc_original >= best_ori_acc or task_num == 0):
            # if Acc > best_Acc or Acc == best_Acc and Acc_original >= best_ori_acc:
                torch.save(self.model.state_dict(), r'checkpoint/pointcloud/PointBest.pkl')
                best_Acc = Acc
                best_ori_acc = Acc_original
        # Model fusion
        # print(f'Acc: {self.validation(val_loader)}')
        # if ref_model is not None:
        #     self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[self.model, ref_model])
        # self.model = wasserstein_fusion.get_wassersteinized_layers_modularized(self.args,[ref_model, self.model])
        #
        # if ref_model is not None:
        #     param_dict = dict(ref_model.named_parameters())
        #     for name, p in self.model.named_parameters():
        #         p.data = 0.2 * param_dict[name].data + 0.8 * p.data

        self.model.load_state_dict(torch.load(r'checkpoint/pointcloud/PointBest.pkl'))  # 选取准确率最好的模型作为当前模型
        self.update_memory(train_loader)
        # self.calc_diameter()
        hyper_hook1.remove()
        # hyper_hook2.remove()
        # hyper_hook3.remove()
        if (ref_model is not None):
            ref_hook.remove()
            cur_hook.remove()
            ref_points_hook.remove()
            cur_points_hook.remove()
        print(f"Ori_Acc: {best_ori_acc}")
        return best_Acc

    def validation(self, val_loader, task_num):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        cnt, acc = 0, 0
        for batch in val_loader:
            # x = batch['pointcloud'].to(device)
            x = batch[0].to(device)
            # y_tmp = batch['cate']
            # y = torch.zeros(len(y_tmp)).long()
            # for i in range(len(y_tmp)):
            #     y[i] = point_dict[y_tmp[i]]
            # y = y.to(device)
            y = batch[1].squeeze(-1).to(device)
            with torch.no_grad():
                y_hat = self.model(x)
                # 任务增量------------------------------------------------------------------------------------------------
                # if task_num > 0:
                #     y_hat_alter = torch.zeros(y_hat.shape[0], 2).to("cuda")
                #     for k in range(y_hat.shape[0]):
                #         now = y[k] // 2
                #         y_hat_alter[k] = y_hat[k, now * 2: now * 2 + 2]
                #         y[k] = y[k] - now * 2
                #     acc += (y_hat_alter.argmax(dim=1) == y).float().sum().item()
                #     y_hat_alter.cpu()
                # else:
                #     acc += (y_hat.argmax(dim=1) == y).float().sum().item()
                acc += (y_hat.argmax(dim=1) == y).float().sum().item()
                # acc += (y_hat.argmax(dim=1) == y).float().sum().item()

                cnt += y_hat.shape[0]
        return acc / cnt

    def validation_original(self, val_loader, task_num):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.eval()
        cnt, acc = 0, 0
        for batch in val_loader:
            # x = batch['pointcloud'].to(device)
            x = batch[0].to(device)
            # y_tmp = batch['cate']
            # y = torch.zeros(len(y_tmp)).long()
            # for i in range(len(y_tmp)):
            #     y[i] = point_dict[y_tmp[i]]
            # y = y.to(device)
            y = batch[1].squeeze(-1).to(device)
            with torch.no_grad():
                y_hat = self.model(x)
                # 任务增量------------------------------------------------------------------------------------------------
                # if task_num > 0:
                #     y_hat = y_hat[:, : 2]

                acc += (y_hat.argmax(dim=1) == y).float().sum().item()
                cnt += y_hat.shape[0]
        return acc / cnt

    def pairwise_distances(self, x, y):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        # x = x / x.norm(dim = 1)
        # y = y / y.norm(dim = 1)
        # x = F.normalize(x, dim = 1)
        # y = F.normalize(y, dim = 1)
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def poincare_distance(self, x, y):
        # Clip input points to avoid numerical instability in arccosh
        x = torch.clamp(x, min=-1.0 + 1e-15, max=1.0 - 1e-15)
        y = torch.clamp(y, min=-1.0 + 1e-15, max=1.0 - 1e-15)

        # Compute hyperbolic distance
        dist = torch.acosh(1 + 2 * torch.norm(x - y, dim=-1) ** 2 / (
                (1 - torch.norm(x, dim=-1) ** 2) * (1 - torch.norm(y, dim=-1) ** 2)))

        return dist.mean()

    def manifold_matching_loss(self, ref_model):
        tot_similarity = 0
        tot_manifold_matching = 0
        param_dict = dict(ref_model.named_parameters())
        for name, param in self.model.named_parameters():
            tot_manifold_matching += self.geodesic_dist(param, param_dict[name])

            # if len(param.shape) == 2:
            #     dis_s = self.pairwise_distances(param, param)
            #     dis_t = self.pairwise_distances(param_dict[name], param_dict[name])
            #     tot_manifold_matching += torch.dist(dis_s,dis_t,2) + torch.dist(param, param_dict[name], 2)

        return tot_manifold_matching

    def geodesic_dist(self, X, Y):
        # return (X - Y).norm()
        XY = (X.data * Y.data).sum(0)
        XY[XY > 1] = 1
        U = torch.arccos(XY)
        return torch.norm(U)

    def feature_matching_loss(self, X, Y, eps=1e-9):
        # X += eps
        # Y += eps
        # X = X / X.norm(dim = 1)
        # Y = Y / Y.norm(dim = 1)
        X = F.normalize(X, dim=1)
        Y = F.normalize(Y, dim=1)
        # geodesic = self.Sphere_dist(X,Y).trace() / X.shape[0]
        # cost = X @ Y.T
        # geodesic = (cost - 2 * torch.diag(cost) + torch.eye(cost.shape[0]).to(cost.device)).mean()
        # geodesic = (-1 * torch.diag(cost) + torch.eye(cost.shape[0]).to(cost.device)).mean()
        # geodesic
        # geodesic = -1 * torch.eye(X.shape[0]) - torch.diag(X @ Y.T).mean()
        geodesic = torch.diag(self.pairwise_distances(X, Y)).mean()
        # geodesic = self.pairwise_distances(X,Y).mean()
        return geodesic

    def calc_diameter(self):
        cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)
        self.model(self.mem_data)
        features = F.normalize(cur_features, dim=1)
        center = features.mean(dim=0, keepdim=True)
        distance = self.pairwise_distances(center, features).squeeze()
        sorted_index = torch.argsort(distance)
        self.diameter.append(distance[sorted_index[-1]].item())
        cur_hook.remove()

    # # 直接替换更新策略
    # def update_memory(self, dataset_loader):
    #     self.model.eval()
    #     for batch in tqdm(dataset_loader):
    #         x_batch = batch[0]
    #         y_batch = batch[1].squeeze(-1)
    #         for index in range(x_batch.shape[0]):
    #             x, y = x_batch[index], y_batch[index]
    #             if self.episodic_mem_size > self.mem_used:  # 如果当前memory库存没有到上限，那就直接把x和y丢进库存里
    #                 self.mem_data[self.mem_used] = x
    #                 self.mem_label[self.mem_used] = y
    #                 self.mem_used += 1
    #             else:
    #                 j = np.random.randint(0, self.data_seen)
    #                 # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
    #                 if j < self.episodic_mem_size:
    #                     self.mem_data[j], self.mem_label[j] = x, y
    #                 # j = np.random.randint(0, self.data_seen)
    #                 # # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
    #                 # if j < self.episodic_mem_size:
    #                 #     self.mem_data[j], self.mem_label[j] = x, y
    #             self.data_seen += 1

    # # 旧的更新策略
    # def update_memory(self, dataset_loader):
    #     self.model.eval()
    #     cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)  # 钩子函数
    #     replay_feature, center, distance, sorted_index = None, None, None, None
    #     total_features = torch.empty(0).to("cuda:0")
    #     for batch in tqdm(dataset_loader):
    #         # x_batch = batch['pointcloud']
    #         x_batch = batch[0]
    #         # y_tmp = batch['cate']
    #         # y_batch = torch.zeros(len(y_tmp)).long()
    #         # for i in range(len(y_tmp)):
    #         #     y_batch[i] = point_dict[y_tmp[i]]
    #         y_batch = batch[1].squeeze(-1)
    #         for index in range(x_batch.shape[0]):
    #             x, y = x_batch[index], y_batch[index]
    #             if self.episodic_mem_size > self.mem_used:  # 如果当前memory库存没有到上限，那就直接把x和y丢进库存里
    #                 self.mem_data[self.mem_used] = x
    #                 self.mem_label[self.mem_used] = y
    #                 self.mem_used += 1
    #             else:
    #                 if center == None:
    #                     for i in range(25):
    #                         tmp = self.mem_data[i * 10: (i + 1) * 10]
    #                         with torch.no_grad():
    #                             self.model(tmp)
    #                         total_features = torch.cat((total_features, cur_features))
    #                     replay_feature = F.normalize(total_features, dim=1)
    #                     center = replay_feature.mean(dim=0, keepdim=True)
    #                     distance = self.pairwise_distances(center, replay_feature).squeeze()
    #                     sorted_index = torch.argsort(distance)
    #                 with torch.no_grad():
    #                     self.model(torch.reshape(x, shape=(-1, 2048, 3)).cuda())
    #                 dis_to_center = torch.dist(F.normalize(cur_features, dim=1), center, 2) ** 2
    #                 if dis_to_center > distance[sorted_index[-1]]:
    #                     j = np.random.randint(0, self.mem_used)
    #                     self.mem_data[j], self.mem_label[j] = x, y
    #                     distance[j] = dis_to_center
    #                     sorted_index = torch.argsort(distance)
    #                 # elif dis_to_center >= distance[sorted_index[0]]:
    #                 else:
    #                     j = np.random.randint(0, self.data_seen)
    #                     # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
    #                     if j < self.episodic_mem_size:
    #                         self.mem_data[sorted_index[j]], self.mem_label[sorted_index[j]] = x, y
    #                         distance[sorted_index[j]] = dis_to_center
    #                         sorted_index = torch.argsort(distance)
    #                 # j = np.random.randint(0, self.data_seen)
    #                 # # if j < self.episodic_mem_size and distance[sorted_index[j]] <= dis_to_center:
    #                 # if j < self.episodic_mem_size:
    #                 #     self.mem_data[j], self.mem_label[j] = x, y
    #             self.data_seen += 1
    #     cur_hook.remove()

    # # 新的更新策略
    # def update_memory(self, dataset_loader):
    #     global cur_points
    #     self.model.eval()
    #     cur_hook = self.model.feature_layer3[0].register_forward_hook(get_cur_points)  # 钩子函数
    #     # cur_hook = self.model.output_layer[0].register_forward_hook(get_cur_feature)  # 钩子函数
    #     replay_feature, center, distance, sorted_index = None, None, None, None
    #     total_features = torch.empty(0).to("cuda:0")
    #     for batch in tqdm(dataset_loader):
    #         x_batch = batch[0]
    #         y_batch = batch[1].squeeze(-1)
    #         for index in range(x_batch.shape[0]):
    #             x, y = x_batch[index], y_batch[index]
    #             if self.episodic_mem_size > self.mem_used:  # 如果当前memory库存没有到上限，那就直接把x和y丢进库存里
    #                 self.mem_data[self.mem_used] = x
    #                 self.mem_label[self.mem_used] = y
    #                 self.mem_used += 1
    #             else:
    #                 if center == None:
    #                     for i in range(50):
    #                         tmp = self.mem_data[i * 10: (i + 1) * 10]
    #                         with torch.no_grad():
    #                             self.model(tmp)
    #                         ########
    #                         # cur_points = torch.max(cur_points, dim=-1, keepdim=False)[0]
    #                         ########
    #                         total_features = torch.cat((total_features, cur_points))
    #                     replay_feature = F.normalize(total_features, dim=1)
    #                     center = replay_feature.mean(dim=0, keepdim=True)
    #                     # distance = self.pairwise_distances(center, replay_feature).squeeze()
    #                     distance = []
    #                     for i in range(500):
    #                         now_dist = self.poincare_distance(center, replay_feature[i].reshape(-1, 256, 2048)).item()
    #                         distance.append(now_dist)
    #
    #                     # sorted_index = torch.argsort(distance)
    #
    #                     indexed_list = list(enumerate(distance))
    #                     sorted_indices = sorted(indexed_list, key=lambda x: x[1])
    #                     sorted_index = [index for index, value in sorted_indices]
    #
    #                     self.idxMax = sorted_index[-1]
    #                 with torch.no_grad():
    #                     self.model(torch.reshape(x, shape=(-1, 2048, 3)).cuda())
    #                 ########
    #                 # cur_points = torch.max(cur_points, dim=-1, keepdim=False)[0]
    #                 ########
    #                 # dis_to_center = torch.dist(F.normalize(cur_points, dim=1), center, 2) ** 2
    #                 dis_to_center = self.poincare_distance(F.normalize(cur_points, dim=1), center).item()
    #                 if dis_to_center > distance[self.idxMax]:
    #                     j = np.random.randint(0, self.mem_used)
    #                     self.mem_data[j], self.mem_label[j] = x, y
    #                     distance[j] = dis_to_center
    #                     self.idxMax = j
    #                 else:
    #                     j = np.random.randint(0, self.data_seen)
    #                     if j < self.episodic_mem_size and j != self.idxMax:
    #                         self.mem_data[j], self.mem_label[j] = x, y
    #                         distance[j] = dis_to_center
    #             self.data_seen += 1
    #     cur_hook.remove()

    # Algorithm 1: Hyperbolic Manifold Expansion Elimination
    def update_memory(self, dataset_loader):
        # Configuration for Algorithm 1
        alpha1 = 1.0  # Weight for local deviation
        alpha2 = 1.0  # Weight for global deviation
        c = 0.01      # Curvature (should match training)
        
        self.model.eval()
        
        # Feature storage
        local_features = []
        global_features = []
        
        def hook_local(module, input, output):
            local_features.append(output.detach())
            
        def hook_global(module, input, output):
            global_features.append(output.detach())

        # Register hooks
        # Local: shared_layer[2] (after first ReLU, 512 dim)
        # Global: shared_layer (output of sequential, 256 dim)
        handle_local = self.model.shared_layer[2].register_forward_hook(hook_local)
        handle_global = self.model.shared_layer.register_forward_hook(hook_global)

        # Helper to extract features for a batch of points
        def extract_features(x_input):
            local_features.clear()
            global_features.clear()
            with torch.no_grad():
                self.model(x_input)
            return local_features[0], global_features[0]

        # 1. Pre-compute Buffer stats if buffer has data
        buffer_local_vecs = []
        buffer_global_vecs = []
        
        if self.mem_used > 0:
            # Create batches for memory to avoid OOM
            mem_loader = DataLoader(
                torch.utils.data.TensorDataset(self.mem_data[:self.mem_used], self.mem_label[:self.mem_used]),
                batch_size=32, shuffle=False
            )
            for bx, by in mem_loader:
                bx = bx.cuda()
                l_feat, g_feat = extract_features(torch.reshape(bx, shape=(-1, 2048, 3)))
                # Map to Tangent Space at origin
                l_vec = logmap0(F.normalize(l_feat, dim=1), c=c)
                g_vec = logmap0(F.normalize(g_feat, dim=1), c=c)
                buffer_local_vecs.append(l_vec)
                buffer_global_vecs.append(g_vec)
            
            # Concatenate
            if buffer_local_vecs:
                V_local_M = torch.cat(buffer_local_vecs)
                V_global_M = torch.cat(buffer_global_vecs)
            else:
                 V_local_M = torch.empty(0, 512).cuda()
                 V_global_M = torch.empty(0, 256).cuda()
        else:
            V_local_M = torch.empty(0, 512).cuda()
            V_global_M = torch.empty(0, 256).cuda()

        # Helper to compute Centroid and Radius (Average Distance)
        def compute_stats(vectors):
            if vectors.shape[0] == 0:
                return torch.zeros(vectors.shape[1]).cuda(), 0.0
            centroid = vectors.mean(dim=0)
            # Radius = Average Euclidean distance in Tangent Space
            dists = torch.norm(vectors - centroid, dim=1)
            radius = dists.mean().item()
            return centroid, radius

        # Initial stats
        c_local, d_local = compute_stats(V_local_M)
        c_global, d_global = compute_stats(V_global_M)

        print(f"Update Memory: Start. Buffer Used: {self.mem_used}. c_local norm: {c_local.norm().item():.4f}, d_local: {d_local:.4f}")

        # 2. Iterate new data
        for batch in tqdm(dataset_loader, desc="Updating Memory (Algo 1)"):
            x_batch = batch[0]
            y_batch = batch[1].squeeze(-1)
            
            # Extract features for batch
            l_feats, g_feats = extract_features(x_batch.cuda())
            
            # Map batch to tangent space
            l_vecs = logmap0(F.normalize(l_feats, dim=1), c=c)
            g_vecs = logmap0(F.normalize(g_feats, dim=1), c=c)

            for i in range(x_batch.shape[0]):
                x, y = x_batch[i], y_batch[i]
                v_local_x = l_vecs[i]
                v_global_x = g_vecs[i]

                # Case 1: Buffer not full
                if self.mem_used < self.episodic_mem_size:
                    self.mem_data[self.mem_used] = x
                    self.mem_label[self.mem_used] = y
                    
                    # Update Vectors
                    V_local_M = torch.cat([V_local_M, v_local_x.unsqueeze(0)])
                    V_global_M = torch.cat([V_global_M, v_global_x.unsqueeze(0)])
                    
                    self.mem_used += 1
                    # Recompute stats every step? (Expensive) -> Maybe periodic or incremental?
                    # For strict Algo 1, we should update. For speed, we update stats every N steps or just let it drift slightly.
                    # Here we do naive update for correctness as per user request.
                    c_local, d_local = compute_stats(V_local_M)
                    c_global, d_global = compute_stats(V_global_M)
                    
                else:
                    # Case 2: Buffer full, check expansion
                    dist_local = torch.norm(v_local_x - c_local)
                    dist_global = torch.norm(v_global_x - c_global)
                    
                    delta = alpha1 * (dist_local - d_local) + alpha2 * (dist_global - d_global)
                    
                    idx_to_replace = -1
                    
                    if delta > 0:
                        # Significant expansion: Random replace
                        idx_to_replace = np.random.randint(0, self.episodic_mem_size)
                    else:
                        # Reservoir sampling chance
                        r = np.random.randint(0, self.data_seen + 1) # data_seen is total steam count
                        if r < self.episodic_mem_size:
                            idx_to_replace = r
                    
                    if idx_to_replace != -1:
                        self.mem_data[idx_to_replace] = x
                        self.mem_label[idx_to_replace] = y
                        
                        # Update Vectors
                        V_local_M[idx_to_replace] = v_local_x
                        V_global_M[idx_to_replace] = v_global_x
                        
                        # Recompute stats
                        c_local, d_local = compute_stats(V_local_M)
                        c_global, d_global = compute_stats(V_global_M)

                self.data_seen += 1

        handle_local.remove()
        handle_global.remove()
        print(f"Update Memory: End. Buffer Used: {self.mem_used}. Stats updated.")

    def update_reservior(self, current_image, current_label):
        """
        Update the episodic memory with current example using the reservior sampling
        """
        if self.episodic_mem_size > self.data_seen:
            self.mem_data[self.data_seen] = current_image
            self.mem_label[self.data_seen] = current_label
            # self.mem_task_id[self.data_seen] = current_task
            self.mem_used += 1
        else:
            j = np.random.randint(0, self.data_seen)
            if j < self.episodic_mem_size:
                self.mem_data[j] = current_image
                self.mem_label[j] = current_label
                # self.mem_task_id[j] = current_task

        self.data_seen += 1

    # def update_reservior(self, current_image, current_label):
    #

    def Sphere_dist(self, x, y, eps=1e-9):
        # Make sure inner product is between -1 and 1
        # assert not torch.isnan(x) and not torch.isnan(y)
        inner = x @ y.T
        # torch.clamp_(inner, -1 + eps, 1 - eps)
        inner = torch.clamp(inner, min=-1, max=1)
        # inner = max(min(inner, 1), -1)
        # print(inner.max(),inner.min())
        assert inner.max() <= 1 and inner.min() >= -1
        return torch.arccos(inner)

    def cost_matrix(self, x, y, p=2):
        # 定义成本矩阵的计算函数
        x_col = x.unsqueeze(1)  # 将 x 扩展成 (1024, 1, 3) 的形状，为了计算成对距离
        y_lin = y.unsqueeze(0)  # 将 y 扩展成 (1, 1024, 3) 的形状，同上
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)  # 计算 x 和 y 之间的成对距离的 p 次方，然后对最后一个维度求和
        return c

    def compute_wasserstein_distance(self, cloud1, cloud2, p=2):
        # 计算两个点云之间的 Wasserstein 距离
        assert cloud1.shape == cloud2.shape, "两个点云必须有相同的形状。"

        # 为每个维度计算成本矩阵
        cost_matrices = [self.cost_matrix(cloud1[:, i].unsqueeze(1), cloud2[:, i].unsqueeze(1), p) for i in
                         range(cloud1.shape[1])]

        # 对每个维度计算 Wasserstein 距离
        distances = [ot.emd2([], [], cost_matrix.numpy()) for cost_matrix in cost_matrices]

        # 返回平均 Wasserstein 距离
        return np.mean(distances)

    # 创建一个 Wasserstein 距离计算的函数
    def compute_wasserstein_distance_torch(self, cloud1, cloud2, p=2, blur=0.01):
        cloud1 = cloud1.permute(0, 2, 1)
        cloud2 = cloud2.permute(0, 2, 1)

        cloud1 = F.normalize(cloud1, p=2, dim=2)
        cloud2 = F.normalize(cloud2, p=2, dim=2)

        # cloud1 和 cloud2 的形状应该是 (batch_size, num_points, dim)
        loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
        # 计算并返回 Wasserstein 距离
        return loss(cloud1.contiguous(), cloud2.contiguous())
