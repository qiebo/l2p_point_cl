import torch
import torch.nn as nn
# from geotorch import Stiefel, orthogonal, grassmannian
# import geotorch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import time
from .pointnet_util import farthest_point_sample, index_points, square_distance
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, dist0, dist, dist_matrix_knn, _mobius_add
from .manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero, GyroplaneConvLayer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

setup_seed(114514)

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        #
        # b, 3, npoint, nsample
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, outdim):
        super().__init__()

        self.out_dim = 0
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))


        # 共享隐藏层
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)


        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)

    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1

    def forward(self, x):
        xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)

        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)

        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False




class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class MLP(nn.Module):
    def __init__(self, outdim, hidden_dim=256, dropout=0.3):
        super(MLP, self).__init__()
        self.in_dim = 2048 * 3
        self.hidden_dim = hidden_dim
        self.out_dim = outdim
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim, bias = False),
            # nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias = False),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, outdim, bias = False)  # Subject to be replaced dependent on task
        # self.last = CosineLinear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        # self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task
        # self.weights_init()

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    def weights_init(self):
        for module in list(self.modules()):
            if hasattr(module, 'weight'):
                nn.init.normal(module.weight, mean = 0.0, std = 1.0)
                # self.orthogonal_init(module)

    def orthogonal_init(self,layer):
        nn.init.orthogonal(layer.weight)
        if hasattr(layer,'bias'):
            nn.init.constant_(layer.bias,0.1)

    def change_out_dim(self, outdim):
        self.last = nn.Linear(self.hidden_dim, outdim, bias=False)
        self.out_dim = outdim

    # def OrthogonalConstraint(self):
    #     for module in list(self.modules()):
    #         if hasattr(module, 'weight'):
    #             if len(module.weight.shape) >= 2:
    #                 if module.out_features != self.out_dim:
    #                     orthogonal(module, "weight", triv='cayley')


class PointNet(nn.Module):
    def __init__(self, outdim):
        super(PointNet, self).__init__()

        # 特征提取层
        self.out_dim = 0
        self.feature_layer = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 共享隐藏层
        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)


    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1


    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x的形状：(batch_size, 3, num_points)
        x = self.feature_layer(x)
        # x的形状：(batch_size, 1024, num_points)
        x, _ = torch.max(x, 2)
        # x的形状：(batch_size, 1024)
        x = self.shared_layer(x)
        # x的形状：(batch_size, 256)
        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)
        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False

class PointNetCurvature(nn.Module):
    def __init__(self, outdim):
        super(PointNetCurvature, self).__init__()

        # 特征提取层
        self.out_dim = 0
        self.feature_layer1 = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.feature_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # 共享隐藏层
        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)

    # def curvature(self, points, k):
    #     curvature_info = torch.zeros_like(points[:, :, 0:1])
        # for i in range(points.shape[0]):
        #     print(f"num{i} start!")
        #     x = points[i]  # 2048 * 特征数
        #     dist = torch.cdist(x, x)
        #     distances, indices = dist.topk(k + 1, largest=False)  # topk函数获得k个最近邻点（包括自身），largest=False表示获取最小的距离
        #     for j in range(x.shape[0]):
        #         tmp = 0
        #         for e in range(k):
        #             node = indices[j][e + 1]
        #             for g in range(k):
        #                 if indices[node][g + 1] != j:
        #                     tmp += math.sqrt(distances[node][g + 1])
        #             for g in range(k):
        #                 if indices[j][g + 1] != node:
        #                     tmp += math.sqrt(distances[j][g + 1] / distances[j][e + 1])
        #         curvature_info[i][j] = (2 - tmp) / k
        # points = torch.cat((points, curvature_info), dim=2)
        # return points

    def curvature(self, points, k):
        # start = time.time()
        n, num_points, num_features = points.shape
        curvature_info = torch.zeros(n, num_points, 1, device=points.device)

        for i in range(n):
            x = points[i]  # 2048 * 特征数
            dist = torch.cdist(x, x)
            distances, _ = dist.topk(k + 1, largest=False)  # topk函数获得k个最近邻点（包括自身），largest=False表示获取最小的距离
            distances = distances[:, 1:]
            curvature_info[i] = 1.0 / (1e-8 + distances.mean(dim=1, keepdim=True))

        points = torch.cat((points, curvature_info), dim=2)
        # end = time.time()
        # print(f"cost_time = {end - start}s")
        return points

    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1


    def forward(self, x):
        x = self.curvature(x, 5)
        x = x.permute(0, 2, 1)
        # x的形状：(batch_size, 4, num_points)
        x = self.feature_layer1(x)
        x = self.feature_layer2(x)
        # x的形状：(batch_size, 1024, num_points)
        x, _ = torch.max(x, 2)
        # x的形状：(batch_size, 1024)
        x = self.shared_layer(x)
        # x的形状：(batch_size, 256)
        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)
        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False



    #     # 全连接层
    #     self.fc_layers = nn.Sequential(
    #         nn.Linear(1024, 512),
    #         nn.BatchNorm1d(512),
    #         nn.ReLU(),
    #         nn.Linear(512, 256),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
    #         nn.Linear(256, outdim),
    #         nn.LogSoftmax(dim=1)
    #     )
    #
    #     #
    #
    # def forward(self, x):
    #     x = x.permute(0, 2, 1)
    #     # x的形状：(batch_size, 3, num_points)
    #     x = self.feature_layer(x)
    #     # x的形状：(batch_size, 1024, num_points)
    #     x, _ = torch.max(x, 2)
    #     # x的形状：(batch_size, 1024)
    #     x = self.fc_layers(x)
    #     # x的形状：(batch_size, 2)
    #     return x
    #
    # def change_out_dim(self, outdim):
    #     in_features = self.fc_layers[-2].in_features
    #     self.fc_layers[-2] = nn.Linear(in_features, outdim)
    #     self.fc_layers[-1] = nn.LogSoftmax(dim=1)
    #     self.out_dim = outdim

class HyperPointNet(nn.Module):
    def __init__(self, outdim):
        super(HyperPointNet, self).__init__()

        self.k = 20


        # 特征提取层
        self.out_dim = 0
        self.feature_layer1 = Conv2dMaxPool(6, 64, kernel_size=1, bias=False)
        # self.feature_layer1 = nn.Sequential(
        #     nn.Conv2d(6, 64, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        self.feature_layer2 = Conv2dMaxPool(64 * 2, 192, kernel_size=1, bias=False)
        # self.feature_layer2 = nn.Sequential(
        #     nn.Conv2d(64 * 2, 192, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(),
        # )
        self.feature_layer3 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )


        # 共享隐藏层
        self.shared_layer = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # 输出层列表
        self.output_layer = nn.ModuleList()
        self.add_output_layer(outdim)


    def add_output_layer(self, add_dim):
        for _ in range(add_dim):
            new_layer = nn.Linear(256, 1)
            self.output_layer.append(new_layer)
            self.out_dim += 1

    # def add_output_layer(self, add_dim):
    #     for _ in range(add_dim):
    #         new_layer = nn.Sequential(
    #             nn.Linear(512, 256),
    #             nn.BatchNorm1d(256),
    #             nn.ReLU(),
    #             nn.Linear(256, 1)
    #         )
    #         self.output_layer.append(new_layer)
    #         self.out_dim += 1


    def forward(self, x):
        x = x.permute(0, 2, 1)  # x的形状：(batch_size, 3, num_points)

        x = get_hyper_graph_feature(x, k=self.k)  # x的形状：(batch_size, 3 * 2, num_points, self.k)
        x1 = self.feature_layer1(x)  # x1的形状：(batch_size, 64, num_points)

        x = get_hyper_graph_feature(x1, k=self.k)  # x的形状：(batch_size, 64 * 2, num_points, self.k)
        x2 = self.feature_layer2(x)  # x2的形状：(batch_size, 192, num_points)

        x = torch.cat((x1, x2), dim=1)  # x的形状：(batch_size, 256, num_points)

        x = self.feature_layer3(x)  # x的形状：(batch_size, 1024, num_points)

        x, _ = torch.max(x, 2)

        # x的形状：(batch_size, 1024)
        x = self.shared_layer(x)
        # x的形状：(batch_size, 256)
        # x = F.normalize(x, p=2, dim=1)
        outputs = []
        for layer in self.output_layer:
            output = layer(x)
            outputs.append(output)
        concatenated_outputs = torch.cat(outputs, dim=1)
        res = F.log_softmax(concatenated_outputs, dim=1)
        return res

    def freeze_layer(self, lays):
        for layer in lays:
            for param in layer.parameters():
                param.requires_grad = False


class Conv2dMaxPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(Conv2dMaxPool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, _ = x.max(dim=-1, keepdim=False)
        return x

# class HyperPointNet(nn.Module):
#     def __init__(self, outdim):
#         super(HyperPointNet, self).__init__()
#
#         self.k = 20
#
#         # 特征提取层
#         self.out_dim = 0
#         self.feature_layer1 = Conv2dMaxPool(6, 64, kernel_size=1, bias=False)
#         # self.feature_layer1 = nn.Sequential(
#         #     nn.Conv2d(6, 64, kernel_size=1, bias=False),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         # )
#         self.feature_layer2 = Conv2dMaxPool(64 * 2, 64, kernel_size=1, bias=False)
#         # self.feature_layer2 = nn.Sequential(
#         #     nn.Conv2d(64 * 2, 192, kernel_size=1, bias=False),
#         #     nn.BatchNorm2d(192),
#         #     nn.ReLU(),
#         # )
#         self.feature_layer3 = Conv2dMaxPool(64 * 2, 128, kernel_size=1, bias=False)
#         self.feature_layer4 = Conv2dMaxPool(128 * 2, 256, kernel_size=1, bias=False)
#         self.feature_layer5 = nn.Sequential(
#             nn.Conv1d(512, 1024, kernel_size=1, bias=False),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(negative_slope=0.1),
#         )
#
#
#         # 共享隐藏层
#         self.shared_layer = nn.Sequential(
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#         )
#
#         # 输出层列表
#         self.output_layer = nn.ModuleList()
#         self.add_output_layer(outdim)
#
#
#     def add_output_layer(self, add_dim):
#         for _ in range(add_dim):
#             new_layer = nn.Linear(256, 1)
#             self.output_layer.append(new_layer)
#             self.out_dim += 1
#
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # x的形状：(batch_size, 3, num_points)
#
#         x = get_hyper_graph_feature(x, k=self.k)  # x的形状：(batch_size, 3 * 2, num_points, self.k)
#         x1 = self.feature_layer1(x)  # x1的形状：(batch_size, 64, num_points)
#
#         x = get_hyper_graph_feature(x1, k=self.k)  # x的形状：(batch_size, 64 * 2, num_points, self.k)
#         x2 = self.feature_layer2(x)  # x2的形状：(batch_size, 64, num_points)
#
#         x = get_hyper_graph_feature(x2, k=self.k)  # x的形状：(batch_size, 64 * 2, num_points, self.k)
#         x3 = self.feature_layer3(x)  # x3的形状：(batch_size, 128, num_points)
#
#         x = get_hyper_graph_feature(x3, k=self.k)  # x的形状：(batch_size, 128 * 2, num_points, self.k)
#         x4 = self.feature_layer4(x)  # x4的形状：(batch_size, 256, num_points)
#
#         x = torch.cat((x1, x2, x3, x4), dim=1)  # x的形状：(batch_size, 512, num_points)
#
#         x = self.feature_layer5(x)  # x的形状：(batch_size, 1024, num_points)
#
#         x, _ = torch.max(x, 2)
#
#         # x的形状：(batch_size, 1024)
#         x = self.shared_layer(x)
#         # x的形状：(batch_size, 256)
#         outputs = []
#         for layer in self.output_layer:
#             output = layer(x)
#             outputs.append(output)
#         concatenated_outputs = torch.cat(outputs, dim=1)
#         res = F.log_softmax(concatenated_outputs, dim=1)
#         return res
#
#     def freeze_layer(self, lays):
#         for layer in lays:
#             for param in layer.parameters():
#                 param.requires_grad = False

def knn(x, k):
    """
    M = Oblique()
    x = x.transpose(1,2)  # (55,1024,3)
    xM = M.proj(x)
    dis = M.dist(xM,xM).neg()
    """
    """
    x = x.transpose(1, 2)  # (55,1024,3)

    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    m0 = e2p(x[0])
    pairwise_distance = dist_matrix(m0, m0, c=e2p.c)  # 1024*1024
    pairwise_distance = pairwise_distance.unsqueeze(0)

    for i in range(1,x.shape[0]):
        m = e2p(x[i]) # 3*1024
        distance_p = dist_matrix(m, m, c=e2p.c) # 1024*1024
        distance_p = distance_p.unsqueeze(0)
        pairwise_distance = torch.cat((pairwise_distance,distance_p),0)
    """
    """
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (55, 1024, 1024)
    """
    """
    x = x.transpose(1, 2) # (55,1024,3)
    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    x = e2p(x) # 55*1024*3
    pairwise_distance = dist_matrix_knn(x,x,c=e2p.c)

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k),topk返回原始数据和索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx
    """

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (55, 1024, 1024)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def knn_hyper(x, k):
    x = x.transpose(1, 2)  # (55,1024,3)
    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    x = e2p(x)  # 55*1024*3
    pairwise_distance = dist_matrix_knn(x, x, c=e2p.c)

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k),topk返回原始数据和索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_hyper_graph_feature(x, k=20, idx=None):
    e2p = ToPoincare(c=0.01, train_c=False, train_x=False)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = e2p(x)  # 55*1024*3

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((_mobius_add(feature, -x, c=e2p.c), x), dim=3).permute(0, 3, 1,
                                                                               2).contiguous()  # batch, num_dims, num_points, k_neighbors

    return feature


class StableMLP(nn.Module):
    # https://github.com/imirzadeh/stable-continual-learning/blob/master/stable_sgd/models.py
    # https://proceedings.neurips.cc/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Supplemental.pdf
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        super(StableMLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1, self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class ToyMLP(nn.Module) :
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        self.in_dim = in_channel * img_sz * img_sz
        self.linear = nn.Sequential()
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def MLP50():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=50)


def MLP100():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=100)

def MLP256(outdim):
    print("\n Using MLP256 \n")
    return MLP(outdim=outdim,hidden_dim=256)

def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    print("\n Using MLP1000 \n")
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)

if __name__ == '__main__':
    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    
    model = MLP100()
    n_params = count_parameter(model)
    print(f"LeNetC has {n_params} parameters")
