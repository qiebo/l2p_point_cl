import torch
import numpy as np
import ot
from hyptorch.nn import ToPoincare
from hyptorch.pmath import poincare_mean, dist_matrix, dist0, dist, dist_matrix_knn, _mobius_add


def cost_matrix(x, y, p=2):
    # 定义成本矩阵的计算函数
    x_col = x.unsqueeze(1)  # 将 x 扩展成 (1024, 1, 3) 的形状，为了计算成对距离
    y_lin = y.unsqueeze(0)  # 将 y 扩展成 (1, 1024, 3) 的形状，同上
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)  # 计算 x 和 y 之间的成对距离的 p 次方，然后对最后一个维度求和
    return c


def compute_wasserstein_distance(cloud1, cloud2, p=2):
    # 计算两个点云之间的 Wasserstein 距离
    assert cloud1.shape == cloud2.shape, "两个点云必须有相同的形状。"

    # 为每个维度计算成本矩阵
    cost_matrices = [cost_matrix(cloud1[:, i].unsqueeze(1), cloud2[:, i].unsqueeze(1), p) for i in
                     range(cloud1.shape[1])]

    # 对每个维度计算 Wasserstein 距离
    distances = [ot.emd2([], [], cost_matrix.numpy()) for cost_matrix in cost_matrices]

    # 返回平均 Wasserstein 距离
    return np.mean(distances)


# 示例用法
cloud1 = torch.rand(1024, 3)  # 生成一个随机的点云
cloud2 = torch.rand(1024, 3)  # 生成另一个随机的点云
distance = compute_wasserstein_distance(cloud1, cloud2)  # 计算这两个点云之间的 Wasserstein 距离
print("Wasserstein Distance: ", distance)  # 打印 Wasserstein 距离
print(cloud1[:, 1].shape)


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


x = torch.rand(64, 3, 1024)
x = get_hyper_graph_feature(x)
print(x.shape)
x = x.max(dim=-1, keepdim=False)[0]
print(x.shape)