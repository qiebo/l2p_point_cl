import os
import pandas as pd
import numpy as np
data_root = r'D:\XZH\华东师范大学\研一上\几何深度学习\持续学习\论文项目代码\Replay_continual_learning\result\CIFAR100\20task\resnet_Manifold_Matching_100_seed9_4.csv'
data_root = r'D:\XZH\华东师范大学\研一上\几何深度学习\持续学习\论文项目代码\Replay_continual_learning\result\CIFAR100\20task\CNN_Manifold_Matching_ResNet18_seed9_4.csv'
data = pd.read_csv(data_root,header=None)
data = np.array(data)[:20,:20]
print(data.shape)
print((np.max(data,axis = 0) - data[-1]).sum() / 20)
