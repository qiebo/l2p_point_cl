MLP permuted:
    epoch: 5
    learning rate: 0.01
    batch size: 16
    memory batch size: 16
    memory size: 100
    result:

MLP permuted:
    epoch: 10
    learning rate: 0.01
    batch size: 16
    memory batch size: 16
    memory size: 200
    result: 89.748%, 89.668%
    
MLP permuted:
    epoch: 10
    learning rate: 0.01
    batch size: 16
    memory batch size: 16
    memory size: 500
    result: 93.04% 91.956%

MLP rotated:
    epoch: 10
    learning rate: 0.01
    batch size: 16
    memory batch size: 16
    memory size: 200
    result: 

MLP rotated:
    epoch: 10
    learning rate: 0.01
    batch size: 16
    memory batch size: 16
    memory size: 500
    result: 92.774%  92.31%

MiniImageNet:
    epoch: 10
    learning rate: 0.005
    batch size: 10
    memory batch size: 32
    memory size: 500
    result:

MiniImageNet:
    epoch: 1
    learning rate: 0.003
    batch size: 10
    memory batch size: 32
    memory size: 200
    result: 50.45%

MiniImageNet:
    epoch: 10
    learning rate: 0.003
    batch size: 10
    memory batch size: 32
    memory size: 200
    result: 59.18%

TinyImageNet:
    epoch: 20
    learning rate: 0.001
    batch size: 10
    memory batch size: 32
    memory size: 200
    result: 45.34%

TinyImageNet:
    epoch: 10
    learning rate: 0.003
    batch size: 10
    memory batch size: 32
    memory size: 200
    result: 46.01%

CIFAR10:
    epoch 10
    learning rate: 0.003
    batch size: 16
    memory batch size: 32
    memory size: 200
    result:
CIFAR10:
    epoch 10
    learning rate: 0.003
    batch size: 16
    memory batch size: 32
    memory size: 200
    result:
CIFAR100:
    epoch: 8
    learning rate: 0.003
    batch size: 10
    memory batch size: 10
    memory size: 500
    result: 63.95% 65.22%
CIFAR100:
    epoch: 5 （1）
    learning rate: 0.01
    batch size: 10
    memory batch size: 32
    memory size: 500
    result: 67.86%
CIFAR100:
    epoch: 5 （1）
    learning rate: 0.01
    batch size: 10
    memory batch size: 32
    memory size: 300
    result: 62.04%
CIFAR100:
    epoch: 5 （1）
    learning rate: 0.01
    batch size: 10
    memory batch size: 32
    memory size: 500
    result: 67.86%
CIFAR100:
    epoch: 6
    learning rate: 0.005
    batch size: 10
    memory batch size: 64
    memory size: 500
    result: 66.93
CIFAR100:
    epoch: 2
    learning rate: 0.003
    batch size: 10
    memory batch size: 64
    memory size: 500
    result: 66.78%
CIFAR100:
    epoch: 5
    learning rate: 0.003
    batch size: 10
    memory batch size: 64
    memory size: 100
    result: 57.99%



------------------------------------------------------------------------ MiniImageNet
python conv_cifar_resnet.py --method_name ER_oblique_resnet --dataset MiniImageNet --rand_split 
python conv_cifar_resnet.py --method_name oblique_resnet --dataset MiniImageNet --rand_split
python conv_MNIST.py --method_name oblique_CNN
python conv_cifar_resnet.py --method_name oblique_resnet2 --rand_split --dataset MiniImageNet

tensorboard --logdir lightning_logs

python conv_MNIST.py --method_name oblique_CNN

---
python CIL_MLP_MNIST.py --method_name oblique_MLP_proj
python CIL_conv_cifar_resnet.py --method_name oblique_resnet --rand_split

python MLP_main.py --method_name MLP_correlation

MaER: [1.1681,1.1531,1.1131,1.1744,1.2032,1.2357,1.1904,1.2106,1.2276,1.2492,1.2535,1.2380,1.2538,1.2453,1.2467,1.2561,1.2341,1.2401]