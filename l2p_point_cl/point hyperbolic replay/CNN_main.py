import os
SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1,0'

import argparse
import torch
from random import shuffle
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import methods
import wandb
import pytorch_lightning as pl

import numpy as np
import os
from tqdm import tqdm
from typing import List
import time, csv, copy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import skdim
from torch.nn.parallel import DataParallel
from pytorch_lightning.callbacks import ModelCheckpoint

# torch.distributed.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# import torch.distributed as dist
# def init_distributed_mode(args):
#     # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
#     # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
#     if'RANK'in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         # LOCAL_RANK代表某个机器上第几块GPU
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif'SLURM_PROCID'in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return
#
#     args.distributed = True
#
#     torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的GPU
#     args.dist_backend = 'nccl'# 通信后端，nvidia GPU推荐使用NCCL
#     dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续

result_root_path = r'./result'
def prepare_dataloaders(args):
    # Prepare dataloaders_Adam
    Dataset = dataloaders.base.__dict__[args.dataset]

    # SPLIT CUB
    if args.is_split_cub :
        print("running split -------------")
        from dataloaders.cub import CUB
        Dataset = CUB
        if args.train_aug :
            print("train aug not supported for cub")
            return
        train_dataset, val_dataset = Dataset(args.dataroot)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class = True)
        n_tasks = len(task_output_space.items())
    # Permuted Permuted_MNIST
    elif args.n_permutation > 0:
        # TODO : CHECK subset_size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug, angle=0, subset_size=args.subset_size)
        print("Working with permuatations :) ")
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class = False)
        n_tasks = args.n_permutation
    # Rotated Permuted_MNIST
    elif args.n_rotate > 0 or len(args.rotations) > 0:
        # TODO : Check subset size
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                 dataroot=args.dataroot,
                                                                                 train_aug=args.train_aug,
                                                                                 n_rotate=args.n_rotate,
                                                                                 rotate_step=args.rotate_step,
                                                                                 remap_class=not args.no_class_remap,
                                                                                 rotations=args.rotations,
                                                                                 subset_size=args.subset_size)
        n_tasks = len(task_output_space.items())

    # Split Permuted_MNIST
    else:
        print("running split -------------")
        # TODO : Check subset size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug,
                                             angle=0, subset_size=args.subset_size)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())

    print(f"task_output_space {task_output_space}")
    # print('****',train_dataset_splits['1'][0])
    return task_output_space, n_tasks, train_dataset_splits, val_dataset_splits


def run(args, task_output_space, n_tasks, train_dataset_splits, val_dataset_splits):
    # Prepare the Agent (model)
    agent_config = args
    agent_config.out_dim = {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space
    model = methods.__dict__[agent_config.method_name].__dict__[agent_config.method_name]
    agent = model(agent_config)

    # distribute training
    # init_distributed_mode(args=args)
    # torch.distributed.init_process_group(backend='nccl')

    # gpus = [0,1]
    # torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    # agent.model = DataParallel(agent.model.cuda(), device_ids=gpus, output_device=gpus[1])
    # agent.model = DataParallel(agent.model)
    # print(agent.model.device)
    # agent.model = agent.model.cuda()
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    # Continual learning
    # Compute the validation scores
    acc, param_id = list(), list()
    for i in tqdm(range(len(task_names)), "task"):

        task_name = task_names[i]

        # distribute
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_splits[task_name])
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_splits[task_name])

        print(f'====================== Task {task_name} =======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                 batch_size = args.batch_size, shuffle=False,
                                                 num_workers=args.workers)

        # train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
        #                                            batch_sampler=train_sampler,
        #                                            pin_memory=True,
        #                                            batch_size=args.batch_size, shuffle=True,
        #                                            num_workers=args.workers)
        # val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
        #                                          pin_memory=True,
        #                                          sampler=val_sampler,
        #                                          batch_size = args.batch_size, shuffle=False,
        #                                          num_workers=args.workers)

        if i == 0:
            Acc = agent.train(train_loader, val_loader, task_name)
        else:
            # if ref_model is None:
            ref_model = copy.deepcopy(agent.model)
            for p in ref_model.parameters():
                p.requires_grad = False
            Acc = agent.train(train_loader, val_loader, task_name, ref_model)
        print('task : {}  Acc : {}'.format(task_name, Acc))
        # if hasattr(agent,'reservior_update'):
        #     print('*****')
        #     agent.reservior_update(train_dataset_splits[task_name])
        # TODO : Add this part ASAP
        # if args.incremental_class:
        #     agent.add_valid_output_dim(task_output_space[task_name])
        tot_acc = list()
        for task in task_names:
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[task],
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
            tot_acc.append(agent.validation(val_loader))
        acc.append(np.array(tot_acc))
        print(tot_acc)

    acc = np.array(acc)
    result_path = os.path.join(result_root_path, args.dataset)
    if args.num_task != None:
        result_path = os.path.join(result_path,str(args.num_task) + 'task')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    tot = args.total_classes * args.mem_size_per_class
    # file_name = args.method_name + '_' + str(tot)
    file_name = args.method_name + '_' + str(tot) + '_ER_WD_MES'
    file_name = file_name + '_seed' + str(args.run_seed)
    data_file_name = file_name + '.csv'
    data_result = os.path.join(result_path, data_file_name)
    with open(data_result,'w',newline='') as to:
        write = csv.writer(to)
        write.writerows(acc)

    # Illustrate TSNE
    return agent

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                      help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
    parser.add_argument('--run_seed', type=int, default=9)
    parser.add_argument('--local_rank', type=int, default=0, required = False)
    parser.add_argument('--num_task', type=int, default=20)
    parser.add_argument('--force_out_dim', type=int, default=0,
                      help="Set 0 to let the task decide the required output dimension", required=False)
    parser.add_argument('--optimizer', type=str, default='SGD',
                      help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...", required=False)
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="MNIST(default)|CIFAR10|CIFAR100|MiniImageNet|TinyImageNet", required=False)
    parser.add_argument('--first_split_size', type=int, default=5, required=False)
    parser.add_argument('--other_split_size', type=int, default=5, required=False)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                      help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,"
                           "6 ...] -> [0,1,2 ...]", required=False)
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                      help="Allow data augmentation during training", required=False)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                      help="Randomize the classes in splits", required=False)

    parser.add_argument('--single_epoch', dest='single_epoch', default=False, action='store_true',
                        help="Randomize the classes in splits", required=False)

    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                      help="Randomize the order of splits", required=False)
    parser.add_argument('--schedule', nargs="+", type=int, default=[5],
                      help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number "
                           "is the end epoch", required=False)
    parser.add_argument('--model_weights', type=str, default=None,
                      help="The path to the file for the model weights (*.pth).", required=False)
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                      help="Force the evaluation on train set", required=False)
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                      help="Non-incremental learning by make all data available in one batch. For measuring "
                           "the upperbound performance.", required=False)
    # parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
    #                   help="The number of output node in the single-headed model increases along with new "
    #                        "categories.", required=False)
    parser.add_argument('--method_name', type=str, default='oblique', required=False, help = 'oblique|subspace_proj|ER_oblique|ER_Reservoir')
    args = parser.parse_args()

    args.gpu = True
    args.group_id = 0
    args.workers = 0
    args.start_seed = 0
    args.end_seed = 0
    # args.run_seed = 9
    args.val_size = 256
    args.scheduler = False
    args.nepoch = 5
    args.val_check_interval = 300
    args.batch_size = 10
    args.train_percent_check = 1
    args.ogd_start_layer = 0
    args.ogd_end_layer = 1e6
    args.memory_size = 100
    args.hidden_dim = 256
    args.pca = False
    args.agem_mem_batch_size = 100
    args.no_transfer = False
    args.n_rotate = 0
    args.rotate_step = 0
    # args.is_split = True
    args.is_split = True
    args.data_seed = 256
    args.rotations = []
    args.toy = False
    args.ogd = False
    args.ogd_plus = False
    args.no_random_name = False
    args.project = 'iclr-2021-cl-prod'
    args.wandb_dryrun = False
    args.wandb_dir = '/'
    args.dataroot = './datasets'
    # args.dataroot = r'D:\XZH\华东师范大学\研一上\几何深度学习\持续学习\论文项目代码\continual-learning-oblique_manifold\datasets'
    args.is_split_cub = False
    args.reg_coef = 0
    args.method_name = 'resnet_Manifold_Matching'
    # args.method_name = 'AlexNet_MM'

    # args.method_name = 'ER_Reservoir'
    # args.method_type = 'ER_Reservoir'
    #
    # args.method_name = 'ER_oblique'
    # args.method_type = 'ER_oblique'

    # args.method_type = 'subspace_proj'
    # args.method_name = 'subspace_proj'

    # args.method_type = 'sgd_base'
    # args.method_name = 'sgd_base'
    args.model_type = 'resnet'
    args.model_name = 'ResNet18'
    args.n_permutation = 0
    args.subset_size = None

    args.gamma = 1
    args.is_stable_sgd = False
    args.momentum = 0
    args.weight_decay = 0
    args.print_freq = 100
    args.no_val = False
    args.total_classes = 100
    args.mem_size_per_class = 5
    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)
    args.eps_mem_batch = 32

    task_output_space, n_tasks, train_dataset_splits, val_dataset_splits = prepare_dataloaders(args)
    agent = run(args,task_output_space, n_tasks, train_dataset_splits, val_dataset_splits)
