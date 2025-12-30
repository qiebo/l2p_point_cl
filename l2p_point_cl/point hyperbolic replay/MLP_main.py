import copy
import os
SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')

import argparse
import torch
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import methods


import numpy as np
import os
from tqdm import tqdm
import time, csv

from pytorch_lightning.callbacks import ModelCheckpoint
result_root_path = r'./result'

def prepare_dataloaders(args):
    # Prepare dataloaders_Adam
    Dataset = dataloaders.base.__dict__[args.dataset]  # 相当于把MNIST函数复制为Dataset

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
    elif args.n_rotate > 0 or len(args.rotations) > 0 :
        # TODO : Check subset size
        print('Rotated ')
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                 dataroot=args.dataroot,
                                                                                 train_aug=args.train_aug,
                                                                                 n_rotate=args.n_rotate,
                                                                                 rotate_step=args.rotate_step,
                                                                                 remap_class= False,
                                                                                 # remap_class=not args.no_class_remap,
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
    agent_config.out_dim = {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space  # 字典，关键词为任务序号，值为输出维度

    model = methods.__dict__[agent_config.method_name].__dict__[agent_config.method_name]  # agent_config.method_name == "MLP_correlation"
    agent = model(agent_config)  # 以当前参数创建模型
    print(args.method_name)
    task_names = sorted(list(task_output_space.keys()), key=int)

     # Continual learning
    # Compute the validation scores
    acc, param_id = list(), list()
    ref_model = None
    for i in tqdm(range(len(task_names)), "task"):
        dir_path = 'checkpoint/task_' + str(i) + '/'
        checkpoint_callback = ModelCheckpoint(dirpath=dir_path, monitor='val_acc', mode='max', save_top_k=1)  # 以指定频率保存 Keras 模型或权重
        if not os.path.exists(dir_path):  # 如果没有当前文件夹，则创建一个
            os.makedirs(dir_path)
        # trainer = pl.Trainer(gpus=1, min_epochs=5, precision=16, callbacks=[
        #     EarlyStopping(monitor='val_acc', mode='max', min_delta=0.0, stopping_threshold=1.0),
        #     checkpoint_callback])
        # trainer = pl.Trainer(gpus=1, min_epochs=10, max_epochs=15, precision=16, callbacks=[
        #     EarlyStopping(monitor='val_acc', mode='max', min_delta=0.0, stopping_threshold=1.0),
        #     checkpoint_callback])
        # trainer = pl.Trainer(gpus=1, max_epochs=10)
        task_name = task_names[i]
        print(f'====================== Task {task_name} =======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
        # Train the agent
        # trainer.fit(agent, train_loader, val_loader)
        if i == 0:
            Acc = agent.train(train_loader, val_loader,task_name)  # 返回当前任务训练五次epoch后的精确度
        else:
            # if ref_model is None:
            ref_model = copy.deepcopy(agent.model)  # 将当前model copy到ref_model
            for p in ref_model.parameters():  # 冻结ref_model的梯度更新
                p.requires_grad = False
            # ref_model = agent.model
            # agent.model = agent.create_model()
            # param_dict = ablation_study_copy.deepcopy(dict(agent.named_parameters()))
            Acc = agent.train(train_loader, val_loader, task_name, ref_model)
            # return
        print('task : {}  Acc : {}'.format(task_name, Acc))

        tot_acc = list()
        for task in task_names:
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[task],
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
            tot_acc.append(agent.validation(val_loader))
        print(tot_acc)
        acc.append(np.array(tot_acc))
        print(acc)


    val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_names[1]],
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers)
    # Illustrate TSNE
    from utils import TSNE
    X, Y = None, None
    for x, y, task in val_loader:
        if X is None:
            X = x
        else:
            X = torch.cat((X, x), dim=0)
        if Y is None:
            Y = y
        else:
            Y = torch.cat((Y, y), dim=0)
    tot = args.total_classes * args.mem_size_per_class

    TSNE.Draw(agent.model.features(X.cuda()).cpu().detach().numpy(), Y.detach().numpy(), f'MaER_{tot}')

    acc = np.array(acc)
    result_path = os.path.join(result_root_path, args.dataset)
    if args.num_task != None:
        result_path = os.path.join(result_path,str(args.num_task) + 'task')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # tot = args.total_classes * args.mem_size_per_class
    file_name = args.method_name + '_' + str(tot)
    # file_name = args.method_name + '_' + str(tot) + '_ER_WD'
    file_name = file_name + '_seed' + str(args.run_seed)
    data_file_name = file_name + '.csv'
    data_result = os.path.join(result_path, data_file_name)

    with open(data_result,'w',newline='') as to:
        write = csv.writer(to)
        write.writerows(acc)

    print(f'Diameter: ', agent.diameter)

    return agent

def set_argument(args):
    args.ground_metric_normalize = 'none'
    args.ground_metric = 'cosine'
    args.exact = True
    args.ensemble_step = 0
    args.importance = None
    args.correction = True
    args.proper_correction = False
    args.past_correction = True
    args.skip_last_layer = False
    args.reg = 1e-2
    args.debug = False
    args.not_squared = True
    args.normalize_wts = True
    args.dist_normalize = True
    args.ground_metric_eff = False
    args.proper_marginals = True
    args.importance = 'l2'
    args.unbalanced = False
    # args.skip_last_layer = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  # 创建argparse对象

    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                      help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)  # 调用add_argument方法添加参数，required代表是否可以省略
    parser.add_argument('--run_seed', type=int, default=9)
    parser.add_argument('--num_task', type=int, default=20)
    parser.add_argument('--force_out_dim', type=int, default=0,
                      help="Set 0 to let the task decide the required output dimension", required=False)
    parser.add_argument('--optimizer', type=str, default='SGD',
                      help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...", required=False)
    parser.add_argument('--dataset', type=str, default='MNIST', help="Permuted_MNIST(default)|CIFAR10|CIFAR100_3|MiniImageNet", required=False)
    parser.add_argument('--first_split_size', type=int, default=5, required=False)
    parser.add_argument('--other_split_size', type=int, default=5, required=False)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                      help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,"
                           "6 ...] -> [0,1,2 ...]", required=False)
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                      help="Allow data augmentation during training", required=False)
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
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
    parser.add_argument('--method_name', type=str, default='MLP', required=False, help = 'oblique|subspace_proj|ER_oblique|ER_Reservoir')
    parser.add_argument('--multiple', type = int, default = 1, required= False)
    args = parser.parse_args()

    set_argument(args)

    args.gpu = True
    args.group_id = 0
    args.workers = 0
    args.start_seed = 0
    args.end_seed = 0
    args.val_size = 256
    args.scheduler = False
    args.nepoch = 5
    args.val_check_interval = 300
    args.batch_size = 16
    args.train_percent_check = 1
    args.hidden_dim = 256
    args.n_rotate = 20
    args.rotate_step = 180 / 20
    # args.is_split = True
    args.is_split = True
    args.data_seed = 256
    # args.rotations = []
    args.rotations = np.random.rand(20) * 180
    args.toy = False
    args.ogd = False
    args.ogd_plus = False
    args.no_random_name = False
    args.project = 'iclr-2021-cl-prod'
    args.wandb_dryrun = False
    args.wandb_dir = '/'
    args.dataroot = './datasets'
    args.is_split_cub = False
    args.reg_coef = 0
    # args.method_name = 'oblique'

    # args.method_name = 'ER_Reservoir'
    # args.method_type = 'ER_Reservoir'
    #
    args.method_name = 'MLP_correlation'
    # args.method_type = 'ER_oblique'

    args.model_type = 'mlp'
    args.model_name = 'MLP'
    args.n_permutation = 20
    # args.n_permutation = 0
    # args.subset_size = None
    args.subset_size = 0.25

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
    # args.eps_mem_batch = 256
    args.eps_mem_batch = 16

    task_output_space, n_tasks, train_dataset_splits, val_dataset_splits = prepare_dataloaders(args)
    agent = run(args, task_output_space, n_tasks, train_dataset_splits, val_dataset_splits)


