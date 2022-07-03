import math
import os
import pickle
import random
import zlib

import numpy as np
import torch
import torch.distributed as dist
from argoverse.evaluation import eval_forecasting
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset


class Argoverse_Dataset(Dataset):
    def __init__(self, ex_file_path, validation=False):
        pickle_file = open(ex_file_path, 'rb')
        self.ex_list = pickle.load(pickle_file)
        pickle_file.close()

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance


def merge_tensors(tensors, device, hidden_size):
    '''
    Args:
        tensors (list of lists of torch.Tensor): List, where each item
                corresponds to a polyline tensor
                len(tensors) = #polylines
                tensors[i].shape = (#vectors in polyline j, hidden_size)
                                    Typically, #vectors == 19 if j is agent,
                                               #vectors in [2, .., 10] if j is lane
        device (int or string): "cuda", "cpu" or index of the gpu if multi gpu used
        hidden_size (int): length of vectors/polylines
    Returns:
        res (torch.Tensor): Full tensor of polylines
                            res.shape = (#polylines, max(len(polylines)), hidden_size)
        lengths (list of int): A list of integers where each items correponds to
                               number of vectors in that polyline
    '''
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


origin_point = None
origin_angle = None
method2FDEs = []


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11227'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def learning_rate_decay(i_epoch, optimizer, args):
    if args.lr_decay_schedule == 0:
        first_decrease = int(args.epoch_num*0.7)
        second_decrease = int(args.epoch_num*0.9)
        if i_epoch in [first_decrease, second_decrease]:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1

    elif args.lr_decay_schedule == 1:
        if i_epoch % 5 == 0 and i_epoch != 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5


def batch_init(mapping):
    global origin_point, origin_angle
    batch_size = len(mapping)

    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']


def to_origin_coordinate(points, idx_in_batch, scale=None):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
        if scale is not None:
            point[0] *= scale
            point[1] *= scale


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = random.randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


def eval_instance_argoverse(batch_size, pred, mapping, file2pred, file2labels, DEs, iter_bar, first_time):
    global method2FDEs
    if first_time:
        method2FDEs = []

    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, 30, 2)
        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2labels[file_name_int] = mapping[i]['origin_labels']

    DE = np.zeros([batch_size, 30])
    for i in range(batch_size):
        origin_labels = mapping[i]['origin_labels']
        FDE = np.min(get_dis_point_2_points(
                origin_labels[-1], pred[i, :, -1, :]))
        method2FDEs.append(FDE)
        for j in range(30):
            DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                    origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
    DEs.append(DE)
    miss_rate = 0.0
    miss_rate = np.sum(np.array(method2FDEs) > 2.0) / len(method2FDEs)

    iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def post_eval(file2pred, file2labels, DEs, logger):

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, 30, 2.0)
    logger.info(metric_results)

    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        logger.info(
            f" {'ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3'}: {score}")


def get_model(args):
    from Vectornet import VectorNet
    rank = args.device

    # loading/creating model
    model = VectorNet(args.feature_dim, args.device)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # loading/creating optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return model, optimizer