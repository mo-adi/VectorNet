import argparse
import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils import *

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser("Forecasting TGN")

# === Data Related Parameters ===
parser.add_argument('--ex_file_path', type=str, help="Path of ex files")
parser.add_argument('--val_ex_file_path', type=str,
                    help="Path of validation ex files")

# === Common Hyperparameters ===
parser.add_argument('--feature_dim', type=int,
                    default=64, help="Dimenson of features")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size as scene")
parser.add_argument('--epoch', type=int, default=5,
                    help="Number of epochs")
parser.add_argument('--learning_rate', type=float,
                    default=0.002, help="Learning rate")
parser.add_argument('--weight_decay', type=float,
                    default=0.0, help="Weight decay")
parser.add_argument('--lr_decay_schedule', type=int,
                    default=0, choices=[0, 1], help="LR rate decay schedule")


# === Model Saving/Loading Parameters ===
parser.add_argument('--model_save_path', type=str, default="./",
                    help="Path to save per epoch model")
parser.add_argument('--pretrain_path', type=str, default=None,
                    help='path of pretrained/checkpoint model')
parser.add_argument('--load_epoch', type=int, default=None,
                    help='epoch of training to be loaded')
parser.add_argument('--training_name', type=str, default=None,
                    help='Name of the training to be saved as')

# === Misc Training Parameters ===
parser.add_argument('--world_size', type=int, default=1)
parser.add_argument('--validation_epoch', type=int,
                    default=1, help="Validation per n epoch")
parser.add_argument('--num_workers', type=int, default=0)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


class Arguments:
    def __init__(self):
        self.ex_file_path = args.ex_file_path
        self.val_ex_file_path = args.val_ex_file_path

        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.lr_decay_schedule = args.lr_decay_schedule

        self.save_path = args.model_save_path
        self.pretrain_path = args.pretrain_path
        self.load_epoch = args.load_epoch
        self.training_name = args.training_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.world_size = args.world_size
        self.validation_epoch = args.validation_epoch
        self.num_workers = args.num_workers

        if self.load_epoch is not None:
            assert self.pretrain_path is not None


def create_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    return logger


def train(model, iter_bar, optimizer, rank, dataset, main_device, args, logger, i_epoch):
    total_loss = 0.0
    for step, batch in enumerate(iter_bar):

        loss = model(batch)
        total_loss += loss.item()
        loss.backward()
        if main_device:
            loss_desc = f"loss = {total_loss/((step+1)*args.batch_size):.5f}"
            iter_bar.set_description(loss_desc)
        optimizer.step()
        optimizer.zero_grad()


def validate(model, dataloader, dataset, logger):
    file2pred = {}
    file2labels = {}
    DEs = []
    iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')

    with torch.no_grad():
        for step, batch in enumerate(iter_bar):

            pred_trajectory, pred_score = model(batch, True)
            batch_size = pred_trajectory.shape[0]
            for i in range(batch_size):
                assert pred_trajectory[i].shape == (6, 30, 2)
                assert pred_score[i].shape == (6,)

            # batch = [scene[0] for scene in batch]
            eval_instance_argoverse(
                batch_size, pred_trajectory, batch, file2pred, file2labels, DEs, iter_bar, step == 0)

    post_eval(file2pred, file2labels, DEs, logger)


def main(rank, args):
    main_device = True if rank == 0 else False
    args.device = rank

    logger = create_logger() if main_device else None

    world_size = args.world_size
    setup(rank, world_size)

    # set train dataset and dataloader
    train_dataset = Argoverse_Dataset(args.ex_file_path)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size//world_size,
                                  pin_memory=False, drop_last=False, shuffle=False, sampler=train_sampler,
                                  collate_fn=batch_list_to_batch_tensors, num_workers=args.num_workers)

    # if main device, load validation dataset
    if main_device:
        val_dataset = Argoverse_Dataset(args.val_ex_file_path, validation=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size//world_size,
                                    pin_memory=False, drop_last=False, shuffle=False, collate_fn=batch_list_to_batch_tensors,
                                    num_workers=args.num_workers)

    model, optimizer = get_model(args)

    for i_epoch in range(args.epoch_num):
        if main_device:
            logger.info(f"=== Epoch {i_epoch}/{args.epoch_num} ===")
            logger.info(
                f"learning_rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
        train_sampler.set_epoch(i_epoch)

        if main_device:
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else:
            iter_bar = train_dataloader

        model.train()
        train(model, iter_bar, optimizer, rank,
              train_dataset, main_device, args, logger, i_epoch)

        if main_device and i_epoch % args.validation_epoch == 0:
            checkpoint = {
                "epoch": i_epoch,
                "model_dict": model.state_dict(),
                "optimizer_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, "checkpoint")
            model.eval()
            validate(model, val_dataloader, val_dataset, logger)

        # torch.cuda.empty_cache()
        if main_device:
            logger.info("=== === === \n")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    args = Arguments()
    world_size = args.world_size
    torch.set_num_threads(args.world_size*6)
    mp.spawn(main, args=[args], nprocs=world_size)