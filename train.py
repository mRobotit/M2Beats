import sys
import datetime
import argparse
import os
import time

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")

import utils.train_utils
from utils.train_utils import log, Saver, network_paras
from data_loader import get_dataloaders

from module.m2bnet import m2bnet

def compute_loss(predict, target, loss_mask, loss_func):
    loss = loss_func(predict, target)
    loss = loss * loss_mask
    loss = torch.sum(loss) / torch.sum(loss_mask)
    return loss

def train_dp():
    parser = argparse.ArgumentParser(description="Args for training BF")
    parser.add_argument('-n', '--name', default="debug",
                        help="Name of the experiment, also the log file and checkpoint directory. If 'debug', checkpoints won't be saved")
    parser.add_argument('-l', '--lr', default=0.001, help="Initial learning rate")
    parser.add_argument('-b', '--batch_size', default=4, help="Batch size")
    parser.add_argument('-e', '--epochs', default=400, help="Num of epochs")
    parser.add_argument('-g', '--gpus', type=int, nargs='+', default=[0], help="Ids of gpu")
    parser.add_argument('--train', default='./DATASET_ASSET/train_data.csv', help="cv to list train file")
    parser.add_argument('--test', default='./DATASET_ASSET/test_data.csv', help="cv to list test file")
    parser.add_argument('-t', '--train_data', default='./DATASET_ASSET/train_data', help="Path of the training data dir")
    parser.add_argument('-c', '--checkpoint', help="If set, load model from the given path")
    parser.add_argument('--lw', default=30, type=float, help="loss weight")
    args = parser.parse_args()

    if args.gpus is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in list(range(torch.cuda.device_count()))])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(g) for g in args.gpus])

    loss_weight = args.lw
    path_train_data = args.train_data
    init_lr = float(args.lr)
    batch_size = int(args.batch_size)
    DEBUG = args.name == "debug"
    params = {
        "DECAY_EPOCH": [],
        "DECAY_RATIO": 0.1,
    }

    log("name:", args.name)
    log("args", args)

    if DEBUG:
        log("DEBUG MODE checkpoints will not be saved")
    else:
        os.makedirs("./logs/", exist_ok=True)
        utils.train_utils.flog = open("./logs/" + args.name + ".log", "w")

    # hyper params
    n_epoch = args.epochs
    max_grad_norm = 3

    # config
    train_loader = get_dataloaders(args.train, path_train_data, batch_size)
    num_batch = len(train_loader) // batch_size
    test_loader = get_dataloaders(args.test, path_train_data, batch_size)
    num_batch_test = len(test_loader) // batch_size

    # create saver
    saver_agent = Saver(exp_dir="./exp/" + args.name, debug=DEBUG)

    # init
    net = m2bnet()

    # tensorboard
    writer = SummaryWriter('./tensorboardlog/')
    writer_idx = 0

    if torch.cuda.is_available():
        net.cuda()
    
    if args.checkpoint is not None:
        path_saved_ckpt = args.checkpoint
        print("[*] load model from: {}".format(path_saved_ckpt))
        net.load_state_dict(torch.load(path_saved_ckpt))

    DEVICE_COUNT = torch.cuda.device_count()
    log("DEVICE COUNT:", DEVICE_COUNT)
    log("VISIBLE: " + os.environ["CUDA_VISIBLE_DEVICES"])

    
    n_parameters = network_paras(net)
    log('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(' > params amount: {:,d}'.format(n_parameters))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)
    loss_func = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([loss_weight]))
    # loss_func = torch.nn.MSELoss(reduction='none')
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()

    log('    train_data:', path_train_data.split("/")[-2])
    log('    batch_size:', batch_size)
    log('    num_batch:', num_batch)
    log('    dataset length:', len(train_loader))
    log('    lr_init:', init_lr)
    for k, v in params.items():
        log(f'    {k}: {v}')

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        test_loss = 0

        if epoch in params['DECAY_EPOCH']:
            log('LR decay by ratio', params['DECAY_RATIO'])
            for p in optimizer.param_groups:
                p['lr'] *= params['DECAY_RATIO']

        # train
        for bidx in range(num_batch):  # num_batch
            net.train()
            saver_agent.global_step_increment()

            # read batch data
            # beat downbeat, mask

            batch_keypoints, batch_timestamps, batch_beat, batch_mask = next(iter(train_loader))
            
            #如果有无效帧则跳过
            # if(torch.min(batch_mask[:,1:]) == 0):
            #     continue

            # batch_beat = torch.unsqueeze(batch_beat, -1)
            batch_mask = torch.unsqueeze(batch_mask, -1)

            batch_keypoints = batch_keypoints.float()
            # batch_timestamps = batch_timestamps.float()
            batch_beat =  batch_beat.float()
            batch_mask = batch_mask.long()

            if torch.cuda.is_available():
                batch_keypoints = batch_keypoints.cuda()
                batch_timestamps = batch_timestamps.cuda()
                batch_beat = batch_beat.cuda()
                batch_mask = batch_mask.cuda()

            # run
            out = net(batch_keypoints)
            # out = torch.sigmoid(out)

            loss = compute_loss(out,batch_beat,batch_mask,loss_func)

            # Update
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Loss: {:.3f} | beat {:.3f}\r'.format(
                    bidx, num_batch, float(loss), loss))
            sys.stdout.flush()

            # acc
            acc_loss += loss.item()

            # log
            saver_agent.add_summary('batch loss', loss.item())
            writer.add_scalar("loss", loss.item(), writer_idx)
            writer_idx += 1

        # eval
        for bidx in range(num_batch_test):
            net.eval()
            batch_keypoints, batch_timestamps, batch_beat, batch_mask = next(iter(test_loader))
            
            # batch_beat = torch.unsqueeze(batch_beat, -1)
            batch_mask = torch.unsqueeze(batch_mask, -1)

            batch_keypoints = batch_keypoints.float()
            # batch_timestamps = batch_timestamps.float()
            batch_beat =  batch_beat.float()
            batch_mask = batch_mask.long()

            if torch.cuda.is_available():
                batch_keypoints = batch_keypoints.cuda()
                batch_timestamps = batch_timestamps.cuda()
                batch_beat = batch_beat.cuda()
                batch_mask = batch_mask.cuda()
            # run
            out = net(batch_keypoints)
            loss = compute_loss(out,batch_beat,batch_mask,loss_func)

            # print
            sys.stdout.write('{}/{} | Loss: {:.3f} | beat {:.3f}\r'.format(
                    bidx, num_batch_test, float(loss), loss))
            sys.stdout.flush()

            # acc
            test_loss += loss.item()
        
        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        log('-' * 80)
        log("train_loss")
        log(time.ctime() + ' epoch: {}/{} | Loss: {:.3f} | time: {}'.format(
            epoch, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))
        log("test_loss")
        epoch_loss_test = test_loss / num_batch_test
        log(time.ctime() + ' epoch: {}/{} | Loss: {:.3f} | time: {}'.format(
            epoch, n_epoch, epoch_loss_test, str(datetime.timedelta(seconds=runtime))))

        saver_agent.add_summary('epoch loss', epoch_loss)

        # save model, with policy
        loss = epoch_loss
        if 0.2 < loss:
            fn = int(loss * 10 + 1) * 10
        elif 0.001 < loss <= 0.20:
            fn = int(loss * 100)
        elif loss <= 0.001:
            log('Finished')
            return
        
        loss_test = epoch_loss_test
        if 0.5 < loss_test:
            fn_test = int(loss_test * 10 + 1) * 10
        else:
            fn_test = int(loss_test * 100)
        saver_agent.save_model(net, name='loss_' + str(fn) + '_test_' + str(fn_test))
        
        # if 1 < loss:
        #     fn = int(loss * 100)
        #     fn_test = int(loss_test * 100)
        #     saver_agent.save_model(net, name='loss_' + str(fn) + '_test_' + str(fn_test))
        # elif 0.001 < loss <= 1:
        #     fn = int(loss * 100)
        #     fn_test = int(loss_test * 100)
        #     saver_agent.save_model(net, name='loss_' + str(fn) + '_test_' + str(fn_test))
        # elif loss <= 0.001:
        #     log('Finished')
        #     return


if __name__ == '__main__':
    train_dp()
