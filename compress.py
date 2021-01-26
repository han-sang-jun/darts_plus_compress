import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
import genotypes
from model_compress import CompressedNetwork as Network
from genotypes import PRIMITIVES
from genotypes import Genotype
from genotypes import op_graph

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.02, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--eps_no_archs', type=int, default=15, help='epochs to do not train arch params')
parser.add_argument('--train_data_dir', type=str, help='train data dir')
parser.add_argument('--test_data_dir', type=str, help='test data dir')
parser.add_argument('--inter_nodes', type=int, default=4)
parser.add_argument('--stem_multiplier', type=int, default=3)
parser.add_argument('--stable_arch', type=int, default=5)
parser.add_argument('--residual_connection', type=bool, default=False)
parser.add_argument('--arch', type=str)
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')

args = parser.parse_args()


def model_compress(args):
    if os.path.isdir(args.save) == False:
        os.makedirs(args.save)
    save_dir = '{}compress-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(save_dir, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.cifar100:
        CIFAR_CLASSES = 100
        data_folder = 'cifar-100-python'
    else:
        CIFAR_CLASSES = 10
        data_folder = 'cifar-10-batches-py'

    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    #  prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.cifar100:
        train_data = dset.CIFAR100(root=args.train_data_dir, train=True, download=True, transform=train_transform)
    else:
        train_data = dset.CIFAR10(root=args.train_data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    iter_per_one_epoch = num_train // (2 * args.batch_size)
    if iter_per_one_epoch >= 100:
        train_extend_rate = 1
    else:
        train_extend_rate = (100 // iter_per_one_epoch) + 1

    iter_per_one_epoch = iter_per_one_epoch * train_extend_rate
    logging.info('num original train data: %d', num_train)
    logging.info('iter per one epoch: %d', iter_per_one_epoch)

    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))
    train_set = torch.utils.data.Subset(train_data, indices[:split])
    valid_set = torch.utils.data.Subset(train_data, indices[split:num_train])

    train_set = torch.utils.data.ConcatDataset([train_set] * train_extend_rate)
    # valid_set = torch.utils.data.ConcatDataset([valid_set]*train_extend_rate)

    train_queue = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(train_set),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(valid_set),
        pin_memory=True, num_workers=args.workers)

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    eps_no_arch = args.eps_no_archs
    epochs = args.epochs

    if args.arch in genotypes.__dict__.keys():
        genotype = eval("genotypes.%s" % args.arch)
    else:
        genotype = eval(args.arch)

    model = Network(genotype, args.init_channels, CIFAR_CLASSES, args.layers, criterion,
                    steps=args.inter_nodes, multiplier=args.inter_nodes,
                    stem_multiplier=args.stem_multiplier,
                    residual_connection=args.residual_connection)
    model = nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    network_params = []
    for k, v in model.named_parameters():
        if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
            network_params.append(v)

    optimizer = torch.optim.SGD(
        network_params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                                   lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=args.arch_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(epochs), eta_min=args.learning_rate_min)

    scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, 30, gamma=0.2)

    train_epoch_record = -1
    arch_train_count = 0
    prev_geno = ''
    prev_rank = None
    rank_geno = None
    result_geno = None
    arch_stable = 0
    best_arch_stable = 0

    for epoch in range(epochs):

        lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)
        epoch_start = time.time()
        # training
        if epoch < eps_no_arch:
            train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer,
                                         optimizer_a, lr, train_arch=False)

        else:
            ops, probs = compressing_parse(model)
            concat = range(2, 2 + model.module._steps)
            genotype = Genotype(
                normal=ops[0], normal_concat=concat,
                reduce=ops[1], reduce_concat=concat,
            )

            if str(prev_geno) != str(genotype):
                prev_geno = genotype
                logging.info(genotype)

            # early stopping

            stable_cond = True
            rank = []
            for i in range(len(probs)):
                rank_tmp = ranking(probs[i])
                rank.append(rank_tmp)

            if prev_rank != rank:
                stable_cond = False
                arch_stable = 0
                prev_rank = rank
                rank_geno = genotype
                logging.info('rank: %s', rank)

            if stable_cond:
                arch_stable += 1

            if arch_stable > best_arch_stable:
                best_arch_stable = arch_stable
                result_geno = rank_geno
                logging.info('arch_stable: %d', arch_stable)
                logging.info('best genotype: %s', rank_geno)

            if arch_stable >= args.stable_arch - 1:
                logging.info('stable genotype: %s', rank_geno)
                result_geno = rank_geno
                break

            train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer,
                                         optimizer_a, lr, train_arch=True)
            arch_train_count += 1

            scheduler_a.step()

        scheduler.step()
        logging.info('Train_acc %f, Objs: %e', train_acc, train_obj)
        epoch_duration = time.time() - epoch_start
        logging.info('Epoch time: %ds', epoch_duration)

        # validation
        if epoch >= eps_no_arch:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('Valid_acc %f, Objs: %e', valid_acc, valid_obj)

        # # early arch training
        # if train_epoch_record == -1:
        #     if train_acc > 70:
        #         arch_train_num = args.epochs - args.eps_no_archs
        #         eps_no_arch = 0
        #         train_epoch_record = epoch
        # else:
        #     if epoch >= train_epoch_record + arch_train_num:
        #         break

        utils.save(model, os.path.join(save_dir, 'weights.pt'))

    # last geno parser
    ops, probs = compressing_parse(model)
    concat = range(2, 2 + model.module._steps)
    genotype = Genotype(
        normal=ops[0], normal_concat=concat,
        reduce=ops[1], reduce_concat=concat,
    )
    logging.info('Last geno: %s', genotype)

    if result_geno == None:
        result_geno = genotype

    return result_geno, best_arch_stable


def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0 and step // args.report_freq > 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R2: %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg


# ranking
def ranking(prob):
    rank = []
    for i in range(len(prob)):
        rank.append(list(np.argsort(prob[i])))

    return rank


# network parsing

def compressing_parse(model):
    with torch.no_grad():
        genotype = model.module.genotype
        cell_names = ['normal', 'reduce']
        param_names = ['alphas_normal', 'alphas_reduce']
        op_result = []
        prob_result = []
        for i in range(len(cell_names)):
            ops = []
            probs = []
            op_names, indices = zip(*getattr(genotype, cell_names[i]))
            arch_param = getattr(model.module, param_names[i])
            for j in range(len(arch_param)):
                argmax_index = np.argmax(arch_param[j].cpu().numpy())
                op = op_graph[op_names[j]][argmax_index]
                ops.append((op, indices[j]))
                probs.append(F.softmax(arch_param[j], dim=0).data.cpu().numpy())

            op_result.append(ops)
            prob_result.append(probs)

        return op_result, prob_result


# main

if __name__ == '__main__':
    start_time = time.time()

    result_geno, best_arch_stable = model_compress(args)

    logging.info('arch stable: %d', best_arch_stable)
    logging.info('result genotype: %s', result_geno)

    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)



