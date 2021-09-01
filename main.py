
import os
import random
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import sys
sys.path.append('../DanceRevolution')
sys.path.append('../DanceRevolution/v2')
from agcn.graph.dance_revolution import DanceRevolutionGraph
from agcn.model.aagcn import Model
from dataset_holder import DanceRevolutionHolder
from dataset import DanceRevolutionDataset


def new_aagcn(num_classes=3):
    graph = DanceRevolutionGraph(labeling_mode='spatial')
    model = Model(num_class=num_classes, num_point=graph.num_node, in_channels=2, graph=graph)
    return model


def run_batch(input_tensor, model):
    # DM: this is just an example to show how the data has to be passed to the model
    B, C, T, V, M = input_tensor.shape
    # input shape:
    # B: batch size,
    # C=2 (xy channels),
    # T: length of sequence (n. of frames),
    # V=25, number of nodes in the skeleton,
    # M=1 number of bodies

    output = model(input_tensor)
    # will return a classification output for each element in the batch, already averaged over time. There should be
    # a dimension with size 1 corresponding to the single body for which we have data

    return output

# TODO: create a PyTorch dataset object feeding DanceRevolution's skeleton data in the expected format.
#  Look at attached notebook to understand Dance Revolution dance format. Look also at dataset_holder.py to see how
#  data can be loaded first and then fed via a Dataset object. I'm including a stub object in dataset.py for your
#  reference
def kfold_split(data_dir, n_splits=5, shuffle=False):
    fnames = sorted(os.listdir(data_dir))
    if shuffle:
        random.shuffle(fnames)
    num_per_split = int(len(fnames)/n_splits)
    print('total number:', len(fnames), 'num per split:', num_per_split)
    for i in range(n_splits):
        test_fnames = fnames[i*num_per_split:(i+1)*num_per_split]
        train_fnames = [item for item in fnames if item not in test_fnames]
        yield train_fnames, test_fnames

def load_data(file_list, split, args):
    # DanceRevolutionHolder transforms a sequence of raw json files into a SkeletonSequence class and store them
    # DanceRevolutionDataset is a subclass of torch.utils.data.Dataset which loads sequences stored in Holder
    # Dataloader loads data from Dataset and output requested sequences
    print('{} data loading'.format(split))
    if split == 'train':
        holder = DanceRevolutionHolder(args.train_dir, split, file_list)
        if args.use_bezier:
            dataset = DanceRevolutionDataset(holder, 'bcurve', bez_degree=5)
        else:
            dataset = DanceRevolutionDataset(holder)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_worker,
                            drop_last=True)
    elif split == 'test':
        holder = DanceRevolutionHolder(args.test_dir, split, file_list)
        if args.use_bezier:
            dataset = DanceRevolutionDataset(holder, 'bcurve', bez_degree=5)
        else:
            dataset = DanceRevolutionDataset(holder)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_worker,
                            drop_last=False)
    else:
        raise ValueError()

    print('{} data loaded'.format(split))
    return loader

def load_optimizer(opt_name, params, base_lr):
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr
        )
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr
        )
    else:
        raise ValueError()
    
    return optimizer

def get_accuracy(output, label):
    value, predict_label = torch.max(output.data, 1)
    return torch.mean((predict_label == label.data).float()).item()

def adjust_learning_rate(optimizer, epoch, args):
    if args.optimizer == 'SGD' or args.optimizer == 'Adam':
        if epoch < args.warm_up_epoch:
            lr = args.base_lr * (epoch + 1) / args.warm_up_epoch
        else:
            lr = args.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(args.step)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    else:
        raise ValueError()

def save_checkpoint(model, epoch_i, args):
    checkpoint = {
                'model': model.state_dict(),
                'args': args,
                'epoch': epoch_i
                }
    save_path = os.path.join(args.output_dir, 'epoch_{}.pt'.format(epoch_i))
    torch.save(checkpoint, save_path)

def train(train_list, test_list, model, creterion, args, writer):
    updates = 0
    loss_value = []
    optimizer = load_optimizer(args.optimizer, model.parameters(), args.base_lr)
    loader = load_data(train_list, 'train', args)
    
    running_loss = 0
    running_acc = 0

    print('Start training')
    for epoch_i in tqdm(range(1, args.epoch+1)):
        model.train()
        adjust_learning_rate(optimizer, epoch_i, args)
        for music, dance, label, metadata in loader:
            # get input
            input = Variable(dance.cuda(), requires_grad=False)
            label = Variable(label.cuda(), requires_grad=False)
            input.requires_grad_()

            # forward
            optimizer.zero_grad()
            output = run_batch(input, model)

            # backward
            loss = creterion(output, label)
            loss.backward()
            
            # update parameters
            optimizer.step()
            updates += 1

            # get statistics
            acc = get_accuracy(output, label)
            running_acc += acc
            running_loss += loss.detach().item()
            
        total_acc = running_acc/updates
        total_loss = running_loss/updates
        writer.add_scalar('train/accuracy', total_acc, updates)
        writer.add_scalar('train/loss', total_loss, updates)
        # print('loss=', total_loss, 'acc=', total_acc, 'iterations=', updates, 'epoch=', epoch_i)
        
        if epoch_i % args.save_per_epochs == 0 and args.save_model:
            save_checkpoint(model, epoch_i, args)
        
        if epoch_i % args.eval_per_epochs == 0:
            evaluate(test_list, model, epoch_i, creterion, args, writer)

def evaluate(test_list, model, epoch, creterion, args, writer):
    model.eval()

    running_loss = 0
    running_acc = 0
    num_batches = 0

    loader = load_data(test_list, 'test', args)

    for music, dance, label, metadata in loader:
        with torch.no_grad():
            # get input
            input = Variable(
                dance.float().cuda(), 
                requires_grad=False)
            label = Variable(
                label.long().cuda(), 
                requires_grad=False)

            # forward
            output = run_batch(input, model)
            loss = creterion(output, label)

            # get statistics
            running_acc += get_accuracy(output, label)
            running_loss += loss.detach().item()
            num_batches += 1
        
    total_acc = running_acc/num_batches
    total_loss = running_loss/num_batches
    writer.add_scalar('test/accuracy', total_acc, epoch)
    writer.add_scalar('test/loss', total_loss, epoch)
    return total_acc

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """ Main function """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_1min_discarded/bcurve', 
                        help='the directory of training data')
    parser.add_argument('--test_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_1min_discarded/bcurve',
                        help='the directory of testing data')
    parser.add_argument('--data_dir', type=str, default='/home/dingxi/DanceRevolution/data/all_1min_discarded/bcurve',
                        help='the directory of all data')
    parser.add_argument('--output_dir', metavar='PATH', default='checkpoints')

    parser.add_argument('--num_worker', type=int, default=16, help='the number of worker for DataLoader')
    parser.add_argument('--run_tensorboard', type=str2bool, default=True, help='Use tensorboard or not')
    parser.add_argument('--save_model', type=str2bool, default=False, help='Save model or not')
    parser.add_argument('--kfold_validation', type=str2bool, default=True, help='Do k-fold validation or not')
    parser.add_argument('--k_in_kfold', type=int, default=5)

    parser.add_argument('--use_bezier', type=str2bool, default=False, help='Use Bezier curve to smooth or not')
    
    # optimizer
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_per_epochs', type=int, metavar='N', default=5)
    parser.add_argument('--eval_per_epochs', type=int, default=5)
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[10, 30, 50], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--warm_up_epoch', default=0)

    args = parser.parse_args()
    
    # Use GPU
    device = torch.device('cuda')
    
    # Prepare K-fold data generator
    kfold = kfold_split(args.train_dir, args.k_in_kfold, shuffle=True)

    for i in range(args.k_in_kfold):
        print('*'*10, '{} in {}-fold test'.format(i+1, args.k_in_kfold), '*'*10)

        # Create AGCN
        net = nn.DataParallel(new_aagcn(), device_ids=[0,1]).to(device)

        # Define loss function
        creterion = nn.CrossEntropyLoss().to(device)

        # Set up Tensorboard
        writer = SummaryWriter()

        # Prepare file lists for training and testing
        train_list, test_list = next(kfold)

        # Training
        train(train_list, test_list, net, creterion, args, writer)

        if not args.kfold_validation:
            break

if __name__ == '__main__':
   main()