"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import pickle
import datetime
import logging
import provider
import importlib
import shutil
import argparse
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from sklearn.metrics import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Load class numbers
with open('data/modelnet40_category_numbered','rb') as f:
    classes = pickle.load(f)

def get_class_name(value):
    for key,val in classes.items():
        if val == value:
            return key

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, epoch, n_epochs, num_class=40):
    total_loss = 0
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    model_ = importlib.import_module(args.model)
    criterion = model_.get_loss()
    y_true_val = list()
    y_pred_val = list()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    for j, (fn, points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, trans_feat = classifier(points)
        
        loss = criterion(pred, target.long(), trans_feat)
        loss = loss.cpu().numpy()
        total_loss += loss
        pred_choice = pred.data.max(1)[1]
        
        pred_values = pred_choice.cpu().numpy()
        true_values = target.cpu().numpy()

        # Append the results to y_true_val and y_pred_val
        for i in range(len(true_values)):
            y_true_val.append(get_class_name(true_values[i]))
            y_pred_val.append(get_class_name(pred_values[i]))
        
        
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))


    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    
    # Dump important information for plotting
    dump_folder = 'log/classification/pointnet2_cls_ssg/dumped/' + str(n_epochs) +'_epochs'
    os.makedirs(dump_folder, exist_ok=True)
    
    # Calculate acc, rec and prec
    acc_score = accuracy_score(y_true_val, y_pred_val)
    rec_score = recall_score(y_true_val, y_pred_val, average='micro')
    prec_score = precision_score(y_true_val, y_pred_val, average='micro')

    # Dump acc, rec, prec 
    with open(dump_folder + '/acc_rec_prec_val.txt', 'a') as f:
        f.write("{} {} {} {}\n".format(epoch, acc_score, rec_score, prec_score))

    # Dump train: iter vs loss
    with open(dump_folder + '/val_loss.txt', 'a') as f:
        f.write("{} {}\n".format(epoch, total_loss))
        
    print('Validation loss:', total_loss)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''

    os.environ["CUDA_VISIBLE_DEVICES"]

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = nn.DataParallel(model.get_model(num_class, normal_channel=args.use_normals))
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        total_loss = 0
        y_true = list()
        y_pred = list()

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (fn, points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            total_loss += loss
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            pred_values = pred_choice.cpu().numpy()
            true_values = target.cpu().numpy()

            # Append the results to y_true and y_pred
            for i in range(len(true_values)):
                y_true.append(get_class_name(true_values[i]))
                y_pred.append(get_class_name(pred_values[i]))
        
        # Dump important information for plotting
        dump_folder = 'log/classification/pointnet2_cls_ssg/dumped/' + str(args.epoch) + '_epochs' 
        os.makedirs(dump_folder, exist_ok=True)

        # Dump train: iter vs loss
        with open(dump_folder + '/train_loss.txt', 'a') as f:
            f.write("{} {}\n".format(epoch, total_loss))

        # Calculate acc, rec and prec
        acc_score = accuracy_score(y_true, y_pred)
        rec_score = recall_score(y_true, y_pred, average='micro')
        prec_score = precision_score(y_true, y_pred, average='micro')

        # Dump acc, rec, prec 
        with open(dump_folder + '/acc_rec_prec_train.txt', 'a') as f:
            f.write("{} {} {} {}\n".format(epoch, acc_score, rec_score, prec_score))

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, epoch, args.epoch, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            if epoch % 10 == 0:
                # Save at every 10th epoch
                save_path = str(checkpoints_dir)
                state = {
                        'epoch': epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                torch.save(state, save_path + '/' + str(epoch) + '_epochs_model.pth')

            global_epoch += 1
    
    logger.info('End of training...')

    # Copy the information of loss, rec, acc, prec to checkpoints
    dump_folder = 'log/classification/pointnet2_cls_ssg/dumped/' + str(args.epoch) + '_epochs' 

    epochs = args.epoch
    epoch_list = list()
    for i in range(1, int(epochs/10)):
        epoch_list.append(i*10)

    for epoch in epoch_list:
        os.makedirs('log/classification/pointnet2_cls_ssg/dumped/' + str(epoch) + '_epochs', exist_ok=True)
        dump_name = 'log/classification/pointnet2_cls_ssg/dumped/' + str(epoch) + '_epochs'

        for info_files in os.listdir(dump_folder):
            with open(dump_folder + '/' + info_files) as f:
                lines = f.readlines()

                for line_number in range(epoch):
                    with open(dump_name + '/' + info_files, 'a') as f:
                        f.write('{}'.format(lines[line_number]))
    
    print('Necessary information copied')

if __name__ == '__main__':
    args = parse_args()
    main(args)
