"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import torch.nn as nn
import pickle
import open3d as o3d
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import confusion_matrix
import seaborn as sns
from umap import UMAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

y_true = list()
y_pred = list()
y_pred_scores = list()

# Load class numbers
with open('data/modelnet40_category_numbered','rb') as f:
    classes = pickle.load(f)

def get_class_name(value):
    for key,val in classes.items():
        if val == value:
            return key

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--model_nepochs', help='Model trained upto that epoch number')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (fn, points, target) in tqdm(enumerate(loader), total=len(loader)):
        
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred

        pred = vote_pool / vote_num

        pred_scr_batch = pred.detach().cpu().numpy()
        
        for i in range(len(pred_scr_batch)):
            y_pred_scores.append(pred_scr_batch[i])

        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        # Create the inference folder
        path_to_inference = 'log/classification/pointnet2_cls_ssg/inference'
        os.makedirs(path_to_inference, exist_ok=True)
        path_to_misclassified = path_to_inference + '/misclassified'
        os.makedirs(path_to_misclassified, exist_ok=True)
        path_to_classified = path_to_inference + '/correct_classified'
        os.makedirs(path_to_classified, exist_ok=True)

        '''Inference'''

        pred_values = pred_choice.cpu().numpy()
        true_values = target.cpu().numpy()

        # Append the results to y_true and y_pred
        for i in range(len(true_values)):
            y_true.append(get_class_name(true_values[i]))
            y_pred.append(get_class_name(pred_values[i]))
        
        for i in range(len(true_values)):
            pred_cls_name = get_class_name(pred_values[i])
            true_cls_name = get_class_name(true_values[i])

            if pred_values[i]!=true_values[i]:
                
                mis_cat_dir = path_to_misclassified + '/' + true_cls_name

                os.makedirs(mis_cat_dir, exist_ok=True)

                mis_fname = mis_cat_dir + '/' + str(j) + '_' + str(i) + '_pred_' + pred_cls_name + '_gt_' + true_cls_name + '.txt'
                
                shutil.copyfile(fn[i], mis_fname)
            else:
                corr_cat_dir = path_to_classified + '/' + true_cls_name
                os.makedirs(corr_cat_dir, exist_ok=True)
                corr_fname = corr_cat_dir + '/' + str(j) + '_' + str(i) + '_pred_' + pred_cls_name + '_gt_' + true_cls_name + '.txt'
                shutil.copyfile(fn[i], corr_fname)

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def plot_confusion_matrix(y_true, y_pred, model_ckp):
    class_names = list()

    with open('data/modelnet40_normal_resampled/modelnet40_shape_names.txt') as f:
        for line in f:
            line = line.rstrip()
            class_names.append(line)

    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Classes')
    plt.xlabel('Predicted Classes')

    os.makedirs('log/classification/pointnet2_cls_ssg/metrics/' + str(model_ckp), exist_ok=True)

    cm_fname = 'log/classification/pointnet2_cls_ssg/metrics/' +  model_ckp + '/cm.png'
    plt.savefig(cm_fname)
    plt.show()


def plot_umap_fc2(model, test_loader, device='cuda'):
    print('Plotting UMAP')
    """Plot UMAP of fc2 layer activations"""
    
    # ModelNet40 class names in order
    class_names = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
        'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
        'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    
    # Storage for activations and labels
    activations = []
    labels = []
    
    # Hook function to capture fc2 output
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    # Register hook on fc2 layer (handle DataParallel)
    fc2_layer = model.module.fc2 if hasattr(model, 'module') else model.fc2
    hook = fc2_layer.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for fn, points, target in test_loader:
            points = points.to(device)
            points = points.transpose(2, 1)
            model(points)
            labels.extend(target.numpy())
    
    # Remove hook
    hook.remove()
    
    # Concatenate all activations
    activations = np.concatenate(activations, axis=0)
    labels = np.array(labels)
    
    # Apply UMAP
    n_components_list = [2,3,4,5,6,7,8,9]
    metric_list = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    min_dist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_neighbors_list = [2, 3, 4, 5, 6, 7, 8, 9]

    umap_folder = 'log/classification/pointnet2_cls_ssg/umap'
    os.makedirs(umap_folder, exist_ok=True)

    for c in n_components_list:
        for m in metric_list:
            for md in min_dist:
                for nb in n_neighbors_list:
                    umap_model = UMAP(n_components=c, random_state=42, metric=m, n_neighbors=nb, min_dist=md)
                    # Best n_components=2, random_state=42, metric='euclidean', n_neighbors=6, min_dist=0.5
                    embedding = umap_model.fit_transform(activations)
                    
                    # Plot
                    plt.figure(figsize=(15, 10))
                    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=20)
                    
                    # Create legend with class names
                    handles = []
                    for i in range(len(class_names)):
                        if i in labels:  # Only show classes that exist in the data
                            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=plt.cm.tab20(i/len(class_names)), 
                                                    markersize=8, label=class_names[i]))
                    
                    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.title('UMAP of FC2 Layer Activations - ModelNet40', fontsize=10)
                    plt.xlabel('UMAP 1', fontsize=10)
                    plt.ylabel('UMAP 2', fontsize=10)
                    plt.tight_layout()
                    plt.savefig(umap_folder + f'/umap_nb_{nb}_comp_{c}_met_{m}_mindist_{md}.png', dpi=300, bbox_inches='tight', facecolor='white')
                    plt.show()

    print('UMAP plot saved')



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"]

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = nn.DataParallel(model.get_model(num_class, normal_channel=args.use_normals))
    if not args.use_cpu:
        classifier = classifier.cuda()

    model_ckp = args.model_nepochs

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/' + str(model_ckp) + '_model.pth', weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        # plot_confusion_matrix(y_true, y_pred, model_ckp)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


    # # Dump ROC info
    # dump_folder = 'log/classification/pointnet2_cls_ssg/dumped/' + model_ckp
    # os.makedirs(dump_folder, exist_ok=True)

    # with open(dump_folder + '/roc_info.txt', 'a') as f:
    #     for i in range(len(y_true)):
    #         # Dump test: y_true and scores
    #         f.write("{} {}\n".format(y_true[i], y_pred_scores[i]))

    # plot_umap_fc2(classifier, testDataLoader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
