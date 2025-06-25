import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

parser = argparse.ArgumentParser('training')
parser.add_argument('--model_epochs', help='use cpu mode')
args = parser.parse_args()
model_epochs = args.model_epochs

def plot_loss_train(path_train, path_val, save_path):
    sns.set_style("whitegrid")

    # Read data from files
    train_data = np.loadtxt(path_train)
    val_data = np.loadtxt(path_val)

    train_epochs = train_data[:, 0]
    train_loss = train_data[:, 1]
    val_epochs = val_data[:, 0]
    val_loss = val_data[:, 1]

    # Create combined loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, 'o-', color='orange', linewidth=2, markersize=3, label='Train Loss')
    plt.plot(val_epochs, val_loss, 'o-', color='purple', linewidth=2, markersize=3, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Training and Validation Loss vs Epochs', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + '/train_val_loss_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plot saved as train_val_loss_plot.png")

def plot_acc_rec_prec(train_path, val_path, save_path):
    # Set seaborn style
    sns.set_style("whitegrid")

    # Read data from files
    train_data = np.loadtxt(train_path)
    val_data = np.loadtxt(val_path)

    train_epochs = train_data[:, 0]
    train_acc = train_data[:, 1]
    train_rec = train_data[:, 2]
    train_prec = train_data[:, 3]

    val_epochs = val_data[:, 0]
    val_acc = val_data[:, 1]
    val_rec = val_data[:, 2]
    val_prec = val_data[:, 3]

    # Create separate plots for each metric
    metrics = [
        ('Accuracy', train_acc, val_acc),
        ('Precision', train_prec, val_prec),
        ('Recall', train_rec, val_rec)
    ]

    for name, train_values, val_values in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(train_epochs, train_values, 'o-', color='orange', linewidth=2, markersize=6, label=f'Train {name}')
        plt.plot(val_epochs, val_values, 'o-', color='purple', linewidth=2, markersize=6, label=f'Validation {name}')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.title(f'Training and Validation {name} vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path + f'/{name.lower()}_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("Accuracy, Recall and Precision plots saved")

def plot_roc(file_path, save_path):
    # ModelNet40 classes
    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
            'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
            'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
            'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
            'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
            'wardrobe', 'xbox']
    
    # Read file
    labels, scores = [], []
    with open(file_path + '/roc_info.txt', 'r') as f:
        content = f.read()
        
    # Split by class names to get each sample
    import re
    pattern = r'(\w+)\s+(\[.*?\])'
    matches = re.findall(pattern, content, re.DOTALL)

    for label, score_str in matches:
        labels.append(label)
        # Remove brackets and newlines, then convert to array
        clean_scores = score_str.strip('[]').replace('\n', ' ')
        scores.append(np.fromstring(clean_scores, sep=' '))

    # Convert to arrays
    y_true = np.array([classes.index(label) for label in labels])
    y_scores = np.array(scores)

    # Binarize labels
    y_bin = label_binarize(y_true, classes=range(len(classes)))

    # Calculate ROC for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        if roc_auc > 0.7:  # Only show good performers
            plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - ModelNet40')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(metrics_folder+'/roc.png')
    plt.show()
    print('ROC Curve plotted')

dumped_folder = 'log/classification/pointnet2_cls_ssg/dumped/' + model_epochs + '_epochs'

metrics_folder = 'log/classification/pointnet2_cls_ssg/metrics/' + model_epochs +'_epochs'

os.makedirs(metrics_folder, exist_ok=True)

plot_loss_train(dumped_folder + '/train_loss.txt', dumped_folder + '/val_loss.txt', metrics_folder)

plot_acc_rec_prec(dumped_folder+ '/acc_rec_prec_train.txt', dumped_folder + '/acc_rec_prec_val.txt', metrics_folder)

plot_roc(dumped_folder, metrics_folder)