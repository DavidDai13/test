# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from center_loss import CenterLoss
from mvcnn_model import MVCNN
from dataset_reader import MultiViewDataSet

parser = argparse.ArgumentParser("feature extraction of sketch images")
# dataset
parser.add_argument('--test-datadir', type=str, default='view_render_img')
parser.add_argument('--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")

parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--num-samples', type=int, default=1258)

# misc
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_false')
parser.add_argument('--model-dir', type=str, default='model')

#features
parser.add_argument('--feat-dim', type=int, default=512)
parser.add_argument('--feat-dir', type=str, default='../features/3D_features/view_feature.mat')

args = parser.parse_args()

def main():
    #torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    #sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        #torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")
        
    transform = transforms.Compose([transforms.ToTensor()])

    dset_train = MultiViewDataSet(args.test_datadir, transform=transform)
    
    trainloader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = MVCNN(args.num_classes,use_gpu=True)
    
    #Load model    
    model.load_state_dict(torch.load(args.model_dir+'/'+'3D_model.pth'))
    model.cuda()
    model.eval()

    # Define two matrices to store extracted features
    view_feature = np.zeros((args.num_samples,args.feat_dim))
    view_labels = np.zeros((args.num_samples,1))
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        data = np.stack(data, axis=1)
        data = torch.from_numpy(data)
        #print(batch_idx)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        #print(batch_idx)
        features, outputs = model.forward(data)
        features_numpy = features.detach().cpu().clone().numpy()
        labels_numpy = labels.detach().cpu().clone().numpy()
        
        view_feature[batch_idx] = features_numpy
        view_labels[batch_idx] = labels_numpy
        if batch_idx % 100 == 0:
            print("==> test samplses [%d/%d]" % (batch_idx,args.num_samples))
        #print(features_numpy)
    
    #save features as .mat file
    feat_data = {'view_feat': view_feature,'view_label': view_labels}
    sio.savemat(args.feat_dir, feat_data, oned_as='column')

 
        
if __name__ == '__main__':
    main()