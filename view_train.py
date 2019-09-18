# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import argparse
import os
import numpy as np

from mvcnn_model import MVCNN
from dataset_reader import MultiViewDataSet
from center_loss import CenterLoss

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
#dataset 
parser.add_argument('--datadir', type=str, default='./view_render_img')
parser.add_argument('--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--num-classes', type=int, default=90)
parser.add_argument('--lr-model', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=0.001, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9, help="learning rate decay")

# model
parser.add_argument('--model', type=str, default='resnet34')
parser.add_argument('--feat-dim', type=int, default=512)

# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-model-freq', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model-dir', type=str, default='model')
parser.add_argument('--count', type=int, default=0)

args = parser.parse_args()
writer = SummaryWriter()

def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    
    total = 0.0
    correct = 0.0
    train_size = len(trainloader)
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        data = np.stack(data, axis=1)
        data = torch.from_numpy(data)
        
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model.forward(data)
        
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent.forward(features, labels)
        #loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent*args.weight_cent
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        avg_acc = correct.item() / total
            
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()
        
        if (batch_idx + 1) % args.print_freq == 0:
            print("Iter [%d/%d] Total Loss: %.4f" % (batch_idx + 1, train_size, loss.item()))
            print("\tSoftmax Loss: %.4f" % (loss_xent.item()))
            print("\tCenter Loss: %.4f" % (loss_cent.item()))
            print("\tAverage Accuracy: %.4f" % (avg_acc))
            print("-------------------------- ")
            
        args.count+=1
        
        writer.add_scalar("Loss",loss.item(),args.count)
        writer.add_scalar("softmax loss",loss_xent.item(),args.count)
        writer.add_scalar("center loss",loss_cent.item(),args.count)
        writer.add_scalar("average accuracy",avg_acc,args.count)
    
def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    
    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")
        
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    dset_train = MultiViewDataSet(args.datadir, transform=transform)
    trainloader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    model = MVCNN(args.num_classes)
    model.cuda()
    
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=args.num_classes, feat_dim=args.feat_dim, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(args.max_epoch):
        #trainloader = iter(train_loader)
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        print("++++++++++++++++++++++++++")
            
        train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              trainloader, use_gpu, args.num_classes, epoch)
        
        if epoch % args.save_model_freq == 0:
            torch.save(model.state_dict(), args.model_dir+'/'+'3D_model.pth')

        if args.stepsize > 0: scheduler.step()
        
    writer.close()
        
if __name__ == '__main__':
    main()