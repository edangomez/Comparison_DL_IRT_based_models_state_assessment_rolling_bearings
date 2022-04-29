from cProfile import label
import time
import argparse
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optimize 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from utils.dataloader import IRT
from models.CNN_model import VargoNet


CALTECH_PATH='/media/SSD0/datasets/101_ObjectCategories'

def main():
    parser = argparse.ArgumentParser(description='Caltech 101 classification')

    parser.add_argument('--model', type=str, default='CNN', choices = ['CNN', 'BoW'],
                        help='Select which model you want to train')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', default='1',metavar='DEVICE_NUMBER',
                        help='select gpu number to run the training')

    parser.add_argument('--batch_size', type=int, default=16, 
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dataAugmentation', action='store_true', default=False,
                        help='Decide if you want to run data augmentation transforms')

    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before '
                            'saving training status')
    parser.add_argument('--save', type=str, default='model.pth',
                        help='file on which to save model weights')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load presaved weights or checkpoints')
    parser.add_argument('--experiment_name', type=str, default='experiment1',
                        help='file on which to save model weights')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.cuda.set_device(int(args.device)) 

    save_path = os.path.join('Experiments', args.experiment_name)
    os.makedirs(save_path, exist_ok=True)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
            
    df_images = pd.read_csv('caltech101.csv',index_col=0)
    images_paths = df_images['path'].values
    labels_idx = df_images['label_idx'].values

    # Split proportional for each category
    x_train, x_test, y_train, y_test = train_test_split(images_paths, labels_idx,
                                                        test_size=0.1,
                                                        random_state=1234,
                                                        stratify=labels_idx)


    if args.model == 'CNN':
        model=trainVargoNet(args,x_train,y_train,save_path)
        loss_test, accuracy,aca,f1_score,confusion_mx = testVargoNet(args,model,x_test,y_test)
        print(f'Accuracy score for {args.model} is {accuracy*100:.2f}% and average loss is {loss_test:.2f}')
        print(f'ACA score for {args.model} is {aca*100:.2f}%')
    else:
        print('Select a valid model: [CNN,BoW]')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def trainVargoNet(args,x_train,y_train,save_path):

    model = VargoNet()
    if args.cuda:
        model = model.cuda()
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    print(f'RUNNING IN GPU {next(model.parameters()).is_cuda} # {torch.cuda.current_device()}')

    if args.dataAugmentation:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.53, 0.32),
            transforms.RandomAffine(
                degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip()
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.53, 0.32),
        ])
    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn':seed_worker,'generator':g} if args.cuda else {}
    train_dataset=IRT(x_train, y_train, CALTECH_PATH, transform=train_transforms) #transforms.Compose([transforms.ToTensor()]), train_transforms
    
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
    
    ## Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimize.Adam(model.parameters(), lr= args.lr)#optimize.SGD(model.parameters(), lr= args.lr, momentum=args.momentum) 

    start = time.time()
        
    for epoch in range(1,args.epochs+1): 

        model.train()
        train_loss = 0

        for batch_idx, (data,target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            ## Forward Pass
            data = data.float()
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(f'Batch: {batch_idx} Loss in epoch: {epoch}: {loss} ' )

        print(f"Avg Loss in epoch {epoch} :  {train_loss/len(train_loader)}")       
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss }

        checkpoint = epoch % args.save_interval == 0
        
        if checkpoint:
            name = 'epoch_' + str(epoch) + '.pth.tar'
            torch.save(state, os.path.join(save_path, name))
            print('Checkpoint saved:', name)

    print(f'TOTAL TRAINING TIME: {(time.time()-start)/60:2f} min')

    return model



def testVargoNet(args,model,x_test,y_test):

    if args.cuda:
        model = model.cuda()
    g = torch.Generator()
    g.manual_seed(args.seed)

    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),  
                                        transforms.Normalize([0.53], [0.32])])
    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn':seed_worker,'generator':g} if args.cuda else {}

    test_dataset=Caltech101(x_test, y_test, CALTECH_PATH, transform=test_transforms)

    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=True,**kwargs)
    
    criterion = nn.CrossEntropyLoss()
 
    model.eval()
    loss_test = 0

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        
        preds = torch.tensor([]).cuda()
        labels = torch.tensor([]).cuda()

        for batch_idx, (data,target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            data = data.float()
            ## Forward Pass
            scores = model(data)
            loss = criterion(scores,target)
            _, predictions = scores.max(1) #Apply log_softmax activation to the predictions and pick the index of highest probability.
            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)
            loss_test += loss.item()
            accuracy = num_correct /num_samples
            preds=torch.cat((preds,predictions),0)
            labels=torch.cat((labels,target),0)

        print(f"Got {num_correct} / {num_samples}") #with accuracy {float(accuracy)* 100:.2f}

    #METRICS
    aca,recall, f1_score, support = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average='macro',zero_division=0)
    confusion_mx = confusion_matrix(labels.cpu(), preds.cpu())

    return loss_test/len(test_loader), accuracy,aca,f1_score,confusion_mx

if __name__ == '__main__':
    main()

