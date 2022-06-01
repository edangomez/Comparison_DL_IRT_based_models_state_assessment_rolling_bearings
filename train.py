from cProfile import label
import time
import argparse
from unittest import result
from xmlrpc.client import boolean
import pandas as pd
import numpy as np
import os
import tqdm

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

from util.dataloader import IRT, IRTHybrid, IRTHybrid2
from models.CNN import AlexNet, EfficientNet, HyAlexNet


ROOT_PATH='/home/edgomez10/Project/TB-and-IB-analysis-of-IRT-for-the-state-assessment-of-rolling-bearings-using-DL/data'    

def main():
    parser = argparse.ArgumentParser(description='Caltech 101 classification')

    parser.add_argument('--dtype', type=str, default='hybrid', choices = ['img', 'irt', 'hybrid', 'hybrid2', 'hybrid3'],
                        help='Select which model you want to train')
    parser.add_argument('--gpuID', type=int, default=1,
                        help='Select GPU ID for training and testing')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device', default='1',metavar='DEVICE_NUMBER',
                        help='select gpu number to run the training')

    parser.add_argument('--batchSize', type=int, default=20, 
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 16)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dataAugmentation', type = bool, default=False,
                        help='Decide if you want to run data augmentation transforms')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--save_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before '
                            'logging training status')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='file on which to save model weights')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load presaved visual dict or checkpoints')
    parser.add_argument('--experiment_name', type=str, default='experiment1',
                        help='file on which to save model weights')
    parser.add_argument('--size', type=str, default='b0', choices=['b0', 'original'],
                        help='for hybrid models, if you want to resize data to efficientNet resolution')
    parser.add_argument('--resize_type', type=str, default='pad',choices=['pad', 'interpol'],
                        help='for hybrid models and no resize if padding is desired or interpolation')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.cuda.set_device(int(args.gpuID)) 

    if args.dtype == 'img':
        save_path = os.path.join('Experiments_img', args.experiment_name)
    elif args.dtype == 'irt':
        save_path = os.path.join('Experiments_irt', args.experiment_name)
    elif args.dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        save_path = os.path.join('Experiments_hybrid', args.experiment_name)
    os.makedirs(save_path, exist_ok=True)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
            
    if args.dtype == 'irt':    
        df_images = pd.read_csv('thermal_mat.csv',index_col=0).sort_values(by=['path'])
    elif args.dtype == 'img':
        df_images = pd.read_csv('thermal_img.csv',index_col=0).sort_values(by=['path'])
    elif args.dtype  in ['hybrid', 'hybrid2', 'hybrid3']:
        df_images = pd.read_csv('thermal_img.csv',index_col=0).sort_values(by=['path'])
        df_irt = pd.read_csv('thermal_mat.csv',index_col=0).sort_values(by=['path'])
        irt_paths = df_irt['path'].values


    images_paths = df_images['path'].values
    labels_idx = df_images['label'].values
    if args.dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        # images_paths = np.concatenate((np.array([irt_paths]).T, np.array([images_paths]).T), axis = 1)
        paths = np.concatenate((np.reshape(np.array(df_images['path']), (2298,1)), np.reshape(np.array(df_irt['path']), (2298,1))), 1)

    # Split proportional for each category
    if args.dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        x_train, x_test, y_train, y_test = train_test_split(paths, labels_idx,
                                                            test_size=0.3,
                                                            random_state=1234,
                                                            stratify=labels_idx)
    else:
        x_train, x_test, y_train, y_test = train_test_split(images_paths, labels_idx,
                                                            test_size=0.3,
                                                            random_state=1234,
                                                            stratify=labels_idx)
    model=trainNet(args, x_train, y_train, save_path, args.dtype, args.gpuID)
    loss_test, accuracy,aca,f1_score,confusion_mx = testNet(args,model,x_test,y_test, dtype = args.dtype, gpuID = args.gpuID)
    print(f'Accuracy score for {args.dtype} is {accuracy*100:.2f}% and average loss is {loss_test:.2f}')
    print(f'ACA score for {args.dtype} is {aca*100:.2f}%')
    # breakpoint()
    results = pd.DataFrame({'loss_test': [loss_test], 'accuracy': [accuracy], 'ACA':[aca],'f1_score': [f1_score], 'confusion_mat':[confusion_mx.reshape(16)]})
    results.to_csv(os.path.join(save_path, f'metrics_{args.experiment_name}.csv'))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def trainNet(args,x_train,y_train,save_path, dtype, gpuID = 1):

    ## Model setting
    # model = AlexNet(dtype=args.dtype)
    # model = HyAlexNet(dtype=args.dtype)
    model = EfficientNet(dtype, version='b0', num_classes=4)
    if args.cuda:
        model = model.cuda(gpuID)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    print(f'RUNNING IN GPU {next(model.parameters()).is_cuda} # {torch.cuda.current_device()}')


    ## Data loading
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
    # kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn':seed_worker,'generator':g} if args.cuda else {}
    # train_dataset=IRT(x_train, y_train, ROOT_PATH, transform=train_transforms, dtype=args.dtype) #transforms.Compose([transforms.ToTensor()]), train_transforms
    
    if dtype in ['irt', 'img']:
        print(f'Train {args.dtype} data loading ...')
        train_dataset=IRT(x_train, y_train, ROOT_PATH, transform=None, dtype=args.dtype, size=args.size)
    elif dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        print(f'Train {args.dtype} data loading ...')
        if args.size == 'b0':
            print('resizing ...')
            train_dataset=IRTHybrid(x_train, y_train, ROOT_PATH, transform=None) #transforms.Compose([transforms.ToTensor()]), train_transforms
        else:
            print('no resize')
            train_dataset=IRTHybrid2(x_train, y_train, ROOT_PATH, resize_type= args.resize_type,transform=None) #transforms.Compose([transforms.ToTensor()]), train_transforms
    train_loader = DataLoader(train_dataset,batch_size=args.batchSize, shuffle=True)#,**kwargs)
    
    ## Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimize.Adam(model.parameters(), lr= args.lr)#optimize.SGD(model.parameters(), lr= args.lr, momentum=args.momentum) 

    start = time.time()
    losses = []

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
            if args.cuda:
                data, target = data.cuda(args.gpuID), target.cuda(args.gpuID)
            data, target = Variable(data), Variable(target)
            # print('data shape: ', data.shape)
            # print('target: ', target.shape)
            data = data.float()
            optimizer.zero_grad()

            scores = model(data)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # if batch_idx % args.log_interval == 0:
            #     print('Batch: {} Loss in epoch: {} - {:.3f}'.format(batch_idx,epoch,loss) )
        return train_loss
    def train_hybrid(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, irt, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            if args.cuda:
                data, irt,  target = data.cuda(args.gpuID), irt.cuda(args.gpuID),target.cuda(args.gpuID)
            data, irt, target = Variable(data), Variable(irt), Variable(target)
            # print('data shape: ', data.shape)
            # print('irt shape: ', irt.shape)
            # print('final im shape: ', torch.cat((data, irt), axis = 1).shape)
            # print('target: ', target.shape)
            data = data.float()
            irt = irt.float()
            optimizer.zero_grad()

            scores = model(data, irt, gpuID)
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # if batch_idx % args.log_interval == 0:
            #     print(f'Batch: {batch_idx} --> Loss: {loss}')
        return train_loss
    train_l = []
    
    for epoch in range(1,args.epochs+1): 

        model.train()
        if args.dtype in ['hybrid', 'hybrid2', 'hybrid3']:
            train_loss = train_hybrid(epoch)
        else:
            train_loss = train(epoch)

        train_l.append(float(train_loss/len(train_loader)))
        # print(train_l)
        print(f"Avg Loss in epoch {epoch} :  {train_loss/len(train_loader)}")       
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss }

        checkpoint = epoch % args.save_interval == 0
        
        if checkpoint:
            name = 'epoch_' + str(epoch) + '.pt'
            torch.save(state, os.path.join(save_path, name))
            
            np_losses = np.array(losses)
            np_losses = {'loss': np_losses}
            np_losses = pd.DataFrame(np_losses)
            train_lo = pd.DataFrame(np.array(train_l))
            train_lo.to_csv(os.path.join(save_path, f'loss_{args.experiment_name}.csv'))
            # if epoch>10:
            #     breakpoint()
            np_losses.to_csv(os.path.join(save_path, f'{args.experiment_name}.csv'))
            print('Checkpoint saved:', name)

    print(f'TOTAL TRAINING TIME: {(time.time()-start)/60:2f} min')



    return model



def testNet(args,model,x_test,y_test, dtype, gpuID):

    if args.cuda:
        model = model.cuda()
    g = torch.Generator()
    g.manual_seed(args.seed)

    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),  
                                        transforms.Normalize([0.53], [0.32])])
    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn':seed_worker,'generator':g} if args.cuda else {}

    # test_dataset=IRT(x_test, y_test, ROOT_PATH, transform=None, dtype=args.dtype)
    if dtype in ['irt', 'img']:
        print(f'Test {args.dtype} data loading ...')
        test_dataset=IRT(x_test, y_test, ROOT_PATH, transform=None, dtype=args.dtype, size=args.size)
    elif dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        print(f'Test {args.dtype} data loading ...')
        if args.size == 'b0':
            test_dataset=IRTHybrid(x_test, y_test, ROOT_PATH, transform=None) #transforms.Compose([transforms.ToTensor()]), train_transforms
        else:
            test_dataset=IRTHybrid2(x_test, y_test, ROOT_PATH, resize_type= args.resize_type, transform=None) #transforms.Compose([transforms.ToTensor()]), train_transforms

    test_loader = DataLoader(test_dataset,batch_size=args.batchSize, shuffle=True,**kwargs)
    
    criterion = nn.CrossEntropyLoss()
 
    model.eval()
    loss_test = 0

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        
        preds = torch.tensor([]).cuda(gpuID)
        labels = torch.tensor([]).cuda(gpuID)

        if dtype in ['irt', 'img']:

            for batch_idx, (data,target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                if args.cuda:
                    data, target = data.cuda(gpuID), target.cuda(gpuID)
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
        elif dtype  in ['hybrid', 'hybrid2', 'hybrid3']:
            for batch_idx, (data, irt, target) in tqdm.tqdm(enumerate(test_loader), total = len(test_loader)):
                if args.cuda:
                    data, irt, target = data.cuda(gpuID), irt.cuda(gpuID),target.cuda(gpuID)
                data, irt, target = Variable(data), Variable(irt), Variable(target)
                data, irt = data.float(), irt.float()

                scores = model(data, irt, args.gpuID)
                loss = criterion(scores, target)
                _, predictions = scores.max(1)
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

# for batch_idx, (data,irt, target) in enumerate(test_loader):
#             if args.cuda:
#                 data, irt, target = data.cuda(gpuID), irt.cuda(gpuID), target.cuda(gpuID)
#             data, irt, target = Variable(data), Variable(irt), Variable(target)
#             data = data.float()
#             irt = irt.float()
#             ## Forward Pass
#             outputs = model(data, irt)
#             loss = criterion(outputs,target)
#             # _, predictions = scores.max() #Apply log_softmax activation to the predictions and pick the index of highest probability.
#             _, predictions = torch.max(outputs, 1)
#             num_correct += (predictions == target).sum()
#             num_samples += predictions.size(0)
#             loss_test += loss.item()
#             accuracy = num_correct /num_samples
#             # breakpoint()
#             preds=torch.cat((preds,predictions),0)
#             labels=torch.cat((labels,target),0)
