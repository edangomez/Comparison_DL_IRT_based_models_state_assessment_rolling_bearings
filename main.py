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
    # parser.add_argument('--save', type=str, default='model.pt',
    #                     help='file on which to save model weights')
    parser.add_argument('--load', action='store_true', default=False,
                        help='Load presaved visual dict or checkpoints')
    parser.add_argument('--experiment_name', type=str, default='experiment1',
                        help='file on which to save model weights')
    parser.add_argument('--size', type=str, default='b0', choices=['b0', 'original'],
                        help='for hybrid models, if you want to resize data to efficientNet resolution')
    parser.add_argument('--resize_type', type=str, default='pad',choices=['pad', 'interpol'],
                        help='for hybrid models and no resize if padding is desired or interpolation')
    parser.add_argument('--save', type=str, default='no',choices=['yes', 'no'],
                        help='whether you want to save or not the testing')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.dtype == 'img':
        save_path = os.path.join('Experiments_img', args.experiment_name)
    elif args.dtype == 'irt':
        save_path = os.path.join('Experiments_irt', args.experiment_name)
    elif args.dtype in ['hybrid', 'hybrid2', 'hybrid3']:
        save_path = os.path.join('Experiments_hybrid', args.experiment_name)
    os.makedirs(save_path, exist_ok=True)

    ROOT_PATH='/home/edgomez10/Project/TB-and-IB-analysis-of-IRT-for-the-state-assessment-of-rolling-bearings-using-DL/data'    

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
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

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

    model = model = EfficientNet(args.dtype, version='b0', num_classes=4)
    state_dict = torch.load('/home/edgomez10/Project/TB-and-IB-analysis-of-IRT-for-the-state-assessment-of-rolling-bearings-using-DL/Experiments_hybrid/efficientNEt_featvect_noResize_interpol/epoch_40.pt')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    loss_test, accuracy,aca,f1_score,confusion_mx = testNet(args,model,x_test,y_test, dtype = args.dtype, gpuID = args.gpuID)
    print(f'Accuracy score for {args.dtype} is {accuracy*100:.2f}% and average loss is {loss_test:.2f}')
    print(f'ACA score for {args.dtype} is {aca*100:.2f}%')
    if args.save == 'yes':
        print('saving ...')
        results = pd.DataFrame({'loss_test': [loss_test], 'accuracy': [accuracy], 'ACA':[aca],'f1_score': [f1_score], 'confusion_mat':[confusion_mx.reshape(16)]})
        results.to_csv(os.path.join(save_path, f'metrics_{args.experiment_name}.csv'))

if __name__ == '__main__':
    main()