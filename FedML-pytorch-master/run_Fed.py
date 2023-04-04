# Load libraries
import math, random, copy, os, glob, time
from itertools import chain, combinations, permutations
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import datasets, transforms as T

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from utils.parameters import args_parser
from utils.data_process import prepareIID, prepareNIID1, prepareNIID2, prepareNIID12
from models.Nets import SmallMLP_MNIST, MediumMLP_MNIST, LargeMLP_MNIST, SmallMLP_EMNIST, MediumMLP_EMNIST, LargeMLP_EMNIST
from utils.save_file import createDirectory, deleteAllModels, saveCheckpoint, print_parameters, loadCheckpoint
from models.Fed import FedAvg
from utils.helpers import powerset, grangerset, aggListOfDicts, getAllClients

from models.server import server
from models.clients import initClients


#新加的
from PIL import ImageFile
import torchvision

 
def train(dataloader, model, loss_fn, optimizer, verbose=False):  #verbose日志显示

    for batch_idx, (data, label) in enumerate(dataloader):
        if batch_idx == STEPS_PER_EPOCH:
            break
        optimizer.zero_grad()  # Resetting gradients after each optimizations

        train_len = 0
        train_correct = 0

        # Sending input , label to device
        data = data.to(device) 
        label = label.to(device)
        output = model(data)
        loss = loss_fn(output, label.reshape((BATCH_SIZE,)).long())
        
        # The loss variable has gradient attached to it so we are removing it so that it can be used to plot graphs
        loss.backward()
        optimizer.step()  # Optimizing the model

        # Checking train accuracy
        
        pred = torch.argmax(F.softmax(output, dim=1),dim=1)
        correct = pred.eq(label)

        train_len += len(label)
        train_correct += correct.sum().item()

        return loss



    '''
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error  计算预测损失值
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation  反向传播
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss
    '''


def test(dataloader, model, loss_fn, verbose=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_idx, (data, label) in enumerate(dataloader):
        if batch_idx == STEPS_PER_TEST_EPOCH:
            break

        with torch.no_grad():
            test_correct = 0
            test_len = 0
            
            data = data.to(device)
            label = label.long().to(device)
            output = model(data)
            loss = loss_fn(output, label)

            pred = torch.argmax(F.softmax(output, dim=1),dim=1)
            correct = pred.eq(label)
            
            test_len += len(label)
            test_correct += correct.sum().item()
    f1 = 0
    return loss, correct, f1



    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            f1 += f1_score(y.cpu(), y_pred.argmax(1).cpu(), average='micro')

    test_loss /= num_batches
    correct /= size
    f1 /= num_batches

    if verbose:
        print(f"Test Error: \n Accuracy: {correct:>8f}, Avg loss: {test_loss:>8f}, F1: {f1:>8f} \n")

    return test_loss, correct, f1
    '''

#训练一系列客户端
def trainClients(clients, server): 
    '''
        Trains a list of client devices and saves their parameters 训练一系列客户端设备列表并保存其参数
    '''
    loss, acc, f1 = {}, {}, {}
    for client in clients:
        train_loss, test_loss, test_acc, test_f1 = trainClient(client, server)

        # Aggregate statistics 汇总统计
        loss[client['name']] = test_loss
        acc[client['name']] = test_acc
        f1[client['name']] = test_f1

    return loss, acc, f1


#训练单个客户端
def trainClient(client, server):
    '''
        Trains a client device and saves its parameters 训练单个客户端设备并保存其参数
    '''

    #获取单个客户端的相关信息
    # Read client behaviour setting
    client_behaviour = client['behaviour']

    # Load local dataset
    client_dataloader = client['dataloader']

    # Get client model and functions
    client_name = client['name']

    client_model = FederatedModel().to(device)
    client_loss_fn = FederatedLossFunc()
    client_optimizer = FederatedOptimizer(client_model.parameters(), lr=FederatedLearnRate, momentum=FederatedMomentum,
                                          weight_decay=FederatedWeightDecay)

    # If client is adversarial, they return randomized parameters 对抗性客户端，返回随机参数
    if client_behaviour == 'ADVERSARIAL':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # Load server model state_dicts (simulating client downloading server model parameters)
    checkpoint = loadCheckpoint(server['filepath'])
    client_model.load_state_dict(checkpoint['model_state_dict'])  # Using current server model parameters
    # client_optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Using current server model parameters

    # If client is a freeloader, they return the same server model parameters 自由性客户端，返回相同参数
    if client_behaviour == 'FREERIDER':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # If client is normal, they train client over N epochs 正常客户端，N个epoch内训练模型
    epochs = args.epoch
    print(f'Training {client_name} over {epochs} epochs...')
    for t in range(epochs):
        train_loss = train(client_dataloader, client_model, client_loss_fn, client_optimizer)

    test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
    print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

    # Save client model state_dicts (simulating client uploading model parameters to server)
    saveCheckpoint(
        client_name,
        client_model.state_dict(),
        client_optimizer.state_dict(),
        client['filepath'],
    )

    return train_loss, test_loss, test_acc, test_f1



#聚合模型训练
def evalFedAvg(server):
    '''
        Load client state dicts, perform parameter aggregation and evaluate contributions for each client
        加载客户端状态，执行参数聚合并评估每个客户端的贡献
    '''
    # Retrieve all clients' uploaded data 检索所有客户端的上传数据
    client_filepaths = glob.glob(f"{server['client_filepath']}/client*.pt")

    # Load client model state_dicts (simulating client downloading server model parameters) 加载客户端模型状态
    client_checkpoints = []
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        client_checkpoints += [client_checkpoint]

    # Get Federated Average of clients' parameters 获取客户端参数的联合平均值
    model_state_dicts = [checkpoint['model_state_dict'] for checkpoint in client_checkpoints]
    fed_model_state_dict = FedAvg(model_state_dicts)

    # Instantiate server model using FedAvg 使用FedAvg实例化服务器模型
    fed_model = FederatedModel().to(device)
    fed_model.load_state_dict(fed_model_state_dict)
    fed_model.eval()

    # Evaluate FedAvg server model 评估FedAvg服务器模型
    start_time = time.time()  # Time evaluation period
    eval_loss, eval_acc, eval_f1 = test(server['dataloader'], fed_model, server['loss_func'])
    time_taken = time.time() - start_time  # Get model evaluation period (in seconds)
    print(f"\n>> Federated Model Acc: {eval_acc:>8f}, Loss: {eval_loss:>8f}, F1: {eval_f1:>8f}\n")

    # Save server model state_dicts (simulating public access to server model parameters)
    saveCheckpoint(
        server['name'],
        fed_model.state_dict(),
        server['optimizer'],
        server['filepath'],
    )

    # Output statistics
    return eval_loss, eval_acc, eval_f1, time_taken



def trainFedAvgModel(rounds): #输入参数为训练的轮数rounds
    '''
        Train a model using naive FedAvg 使用本地的FedAvg算法训练模型
    '''

    loss, acc, f1, eval_time = [], [], [], []
    for i in range(rounds):
        print(f'\n=======================\n\tROUND {i + 1}\n=======================')
        clients_loss, clients_acc, clients_f1 = trainClients(clients, server)
        server_loss, server_acc, server_f1, time_taken = evalFedAvg(server)

        # Compile performance measures 训练效果
        loss += [{**clients_loss, **{'server': server_loss}}]
        acc += [{**clients_acc, **{'server': server_acc}}]
        f1 += [{**clients_f1, **{'server': server_f1}}] #f1是什么指标？？？
        eval_time += [time_taken]

    # Output statistics
    return aggListOfDicts(loss), aggListOfDicts(acc), aggListOfDicts(f1), eval_time



#主函数部分
if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training 选择cpu或者gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"  

    # Create subdirectories 创建存放模型的子目录
    model_path = './models'

    # Initialize system and define helper functions 初始化系统并定义助手功能
    createDirectory(model_path)

    # Delete existing .pt files from previous run 从上次运行中删除现有.pt文件
    deleteAllModels(model_path)

    #训练集与测试集
    ImageFile.LOAD_TRUNCATED_IMAGES = True # To prevent error during loading broken images
    PATH_TRAIN = "train"
    PATH_TEST  = "test"

    BATCH_SIZE = 32

    TOTAL_SIZE = len(os.listdir(PATH_TRAIN + "/Normal")) + len(
    os.listdir(PATH_TRAIN + "/Infected"))
    TOTAL_TEST_SIZE = len(os.listdir(PATH_TEST + "/Normal")) + len(
    os.listdir(PATH_TEST + "/Infected"))
    STEPS_PER_EPOCH = TOTAL_SIZE // BATCH_SIZE
    STEPS_PER_TEST_EPOCH = TOTAL_TEST_SIZE // BATCH_SIZE


    IMAGE_H, IMAGE_W = 224, 224
    transform = torchvision.transforms.Compose(
    [  # Applying Augmentation
        torchvision.transforms.Resize((IMAGE_H, IMAGE_W)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(30),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])  # Normalizing data
    
    train_data = torchvision.datasets.ImageFolder(root=PATH_TRAIN, transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=PATH_TEST, transform=transform)

    
    # Split training dataset for clients 为客户端拆分训练数据集
    NUM_OF_CLIENTS = args.num_normal_clients + args.num_freerider_clients + args.num_adversarial_clients #客户端数量
    #分布类型distribution
    if args.distribution_type == 'IID':
        train_datasets = prepareIID(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_1':
        train_datasets = prepareNIID1(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_2':
        train_datasets = prepareNIID2(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_12':
        train_datasets = prepareNIID12(train_data, NUM_OF_CLIENTS)

    #加载数据集
    train_dataloaders = [DataLoader(train_dataset, batch_size=BATCH_SIZE) for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE * 2) #batch_size批次大小

    # Define network model architecture 定义网络模型体系结构
    FederatedModel = None #联邦学习网络模型
    FederatedModel = torchvision.models.resnet18(False)
    feature = FederatedModel.fc.in_features
    FederatedModel.fc = nn.Linear(feature, 2)
    FederatedModel.to(device)



    # Define network training functions and hyper-parameters 定义网络训练功能和超参数
    # Training hyper-parameters and functions for the Federated model 为联邦模型训练超参数和函数
    FederatedLossFunc = nn.CrossEntropyLoss() #损失函数
    FederatedOptimizer = torch.optim.AdamW(FederatedModel.parameters(), lr=1e-4)
    FederatedLearnRate = 0.0001 #学习率
    FederatedMomentum = args.momentum #动量
    FederatedWeightDecay = args.weight_decay #权重衰退


    # Initalize server and clients 初始化服务端和客户端
    #创建一个对象server，该对象为server类
    server = server(FederatedModel, FederatedLossFunc, FederatedOptimizer, FederatedLearnRate, FederatedMomentum,
                        FederatedWeightDecay)
    server = server.initServer(model_path, 'FedAvg', test_dataloader) #def initServer(self, model_path, folder_name, dataloader)
    #创建一系列客户端，clients是一个列表
    clients = initClients(args.num_normal_clients, args.num_freerider_clients, args.num_adversarial_clients, server,
                              train_dataloaders)

    # Train and evaluate 
    fedavg_loss, fedavg_acc, fedavg_f1, fedavg_time = trainFedAvgModel(args.common_rounds)