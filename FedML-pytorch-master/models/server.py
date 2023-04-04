import torch
import sys,os
from utils.save_file import createDirectory, saveCheckpoint


#定义server()这个类，代表服务器
class server():
    def __init__(self, FederatedModel, FederatedLossFunc, FederatedOptimizer, FederatedLearnRate, FederatedMomentum, FederatedWeightDecay):
        self.FederatedModel = FederatedModel #模型
        self.FederatedLossFunc = FederatedLossFunc #损失函数
        self.FederatedOptimizer = FederatedOptimizer #优化器
        self.FederatedLearnRate = FederatedLearnRate #学习率
        self.FederatedMomentum = FederatedMomentum #动量
        self.FederatedWeightDecay = FederatedWeightDecay #权重衰退


    def initServer(self, model_path, folder_name, dataloader):
        '''
            Initializes server model and returns object with attributes 初始化服务器模型并返回具有属性的对象
        '''
        print('Initializing server model...')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Spawn server model and functions 衍生服务器模型和功能
        server_name = 'server'
        server_model = self.FederatedModel.to(device)
        server_loss_func = self.FederatedLossFunc
        #server_optimizer = torch.optim.AdamW(server_model.parameters(), lr=1e-4)
        server_optimizer = self.FederatedOptimizer(server_model.parameters(), lr=self.FederatedLearnRate,momentum=self.FederatedMomentum, weight_decay=self.FederatedWeightDecay)
        server_dataloader = dataloader

        print(server_model, '\n')
        print(server_optimizer)

        #创建新的目录
        createDirectory(f'{model_path}/{folder_name}/server')
        createDirectory(f'{model_path}/{folder_name}/client')

        # Collect objects into a reference dictionary 将对象收集到引用词典中，这里指服务器的相关信息
        server = {
            'name': server_name,
            'model': server_model,
            'dataloader': server_dataloader,
            'optimizer': server_optimizer,
            'loss_func': server_loss_func,
            'filepath': f'{model_path}/{folder_name}/server/server_model.pt',
            'client_filepath': f'{model_path}/{folder_name}/client'
        }

        # Save server model state_dicts (simulating public access to server model parameters) 保存服务器模型
        saveCheckpoint(
            server_name,
            server_model.state_dict(),
            server_optimizer.state_dict(),
            server['filepath'],
            verbose=True
        )

        return server
