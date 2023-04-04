
#初始化客户端
#输入参数分别为：num_norm(norm类型客户端数量)、num_free(free类型客户端数量)、num_avsl(avsl类型客户端数量)、
def initClients(num_norm, num_free, num_avsl, server, dataloaders):
    '''
        Initializes clients objects and returns a list of client object 初始化客户端对象并返回客户端对象列表
    '''

    print('Initializing clients...')
    # Setup client devices 设置客户端设备
    behaviour_list = [
        *['NORMAL' for i in range(num_norm)],
        *['FREERIDER' for i in range(num_free)],
        *['ADVERSARIAL' for i in range(num_avsl)],
    ]

    clients = []
    for n, behaviour in enumerate(behaviour_list):
        # Spawn client model and functions 衍生客户端模型和功能
        client_name = f'client_{n}'

        # Collect client's objects into a reference dictionary 将客户端的对象收集到参考字典中
        clients += [{
            'name': client_name,
            'behaviour': behaviour,
            'filepath': f'{server["client_filepath"]}/{client_name}.pt',
            'dataloader': dataloaders[n]
        }]

    print('Client Name / Behaviour:', [(client['name'], client['behaviour']) for client in clients], '\n')

    return clients

