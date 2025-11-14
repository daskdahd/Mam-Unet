import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

# 🔥 新增函数：创建解冻时的差异化学习率优化器
def create_unfreeze_optimizer(model, base_lr, optimizer_type='adam'):
  
    # 🎯 参数分组策略
    backbone_params = []
    transmamba_params = []
    decoder_params = []
    attention_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'transmamba' in name.lower():
                transmamba_params.append(param)
            elif any(att in name.lower() for att in ['attention', 'eca', 'se', 'cbam', 'ema', 'bra']):
                attention_params.append(param)
            elif 'resnet' in name.lower():
                backbone_params.append(param)
            else:
                decoder_params.append(param)

    
    # 🔥 创建差异化学习率优化器
    param_groups = []
    
    # backbone: 极小学习率保护预训练权重
    if len(backbone_params) > 0:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * 0.05,  # 🔥 只用5%的学习率!
            'weight_decay': 1e-4,
            'name': 'backbone'
        })
    
    # TransMamba: 小学习率
    if len(transmamba_params) > 0:
        param_groups.append({
            'params': transmamba_params,
            'lr': base_lr * 0.2,   # 🔥 20%的学习率
            'weight_decay': 1e-4,
            'name': 'transmamba'
        })
    
    # 注意力机制: 中等学习率
    if len(attention_params) > 0:
        param_groups.append({
            'params': attention_params,
            'lr': base_lr * 0.3,   # 🔥 30%的学习率
            'weight_decay': 1e-4,
            'name': 'attention'
        })
    
    # decoder: 正常学习率
    if len(decoder_params) > 0:
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr,         # 🔥 100%的学习率
            'weight_decay': 1e-4,
            'name': 'decoder'
        })
    
    # 创建优化器
    if optimizer_type == 'adam':
        optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)
    
    # 打印学习率配置
    for i, group in enumerate(optimizer.param_groups):
        print(f"   {group.get('name', f'group_{i}')}: lr={group['lr']:.2e}")
    
    return optimizer

def gradual_unfreeze_strategy(model, epoch, freeze_epoch):
    """
    渐进式解冻策略，避免突然解冻导致的精度下降
    """
    if epoch < freeze_epoch:
        # 完全冻结阶段
        for name, param in model.named_parameters():
            if 'resnet' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        return "frozen"
        
    elif epoch < freeze_epoch + 3:
        # 阶段1: 只解冻layer4和TransMamba
        for name, param in model.named_parameters():
            if any(x in name for x in ['layer4', 'transmamba']):
                param.requires_grad = True
            elif 'resnet' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        return "partial_stage1"
        
    elif epoch < freeze_epoch + 6:
        # 阶段2: 解冻layer3+layer4+TransMamba
        for name, param in model.named_parameters():
            if any(x in name for x in ['layer3', 'layer4', 'transmamba']):
                param.requires_grad = True
            elif 'resnet' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        return "partial_stage2"
        
    else:
        # 阶段3: 完全解冻
        for param in model.parameters():
            param.requires_grad = True
        return "fully_unfrozen"

def monitor_gradients(model, epoch, log_dir):
    """监控梯度范数变化"""
    backbone_grad_norm = 0
    decoder_grad_norm = 0
    transmamba_grad_norm = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item() ** 2
            if 'transmamba' in name.lower():
                transmamba_grad_norm += grad_norm
            elif 'resnet' in name.lower():
                backbone_grad_norm += grad_norm
            else:
                decoder_grad_norm += grad_norm
    
    backbone_grad_norm = backbone_grad_norm ** 0.5
    decoder_grad_norm = decoder_grad_norm ** 0.5
    transmamba_grad_norm = transmamba_grad_norm ** 0.5
    
    # 保存梯度信息到文件
    grad_log_path = os.path.join(log_dir, "gradient_norms.txt")
    with open(grad_log_path, 'a') as f:
        f.write(f"Epoch {epoch}: Backbone={backbone_grad_norm:.6f}, "
                f"Decoder={decoder_grad_norm:.6f}, TransMamba={transmamba_grad_norm:.6f}\n")
    
    if epoch % 5 == 0:  # 每5轮打印一次
        print(f"📊 Epoch {epoch} 梯度范数 - Backbone: {backbone_grad_norm:.6f}, "
              f"Decoder: {decoder_grad_norm:.6f}, TransMamba: {transmamba_grad_norm:.6f}")

'''
训练自己的语义分割模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为png图片，无需固定大小，传入训练前会自动进行resize。
   由于许多同学的数据集是网络上下载的，标签格式并不符合，需要再度处理。一定要注意！标签的每个像素点的值就是这个像素点所属的种类。
   网上常见的数据集总共对输入图片分两类，背景的像素点值为0，目标的像素点值为255。这样的数据集可以正常运行但是预测是没有效果的！
   需要改成，背景的像素点值为0，目标的像素点值为1。
   如果格式有误，参考：https://github.com/bubbliiiing/segmentation-format-fix

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中
   
3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
if __name__ == "__main__":
    #---------------------------------#
    #   训练前检查是否已完成训练
    #---------------------------------#
    import glob
    logs_dir = 'logs'
    if os.path.exists(logs_dir):
        loss_dirs = glob.glob(os.path.join(logs_dir, "loss_*"))
        if loss_dirs:
            latest_log = max(loss_dirs, key=os.path.getctime)
            
            # 检查训练完成标志
            completion_flag = os.path.join(latest_log, "TRAINING_COMPLETED")
            early_stop_flag = os.path.join(latest_log, "EARLY_STOPPED")
            
            if os.path.exists(completion_flag) or os.path.exists(early_stop_flag):
                print("=" * 80)
                print("🛑 检测到训练已完成！")
                print(f"📁 训练日志目录: {latest_log}")
                if os.path.exists(completion_flag):
                    print("✅ 状态: 正常完成")
                    with open(completion_flag, 'r', encoding='utf-8') as f:
                        print(f.read())
                else:
                    print("⚠️ 状态: 早停完成")
                    with open(early_stop_flag, 'r', encoding='utf-8') as f:
                        print(f.read())
               
                
                print("=" * 80)
                exit(0)
    
    #---------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    #---------------------------------#
    Cuda = True
    #----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False

    #-----------------------------------------------------#
    #   主干网络选择
    #   vgg
    #   resnet50
    #-----------------------------------------------------#
    backbone    = "resnet50"

    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #                   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #                   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #                   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "model_data/unet_resnet_voc.pth"

    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #   
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 1e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 1e-4。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合语义分割，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在120-300之间调整。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       由于resnet50中有BatchNormalization层
    #       当主干为resnet50的时候batch_size不可为1
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch
    #                       (当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size
    #                       (当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 10  
    Freeze_batch_size   = 4                      
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #-----------------------------------------------------#
    #   input_shape     输入图片的大小，32的倍数
    #-----------------------------------------------------#
    input_shape = [512, 512]
    #-----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    #-----------------------------------------------------#
    num_classes = 20
    # 12 21 4   20
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    use_caa_hsfpn =    True      # 控制CAA_HSFPN空间坐标注意力模块
    use_c2f_iel =       True      # 控制C2f_IEL特征表达增强模块
    use_transmamba =   True     # 控制TransMamba特征表达增强模块
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
    #   数据集路径
    #------------------------------#
    VOCdevkit_path  = ''
    #------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #------------------------------------------------------------------#
    dice_loss       = False
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #------------------------------------------------------------------#
    focal_loss      = False
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers     = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone, 
             use_caa_hsfpn=use_caa_hsfpn, use_c2f_iel=use_c2f_iel,use_transmamba=use_transmamba).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        
        # 🔥 添加解冻监控变量
        unfreeze_status_last = None
        
        # 添加早停和最佳权重保存的变量初始化
        if local_rank == 0:
            best_miou = 0
            best_epoch = 0
            patience_counter = 0
            early_stopping_patience = 100 
            best_weights_path = os.path.join(log_dir, "best_epoch_weights.pth")
    
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   🔥 修改后的解冻逻辑
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                print(f"\n🔓 解冻时刻到达! Epoch {epoch}")
                print(f"📊 冻结期训练完成，开始解冻阶段")
                
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   计算学习率（但不立即使用标准优化器）
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                #---------------------------------------#
                #   🔥 关键修改：使用差异化学习率优化器
                #---------------------------------------#
                print("🚀 创建解冻专用优化器...")
                optimizer = create_unfreeze_optimizer(
                    model=model_train.module if hasattr(model_train, 'module') else model_train,
                    base_lr=Init_lr_fit,
                    optimizer_type=optimizer_type
                )
                
                #---------------------------------------#
                #   重新创建学习率调度器
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                #---------------------------------------#
                #   🔥 渐进式解冻而非一次性解冻
                #---------------------------------------#
                unfreeze_status = gradual_unfreeze_strategy(model, epoch, Freeze_Epoch)
                print(f"🔓 解冻状态: {unfreeze_status}")
                unfreeze_status_last = unfreeze_status
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True
            
            #---------------------------------------#
            #   🔥 解冻后继续渐进式解冻
            #---------------------------------------#
            elif epoch >= Freeze_Epoch and UnFreeze_flag and Freeze_Train:
                unfreeze_status = gradual_unfreeze_strategy(model, epoch, Freeze_Epoch)
                if unfreeze_status != unfreeze_status_last:
                    print(f"🔄 解冻状态更新: {unfreeze_status}")
                    unfreeze_status_last = unfreeze_status

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            train_loss, val_loss = fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            # 🔥 监控梯度变化（解冻期间）
            if local_rank == 0 and epoch >= Freeze_Epoch:
                monitor_gradients(model_train.module if hasattr(model_train, 'module') else model_train, epoch, log_dir)

            # 早停和最佳权重保存逻辑
            if local_rank == 0:
                current_miou = eval_callback.get_miou() if eval_callback else 0
                if current_miou > best_miou:
                    best_miou = current_miou
                    best_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), best_weights_path)
                    print(f"🎉 保存最佳权重! Epoch {best_epoch}, mIoU: {best_miou:.2f}%")
                else:
                    patience_counter += 1
                    if current_miou > 0:
                        print(f"⚠️ mIoU未提升 ({current_miou:.2f}% vs {best_miou:.2f}%), 耐心计数: {patience_counter}/{early_stopping_patience}")
                
                # 🔥 解冻期间不要过早停止，给模型更多时间适应
                min_unfreeze_epochs = 10
                actual_unfreeze_epochs = epoch - Freeze_Epoch if epoch >= Freeze_Epoch else 0
                
                if (patience_counter >= early_stopping_patience and current_miou > 0 and 
                    actual_unfreeze_epochs >= min_unfreeze_epochs):
                    print(f"🛑 触发早停! 连续{early_stopping_patience}轮mIoU未提升 (解冻{actual_unfreeze_epochs}轮后)")
                    
                    # 创建早停标志文件
                    early_stop_flag = os.path.join(log_dir, "EARLY_STOPPED")
                    with open(early_stop_flag, 'w', encoding='utf-8') as f:
                        f.write(f"训练提前停止于 Epoch {epoch + 1}\n")
                        f.write(f"触发时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"最佳mIoU: {best_miou:.2f}% (Epoch {best_epoch})\n")
                        f.write(f"早停原因: 连续{early_stopping_patience}轮mIoU未提升\n")
                    
                    break

            if distributed:
                dist.barrier()

    # 训练完成后的总结和评估 - 整合版本
    if local_rank == 0:
        print("\n" + "="*80)
        
        # 检查是否是早停结束
        early_stop_flag = os.path.join(log_dir, "EARLY_STOPPED")
        is_early_stopped = os.path.exists(early_stop_flag)
        
        if is_early_stopped:
            print("🛑 训练提前结束（早停）!")
        else:
            print("🎉 训练正常完成!")
            
            # 创建正常完成标志文件
            completion_flag = os.path.join(log_dir, "TRAINING_COMPLETED")
            with open(completion_flag, 'w', encoding='utf-8') as f:
                f.write(f"训练正常完成于 Epoch {epoch + 1}\n")
                f.write(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"最佳mIoU: {best_miou:.2f}% (Epoch {best_epoch})\n")
                f.write(f"总训练轮数: {UnFreeze_Epoch}\n")
        
        print(f"🏆 最佳mIoU: {best_miou:.2f}% (Epoch {best_epoch})")
        print(f"📁 最佳权重: {best_weights_path}")
        
        # 🔥 开始最终性能评估
        print("\n" + "="*60)
        print("🎯 开始最终性能评估...")
        
        # 导入评估模块
        try:
            from utils.metrics_calculator import ComprehensiveMetricsCalculator
            
            # 加载最佳权重
            if os.path.exists(best_weights_path):
                print(f"📥 加载最佳权重: {best_weights_path}")
                model.load_state_dict(torch.load(best_weights_path, map_location=device))
                print(f"✅ 已加载第 {best_epoch} 轮的最佳权重 (mIoU: {best_miou:.2f}%)")
            else:
                print("⚠️ 未找到最佳权重文件，使用当前权重进行评估")
            
            # 创建评估器
            metrics_calculator = ComprehensiveMetricsCalculator(
                model=model,
                device=device,
                input_shape=input_shape,
                num_classes=num_classes
            )
            
            # 计算所有指标
            final_metrics = metrics_calculator.calculate_all_metrics(
                dataloader=gen_val,
                max_samples=500  # 限制验证样本数量，可根据需要调整
            )
            
            # 打印最终结果
            print("\n" + "🏆" + "="*50 + "🏆")
            print("🎊 最终性能评估结果:") 
            print("="*52)
            print(f"🚀 FPS (推理速度):        {final_metrics['fps']:.2f} frames/sec")
            print(f"⏱️ 平均推理时间:          {final_metrics['avg_inference_time_ms']:.2f} ms")
            print(f"🎯 aACC (整体像素准确率): {final_metrics['aacc']:.2f}%")
            print(f"📈 mACC (平均类别准确率): {final_metrics['macc']:.2f}%")
            print(f"🎯 mIoU (最佳):          {best_miou:.2f}% (Epoch {best_epoch})")
            print("="*52)
            
            # 保存评估报告
            metrics_calculator.save_metrics_report(
                metrics=final_metrics,
                save_dir=log_dir,
                model_name="Enhanced UNet with CAA_HSFPN + C2f_IEL"
            )
            
            # 标记是否成功评估
            evaluation_success = True
            
        except ImportError:
            print("❌ 无法导入评估模块，跳过FPS、aACC、mACC计算")
            print("💡 请确保创建了 utils/metrics_calculator.py 文件")
            final_metrics = {
                'fps': 0.0,
                'avg_inference_time_ms': 0.0,
                'aacc': 0.0,
                'macc': 0.0
            }
            evaluation_success = False
        except Exception as e:
            print(f"❌ 评估过程中出现错误: {str(e)}")
            print("⚠️ 跳过详细评估，仅保存基本训练信息")
            final_metrics = {
                'fps': 0.0,
                'avg_inference_time_ms': 0.0,
                'aacc': 0.0,
                'macc': 0.0
            }
            evaluation_success = False
        
        # 🔥 保存完整的训练总结信息（只执行一次）
        summary_path = os.path.join(log_dir, "training_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("训练总结报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"训练完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"训练状态: {'提前停止（早停）' if is_early_stopped else '正常完成'}\n")
            f.write(f"模型架构: {backbone}\n")
            f.write(f"输入尺寸: {input_shape}\n")
            f.write(f"类别数量: {num_classes}\n")
            f.write(f"冻结epoch: {Freeze_Epoch}\n")
            f.write(f"总训练轮数: {UnFreeze_Epoch}\n")
            f.write(f"实际训练轮数: {epoch + 1}\n\n")
            
            f.write("最佳结果:\n")
            f.write(f"最佳mIoU: {best_miou:.2f}%\n")
            f.write(f"最佳Epoch: {best_epoch}\n")
            f.write(f"最佳权重: {best_weights_path}\n\n")
            
            f.write("训练配置:\n")
            f.write(f"差异化学习率: 启用\n")
            f.write(f"渐进式解冻: 启用\n")
            f.write(f"梯度监控: 启用\n\n")
            
            # 添加评估结果
            if evaluation_success:
                f.write("最终性能评估结果:\n")
                f.write("="*30 + "\n")
                f.write(f"FPS (推理速度): {final_metrics['fps']:.2f} frames/sec\n")
                f.write(f"平均推理时间: {final_metrics['avg_inference_time_ms']:.2f} ms\n")
                f.write(f"aACC (整体像素准确率): {final_metrics['aacc']:.2f}%\n")
                f.write(f"mACC (平均类别准确率): {final_metrics['macc']:.2f}%\n")
                f.write(f"处理样本数: {final_metrics.get('processed_samples', 0)}\n")
                f.write(f"总像素数: {final_metrics.get('total_pixels', 0):,}\n")
            else:
                f.write("评估结果:\n")
                f.write("="*30 + "\n")
                f.write("详细评估未能完成，请检查评估模块\n")
        
        print(f"📁 完整训练总结保存到: {summary_path}")
        
        if is_early_stopped:
            print("⚠️ 训练因早停而结束")
        else:
            print("✅ 训练正常完成")
        
        if evaluation_success:
            print("📊 性能评估完成")
        
        print("💡 下次运行将被阻止，如需重新训练请删除标志文件")
        print("🎉 所有任务完成!")

    # 关闭tensorboard writer
    loss_history.writer.close()
