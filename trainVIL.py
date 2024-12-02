import torch
import torch.optim as optim
import torch.utils.data as data
import torch.cuda.amp as amp
scaler = amp.GradScaler()
import numpy as np
import os
import os.path as osp
import time
import argparse
import random
from progress.bar import Bar
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from libs.utils.logger import Logger, AverageMeter
from libs.utils.utility import write_mask, save_checkpoint_V2
from libs.utils.optimizer import build_optimizer
from libs.utils.utility import vis_while_train

from libs.models.RouterV4 import RouterWithB
from libs.dataset.dataV3 import DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transformV4 import GenerateLaneLine
from libs.utils.lossV5 import DILaneCriterionV5


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
dist.init_process_group(backend="nccl",
                        init_method='env://')

# from optionsV3 import OPTION as opt
from libs.utils.config import Config
opt = Config.fromfile('./optionsV3.py')

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='0', type=str, help='set gpu id to train the network, split with comma')
    # parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    return parser.parse_args()

def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _init_fn(worker_id):
    np.random.seed(int(3407)+worker_id)

def main():
    start_epoch = 0
    # Fix seeds
    seed_torch()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    # Data
    print('==> Preparing dataset')
    train_transformer = GenerateLaneLine(opt, training=True)
    try:
        if isinstance(opt.trainset, list): 
            datalist = []
            for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip): #['VIL100'] [3] [5]
                ds = DATA_CONTAINER[dataset](
                    train=True,
                    sampled_frames=opt.sampled_frames, #16
                    transform=train_transformer,
                    max_skip=max_skip, #5
                    samples_per_video=opt.samples_per_video
                )
                datalist += [ds] * freq # *3
            trainset = data.ConcatDataset(datalist)
        else:
            max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
            trainset = DATA_CONTAINER[opt.trainset](
                train=True,
                sampled_frames=opt.sampled_frames,
                transform=train_transformer,
                max_skip=max_skip,
                samples_per_video=opt.samples_per_video
            )
    except KeyError as e:
        print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
        print(list(DATA_CONTAINER.keys()))
        exit()

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                    batch_size=opt.train_batch,
                                    sampler=DistributedSampler(trainset, shuffle=True),
                                    pin_memory=True,
                                    num_workers=8,
                                    collate_fn=multibatch_collate_fn,
                                    drop_last=True,
                                    worker_init_fn=_init_fn)
    
    # Model
    print("==> creating model")
    criterion = DILaneCriterionV5(cfg=opt)
    model = RouterWithB(cfg=opt, criterion=criterion)
    #DDP封装前要装载进对应的gpu
    model = model.to(device)
    print('Total params need to train: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # set training parameters 
    # for p in model.backbone.parameters():
    #     p.requires_grad = False

    optimizer = build_optimizer(opt, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainset)*opt.epochs//torch.cuda.device_count())

    minloss = float('inf')

    opt.checkpoint_model = osp.join(osp.join(opt.checkpoint, opt.valset, opt.setting, 'model'))
    if not osp.exists(opt.checkpoint_model):
        os.makedirs(opt.checkpoint_model)

    logger = Logger(os.path.join(opt.checkpoint+'log.txt'), resume=True)

    if opt.initial_model:
        print('==> Init from checkpoint {}'.format(opt.initial_model))
        assert os.path.isfile(opt.initial_model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial_model, map_location='cuda:{}'.format(local_rank))
        # 过滤参数形状发生变化的层
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if (k in model_dict and 'pre_norm' not in k and 'DWNets' not in k)}
        # model.load_state_dict(pretrained_dict, strict=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif opt.resume_model:
        print('==> Resuming from checkpoint {}'.format(opt.resume_model))
        assert os.path.isfile(opt.resume_model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_model, map_location='cuda:{}'.format(local_rank))
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        skips = checkpoint['max_skip']
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                trainloader.dataset.set_max_skip(skips) #skip
        except:
            print('[Warning] Initializing max skip fail')
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) #revcol中没有BN层
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[local_rank],
                                                    #   output_device=local_rank, #什么用？
                                                      broadcast_buffers=False, #好像没用
                                                      find_unused_parameters=True)
    # model._set_static_graph() #._set_static_graph()会自动将find_unused_parameters设置为True
    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    for epoch in range(start_epoch, opt.epochs):
        trainloader.sampler.set_epoch(epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, optimizer.param_groups[0]['lr']))
        model.train()
        train_loss = trainOneEpoch(trainloader,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            device=device)
        # append logger file
        logger.log(epoch + 1, opt.learning_rate, train_loss)
        # adjust max skip 随着epochs的增加 加长sample frames之间的距离
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()
        # save model
        is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
            if isinstance(trainloader.dataset, data.ConcatDataset) \
            else trainloader.dataset.max_skip
        if local_rank == 0: #只保存一个进程中的模型
            if ((epoch + 1) % opt.epoch_per_test == 0) or (is_best):
                save_checkpoint_V2({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'loss': train_loss,
                    'minloss': minloss,
                    'optimizer': optimizer.state_dict(),
                    'max_skip': skips,
                    'scheduler': scheduler.state_dict(),
                }, epoch + 1, is_best, checkpoint=opt.checkpoint_model)
    logger.close()
    print('minimum loss:', minloss)
    
def trainOneEpoch(trainloader, model, optimizer, scheduler, epoch, device):
    data_time = AverageMeter()
    loss = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader): #循环iter
        frames, masks, lanes_lines, objs, infos, flows, occlusions = data #一个batch的数据
        # measure data loading time
        data_time.update(time.time() - end)

        frames = frames.to(device) #[1, B, 3, 320, 640]
        masks = masks.to(device) #[1, B, 9, 320, 640]
        lanes_lines = lanes_lines.to(device) #[1, B, 8, 78]
        objs = objs.to(device) #tensor([8])
        flows = flows.to(device) #[1, B, 320, 640, 2]
        occlusions = occlusions.to(device) #[1, B, 8] #-1无 0未遮挡 1遮挡
        objs[objs == 0] = 1

        N, T, C, H, W = frames.size()

        total_loss = 0.0
        inputs = {}
        for idx in range(N): # N=1 逐clip进行分析
            inputs['frame'] = frames[idx] #[9, 3, 320, 640]
            inputs['mask'] = masks[idx] #[9, 9, 320, 640]
            inputs['lanes'] = lanes_lines[idx] #[9, 8, 78]
            inputs['lane_ids'] = inputs['lanes'][:, :, 1] #[9, 8]
            inputs['gt_flows'] = flows[idx] #FIXME
            inputs['occlusion'] = occlusions[idx] #[B, 8] 车道线阻挡与否 -1:无 0:未遮挡 1:遮挡
            inputs['num_objects'] = objs[idx]
            inputs['info'] = infos[idx]
            # total_loss = model.module(inputs)
            total_loss += model(inputs)
        total_loss /= N * T
        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

        # compute gradient and do SGD step (divided by accumulated steps)
        assert torch.isnan(total_loss).sum() == 0, print("Loss is NAN!!!")
        # total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2) #梯度裁剪
        # optimizer.step()
        scaler.scale(total_loss).backward() #
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        optimizer.zero_grad()
        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.3f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.item() #loss.avg
        )
        print('-'*10 + str(loss.avg))
        bar.next()
    bar.finish()
    return loss.avg

if __name__ == '__main__':
    main()