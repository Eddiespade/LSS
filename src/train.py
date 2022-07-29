"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info


def train(version,
            dataroot='/data/nuscenes',      # 数据集根路径
            nepochs=10000,                  # epochs次数
            gpuid=0,                        # gpu的id

            H=900, W=1600,                  # 原始的图像尺寸
            resize_lim=(0.193, 0.225),      # 数据增强随机缩小图像的比率
            final_dim=(128, 352),           # 最终样本的尺寸 H W
            bot_pct_lim=(0.0, 0.22),        # 最终图像块的区域
            rot_lim=(-5.4, 5.4),            # 图像旋转角度的取值范围
            rand_flip=True,                 # 是否选择随机翻转增强
            ncams=5,                        # 每次训练选择相机数量
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',                # 日志/权重文件保存路径

            xbound=[-50.0, 50.0, 0.5],      #
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=1,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    # ---------------------------------------------------- 加载数据集 ----------------------------------------------------
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    # --------------------------------------------------- 加载网络模型 ---------------------------------------------------
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)
    # ---------------------------------------------------- 加载优化器 ----------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)               # 用于tensorboard
    val_step = 1000 if version == 'mini' else 10000

    # ----------------------------------------------------- 开始训练 -----------------------------------------------------
    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed(1117)    # 设置随机种子
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            # ----------------------------------------------------------------------------------------------
            # imgs：             [bsz, 5, 3, 128, 352]
            # rots：             [bsz, 5, 3, 3]
            # trans：            [bsz, 5, 3]
            # intrins：          [bsz, 5, 3, 3]
            # post_rots：        [bsz, 5, 3, 3]
            # post_trans：       [bsz, 5, 3]
            # binimgs：          [bsz, 1, 200, 200]
            # ----------------------------------------------------------------------------------------------
            t0 = time()
            opt.zero_grad()
            # preds [bsz, 1, 200, 200]
            preds = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
            # binimgs [bsz, 1, 200, 200]
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            # -------------------------------------------- 打印输出及保存模型 ---------------------------------------------
            if counter % 10 == 0:
                print(counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/iou', val_info['iou'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()
