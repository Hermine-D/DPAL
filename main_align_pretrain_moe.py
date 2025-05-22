# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
import sys
from pathlib import Path
from typing import Iterable
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from PIL import Image
import matplotlib.pyplot as plt
import random
import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_pretrain
import vision_transformer as vits
import util.lr_sched as lr_sched


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--height', default=224, type=int, help="""Height of image""")
    parser.add_argument('--width', default=224, type=int, help="""Width of image""")

    # Model parameters
    parser.add_argument('--model', default='kd_csl_vit_tiny_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--teacher_model', default='expert_vit_base', type=str, metavar='MODEL',
                        help='Name of teacher model')
    
    parser.add_argument('--teacher_pretrained', default='',
                        help='teacher model from checkpoint')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--csm_out_dim', default=65536, type=int, help="""Dimensionality of
        the CSM head output. """)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) 
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""") 
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int, 
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.8, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--crop_height', default=96, type=int, help="""Height of crop image""")
    parser.add_argument('--crop_width', default=96, type=int, help="""Width of crop image""")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    # parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--debug', action='store_true')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    if args.debug and misc.get_rank() == 0:
        import debugpy
        debugpy.listen(("localhost", 1234))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print("Debugger is attached.")

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print("distrubtion: ",args.distributed)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transform_train = DataAugmentation((args.height,args.width),
        (args.crop_height,args.crop_width),
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    teacher = vits.__dict__[args.teacher_model](pretrained=args.teacher_pretrained)
    student = models_pretrain.__dict__[args.model](img_size=(args.height, args.width))
    ema_teacher = models_pretrain.__dict__[args.model](img_size=(args.height, args.width))

    teacher.to(device)
    student.to(device)
    ema_teacher.to(device)
    from util.count_param import count_parameters
    print("#"*50, "student param", "#"*50)
    count_parameters(student.backbone)
    count_parameters(student)

    ema_teacher_without_ddp = ema_teacher
    model_without_ddp = student
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = student.module
    ema_teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False 
    for p in ema_teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: student is {args.model} network and teacher is {args.teacher_model} network.")

    dpal_kd_loss = DPALKDLoss().cuda()

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    
    loss_scaler = NativeScaler()
    momentum_schedule = cosine_scheduler(0.996, 1, args.epochs, len(data_loader_train))
    to_restore = {"epoch": 0}
    misc.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=ema_teacher,
        optimizer=optimizer
    )
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            student, teacher, ema_teacher, ema_teacher_without_ddp, dpal_kd_loss, momentum_schedule,
            data_loader_train, optimizer, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        save_dict = {
                'student': student.state_dict(),
                'teacher': ema_teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args
        }
        misc.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            
            misc.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, ema_teacher, ema_teacher_without_ddp, dpal_kd_loss, momentum_schedule,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    student.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = [sms.cuda(non_blocking=True) for sms in samples]
        meta = {}
        if args.local_crops_number>0:
            meta['region_imgs'] = samples[1:1+args.local_crops_number]
        else:
            meta['region_imgs'] = None
        meta['crowd_imgs'] = samples[-1]

        with torch.cuda.amp.autocast():
            pred_t = teacher.forward(samples[0], meta)
            pred_s, moe_loss, experts_loss = student(samples[0], meta)
            m_loss = dpal_kd_loss(pred_s['aligned_global_feats'], pred_s['aligned_local_feats'], pred_s['relations'], pred_t['feats_from_teacher_global'], pred_t['feats_from_teacher_local'], pred_t['relations'])
            if epoch < 20:
                m_loss['moe_loss'] = 0.1 * moe_loss # 0.1
            else:
                m_loss['moe_loss'] = 0.01 * moe_loss
            m_loss['experts_loss'] = experts_loss
            loss = m_loss['align_global_loss'] + m_loss['align_relation_loss'] + m_loss['align_local_loss'] + m_loss['moe_loss'] + m_loss['experts_loss']
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            s = ""
            for key, value in m_loss.items():
                s += key + ": " + str(value.item()) + ", "
                print(s)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=student.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        with torch.no_grad():
            m = momentum_schedule[data_iter_step]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), ema_teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value, align_patch_loss=m_loss['align_local_loss'].item(), align_rep_loss=m_loss['align_global_loss'].item(), align_att_loss=m_loss['align_relation_loss'].item(), moe_loss=m_loss['moe_loss'].item(), experts_loss=m_loss['experts_loss'].item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class DataAugmentation(object):
    def __init__(self, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number, ref_size=(256, 256)):
        ref_oimage_path='./data/pretrain_dataset/train/others'
        self.list_ref_oimg_files = []
        for file in os.listdir(ref_oimage_path):
            file_path = os.path.join(ref_oimage_path, file)
            self.list_ref_oimg_files.append(file_path)
        self.num_ref_obj = len(self.list_ref_oimg_files)

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        ratio = (0.4,0.6)
        if size == (224,224):
            ratio = (0.75, 1.3333333333333333)
        elif size == (256,192):
            ratio = (0.4,0.6)
        elif size == (256,128):
            ratio = (0.4,0.6)
        elif size == (384,128):
            ratio = (0.25,0.4)
        print(global_crops_scale, size, ratio)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),  # 3 is bicubic
            # transforms.RandomHorizontalFlip(),
            flip_and_color_jitter,
            misc.GaussianBlur(1.0),
            normalize]
        )
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=global_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])
        self.global_transfo3 = transforms.Compose([
            transforms.RandomResizedCrop(size=ref_size, scale=(0.2, 1.0), interpolation=3, ratio=(0.75, 1.3333333333333333)),
            flip_and_color_jitter,
            misc.GaussianBlur(0.1),
            misc.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=local_crops_scale, interpolation=3, ratio=ratio),
            flip_and_color_jitter,
            misc.GaussianBlur(p=0.5),
            normalize,
        ])
        
        
        assert crop_size[0] < ref_size[0] and crop_size[1] < ref_size[1]
        self.end_points = (ref_size[0]-crop_size[0]-1, ref_size[1]-crop_size[1]-1)
        self.region_size = crop_size

    def __call__(self, image):

        multi_scales = []
        aug_img_1 = self.global_transform(image)
        multi_scales.append(aug_img_1)

        for _ in range(self.local_crops_number):
            multi_scales.append(self.local_transfo(image))
        region_img = torch.nn.functional.interpolate(aug_img_1.unsqueeze(0), size=self.region_size).squeeze(0)
        random.seed(time.time())
        idx_obj = random.randint(0,self.num_ref_obj-1)
        obj_img = Image.open(self.list_ref_oimg_files[idx_obj]).convert('RGB')
        obj_img = self.global_transfo3(obj_img)

        start_point_h = int(np.random.choice(np.arange(0, self.end_points[0], 1), 1))
        start_point_w = int(np.random.choice(np.arange(0, self.end_points[1] - self.region_size[1], 1), 1))
        end_point_h = start_point_h + self.region_size[0]
        end_point_w = start_point_w + self.region_size[1]
        obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w] = 0.7*multi_scales[-1] + 0.3*obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w]

        start_point_h = int(np.random.choice(np.arange(0, self.end_points[0], 1), 1))
        start_point_w = int(np.random.choice(np.arange(end_point_w, self.end_points[1], 1), 1))
        end_point_h = start_point_h + self.region_size[0]
        end_point_w = start_point_w + self.region_size[1]
        obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w] = 0.7*region_img + 0.3*obj_img[:, start_point_h:end_point_h, start_point_w:end_point_w]
 

        multi_scales.append(obj_img)

        return multi_scales

class DPALKDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_feats_global, s_feats_local, s_relations, t_feats_global, t_feats_local, t_relations):
        rep_sim_global_loss = torch.Tensor([0]).cuda()
        rep_sim_local_loss = torch.Tensor([0]).cuda()
        n_rep_global_loss_terms = 0
        n_rep_local_loss_terms = 0
        relation_sim_loss = torch.Tensor([0]).cuda()
        n_relation_loss_terms = 0
        # global
        for t_feat in t_feats_global: # 1 single
            for s_feat in s_feats_global: # n+1
                loss = nn.MSELoss(reduction="none")(s_feat, t_feat).mean(-1).mean()
                # if not math.isfinite(loss.item()):
                #     loss = 0. * s_feat.mean()
                rep_sim_global_loss += loss
                n_rep_global_loss_terms += 1
        # local + relation
        res = [(256,128),(256,256)]
        for iq in range(len(t_feats_local)): # 1 single + 1 multi
            if s_relations and t_relations:
                s_qk_relation, s_vv_relation = s_relations[iq]
                t_qk_relation, t_vv_relation = t_relations[iq]
                i_s_qk_relation = s_qk_relation.log()
                i_s_vv_relation = s_vv_relation.log()
                qk_loss = nn.KLDivLoss(reduction="none")(i_s_qk_relation, t_qk_relation).sum(-1).mean()
                vv_loss = nn.KLDivLoss(reduction="none")(i_s_vv_relation, t_vv_relation).sum(-1).mean()

                relation_sim_loss += (qk_loss + vv_loss)
                n_relation_loss_terms += 1
            if s_feats_local[iq].shape != t_feats_local[iq].shape:
                B, _, L = s_feats_local[iq].shape
                new_ph, new_pw = res[iq][0] // 16, res[iq][1] // 16
                ph, pw = res[iq][0] // 32, res[iq][1] // 32
                s_feats_patch[iq] = nn.functional.interpolate(
                    s_feats_patch[iq].reshape(B, ph, pw, L).permute(0, 3, 1, 2), # B, L, ph, pw
                    mode="bicubic",
                    size=(new_ph, new_pw)
                ).reshape(B, L, new_ph*new_pw).permute(0, 2, 1)
            loss = nn.MSELoss(reduction="none")(s_feats_local[iq], t_feats_local[iq]).mean(-1).mean()

            rep_sim_local_loss += loss
            n_rep_local_loss_terms += 1
                    
        rep_sim_global_loss /= n_rep_global_loss_terms
        rep_sim_local_loss /= n_rep_local_loss_terms
        if n_relation_loss_terms:
            relation_sim_loss /= n_relation_loss_terms
        
        return {'align_global_loss':rep_sim_global_loss, 'align_local_loss':rep_sim_local_loss, 'align_relation_loss':relation_sim_loss}
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
