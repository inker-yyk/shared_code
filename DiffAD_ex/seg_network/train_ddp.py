import torch
# from models.diffusion import Model
# from models.ema import EMAHelper
from omegaconf import OmegaConf
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import sys
sys.path.append("/hdd2/yyk/DiffAD-main_4")
from rec_network.util import instantiate_from_config
from rec_network.models.diffusion.ddim import DDIMSampler
import cv2
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):
    
    # ddp 
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        ddp = True
    else:
        ddp = False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ddp数据处理
    dataset = MVTecDRAEMTrainDataset(args.data_path, args.anomaly_source_path,
                                     resize_shape=[256, 256], k_shot=4)
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler)
    else:
         dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    run_name = 'DRAEM_test_' + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs) + '_'

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name + "/"))

    config = OmegaConf.load("/hdd2/yyk/DiffAD-main_4/configs/mvtec.yaml")

    config.model.params["ckpt_path"] = args.ckpt_load_path
    model = instantiate_from_config(config.model)

    model = model.to(device)
    sampler = DDIMSampler(model)

    model_seg = DiscriminativeSubNetwork(in_channels=9, out_channels=2)
    model_seg = model_seg.to(device)
    model_seg.apply(weights_init)

    # 使用ddp封装模型，同时对学习率进行了放大。因为使用学习率*梯度更新参数。每张卡的模型参数是一致的，由于多卡训练导致有效批次增大。比如之前是单卡学习率0.01，反向传播的时候使用0.01*一个批次的梯度进行参数更新。现在是多卡，所以是0.01*gpu数目*gpu数目的批次的梯度更新参数
    if ddp:
        model_seg = DDP(model_seg, device_ids=[args.local_rank])
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1
    
    optimizer = torch.optim.Adam([{"params": model_seg.parameters(), "lr": args.lr*gpu_num}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.8, args.epochs * 0.9], gamma=0.2, last_epoch=-1)

    loss_focal = FocalLoss()

    

    n_iter = 0
    
    start_time = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        end_time = time.time()
        print("Epoch: " + str(epoch) + " Learning rate: " + str(lr), "  ", )
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
        
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].to(device)
            aug_gray_batch = sample_batched["augmented_image"].to(device)
            anomaly_mask = sample_batched["anomaly_mask"].to(device)
            cond_all = sample_batched["cond"]
            cond_all["c_concat"] = cond_all["c_concat"].cuda(device)

            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning(cond_all["c_crossattn"])]
            cond["c_concat"] =  [model.encode_first_stage((cond_all["c_concat"])).mode()]
            c = cond["c_concat"][0]
            shape = c.shape[1:]
            noise = torch.randn_like(c)
            t = torch.randint(400, 500, (c.shape[0],), device=device).long()
            c_noisy = model.q_sample(x_start=c, t=t, noise=noise)
            cond["c_concat"][0] = c_noisy
            
            samples_ddim, _ = sampler.sample(S=50,
                                             conditioning=cond, # or conditioning=c_noisy
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            gray_rec = model.decode_first_stage(samples_ddim)

            samples_ddim1 = 0.5 * samples_ddim + 0.5 * c
            gray_rec1 = model.decode_first_stage(samples_ddim1)

            joined_in = torch.cat((gray_rec, gray_rec1, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if not ddp or (ddp and dist.get_rank() == 0):
                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                    
                sample_path = os.path.join(args.log_path, "samples")
                os.makedirs(sample_path, exist_ok=True)
                if args.visualize and n_iter % 100 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]

                    outpath = os.path.join(sample_path, 'batch_augmented' + str(n_iter) + '.jpg')
                    sample = aug_gray_batch.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(outpath, sample)

                    outpath = os.path.join(sample_path, 'batch_recon_target' + str(n_iter) + '.jpg')
                    sample = gray_batch.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(outpath, sample)

                    outpath = os.path.join(sample_path, 'batch_recon_out' + str(n_iter) + '.jpg')
                    sample = gray_rec.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(outpath, sample)

                    outpath = os.path.join(sample_path, 'batch_recon_inter' + str(n_iter) + '.jpg')
                    sample = gray_rec1.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    cv2.imwrite(outpath, sample)

                    outpath = os.path.join(sample_path, 'mask_target' + str(n_iter) + '.jpg')
                    sample = anomaly_mask[0].detach().cpu().numpy()[0] * 255
                    cv2.imwrite(outpath, sample)

                    outpath = os.path.join(sample_path, 'mask_out' + str(n_iter) + '.jpg')
                    sample = t_mask[0].detach().cpu().numpy()[0] * 255
                    cv2.imwrite(outpath, sample)

            n_iter += 1

        scheduler.step()

        if not ddp or (ddp and dist.get_rank() == 0):
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name + "seg.pckl"))
        
        start_time = time.time()
        
        # if epoch % 100 == 0 :
        #     torch.save(model_seg.state_dict(),
        #                os.path.join(args.checkpoint_path, run_name + str(epoch) + "_seg.pckl"))


if __name__ == "__main__":
    import argparse

    # 设置local_rank,默认设置成-1。因为如果使用torch.distributed.launch启动ddp，则local_rank自动为每张卡分配。从0开始。要记住每张卡的程序的区别就是local_rank不同。因为我们要使用主进程测试和保存模型
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank',type=int,default=-1)

    # parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    # parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False) # 仅单卡
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true',default=True)

    parser.add_argument("--type", type=str, default="simple")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=3)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", type=list, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn_resolutions", type=list, default=[16, ])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--var_type", type=str, default="fixedsmall")
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--resamp_with_conv', action='store_true')
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument('--ddim_log_path', action='store', type=str, default="../ddim-main/mvtec/logs")
    parser.add_argument("--timesteps", type=int, default=10, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0)

    parser.add_argument("--logit_transform", default=False)
    parser.add_argument("--uniform_dequantization", default=False)
    parser.add_argument("--gaussian_dequantization", default=False)
    parser.add_argument("--random_flip", default=True)
    parser.add_argument("--rescaled", default=True)
    parser.add_argument("--sample_type", type=str, default="generalized",
                        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    parser.add_argument("--ckpt_load_path", type=str, default="")

    args = parser.parse_args()

    train_on_device(args)

