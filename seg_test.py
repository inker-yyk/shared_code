import torch
import torch.nn.functional as F

# from models.diffusion import Model
# from models.ema import EMAHelper
from omegaconf import OmegaConf
from PIL import Image
from data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import cv2
import sys
sys.path.append("/hdd2/yyk/DiffAD-main_2")
from rec_network.util import instantiate_from_config
from rec_network.models.diffusion.ddim import DDIMSampler

obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood']

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap, txt_path, obj_name):

    fin_str = "--------------------------\n"
    fin_str += obj_name + "\n" + "img_auc," + run_name

    fin_str += "," + str(np.round(image_auc, 3))
    fin_str += "\n"
    fin_str += "pixel_auc," + run_name

    fin_str += "," + str(np.round(pixel_auc, 3))
    fin_str += "\n"
    fin_str += "img_ap," + run_name

    fin_str += "," + str(np.round(image_ap, 3))
    fin_str += "\n"
    fin_str += "pixel_ap," + run_name

    fin_str += "," + str(np.round(pixel_ap, 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open(txt_path, 'a+') as file:
        file.write(fin_str)


def test(obj_name, mvtec_path, checkpoint_path, base_model_name, ldm_path, txt_path, output_path):

    img_dim = 256
    run_name = base_model_name + "_" + obj_name + '_'

    config = OmegaConf.load("/hdd2/yyk/DiffAD-main_2/configs/mvtec.yaml")
    config.model.params["ckpt_path"] = ldm_path
    model = instantiate_from_config(config.model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    model_seg = DiscriminativeSubNetwork(in_channels=9, out_channels=2)
    temp_name = os.listdir(checkpoint_path)
    temp_path= os.path.join(checkpoint_path, temp_name[0])
    state_dict = torch.load(temp_path, map_location='cpu')
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 移除 'module.' 前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    model_seg.load_state_dict(new_state_dict)    
    model_seg.cuda()
    model_seg.eval()
    dataset = MVTecDRAEMTestDataset(mvtec_path, obj_name, resize_shape=[img_dim, img_dim])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    cnt_display = 0

    for i_batch, sample_batched in enumerate(dataloader):
        gray_batch = sample_batched["image"].cuda()

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        
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

        shape = c.shape[1:]
        samples_ddim, _ = sampler.sample(S=10,
                                         conditioning=cond, # or conditioning=c_noisy
                                         batch_size=c.shape[0],
                                         shape=shape,
                                         verbose=False)

        gray_rec = model.decode_first_stage(samples_ddim)

        samples_ddim1 = 0.5 * samples_ddim + 0.5 * c
        gray_rec1 = model.decode_first_stage(samples_ddim1)

        joined_in = torch.cat((gray_rec.detach(), gray_rec1.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        t_mask = out_mask_sm[:, 1:, :, :]

        os.makedirs(os.path.join(output_path, obj_name), exist_ok=True)
        outpath = os.path.join(output_path, obj_name, 'rec_images' + str(cnt_display) + '.jpg')
        sample = gray_rec.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        cv2.imwrite(outpath, sample)

        outpath = os.path.join(output_path, obj_name, 'gt_images' + str(cnt_display) + '.jpg')
        sample = gray_batch.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        cv2.imwrite(outpath, sample)

        outpath = os.path.join(output_path, obj_name, 'out_masks' + str(cnt_display) + '.jpg')
        sample = t_mask[0].detach().cpu().numpy()[0] * 255
        cv2.imwrite(outpath, sample)

        outpath = os.path.join(output_path, obj_name, 'in_masks' + str(cnt_display) + '.jpg')
        sample = true_mask[0].detach().cpu().numpy()[0] * 255
        cv2.imwrite(outpath, sample)

        heatmap = t_mask[0].detach().cpu().numpy()[0]
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        show = heatmap * 0.5 + gray_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
        outpath = os.path.join(output_path, obj_name, 'heatmap' + str(cnt_display) + '.jpg')
        cv2.imwrite(outpath, show)
        cnt_display += 1

        out_mask_cv = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 21, stride=1,
                                                           padding=21 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction) 
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

    print(obj_name)
    print("AUC Image:  " + str(auroc))
    print("AP Image:  " + str(ap))
    print("AUC Pixel:  " + str(auroc_pixel))
    print("AP Pixel:  " + str(ap_pixel))

    write_results_to_file(run_name, auroc, auroc_pixel, ap, ap_pixel, txt_path, obj_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--ema', action='store_true')

    parser.add_argument("--logit_transform", default=False)
    parser.add_argument("--uniform_dequantization", default=False)
    parser.add_argument("--gaussian_dequantization", default=False)
    parser.add_argument("--random_flip", default=True)
    parser.add_argument("--rescaled", default=True)
    parser.add_argument("--sample_type", type=str, default="generalized",
                        help="sampling approach (generalized or ddpm_noisy)")
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)")
    parser.add_argument("--ldm_ckpt_path", type=str, default="")
    parser.add_argument("--txt_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    
    with torch.cuda.device(args.gpu_id):
        for obj_name in obj_list:
            test(obj_name, args.data_path, args.checkpoint_path, args.base_model_name, args.ldm_ckpt_path, args.txt_path, args.output_path)
