import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import cv2
import sys
sys.path.append("/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2")
from rec_network.main import instantiate_from_config
from rec_network.models.diffusion.ddim import DDIMSampler
from rec_network.data.mvtec import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader



def combine_three_and_save_images(condition, sample1, sample2, outpath, direction='horizontal'):
    """
    Combine three images and save the result as a single image file.
    
    Parameters:
    - condition: The first image to be combined.
    - sample1: The second image to be combined.
    - sample2: The third image to be combined.
    - outpath: The output path where the combined image will be saved.
    - direction: Direction of combination ('horizontal' or 'vertical').
    """
    # Ensure all images have the same number of channels
    if condition.shape[2] != sample1.shape[2] or condition.shape[2] != sample2.shape[2]:
        # Convert grayscale images to BGR if necessary
        if condition.shape[2] == 1:
            condition = cv2.cvtColor(condition, cv2.COLOR_GRAY2BGR)
        if sample1.shape[2] == 1:
            sample1 = cv2.cvtColor(sample1, cv2.COLOR_GRAY2BGR)
        if sample2.shape[2] == 1:
            sample2 = cv2.cvtColor(sample2, cv2.COLOR_GRAY2BGR)

    # Adjust size if needed
    if direction == 'horizontal':
        # Make sure all images have the same height
        max_height = max(condition.shape[0], sample1.shape[0], sample2.shape[0])
        condition = cv2.resize(condition, (int(condition.shape[1] * max_height / condition.shape[0]), max_height))
        sample1 = cv2.resize(sample1, (int(sample1.shape[1] * max_height / sample1.shape[0]), max_height))
        sample2 = cv2.resize(sample2, (int(sample2.shape[1] * max_height / sample2.shape[0]), max_height))
        
        # Combine horizontally
        combined_image = np.hstack((condition, sample1, sample2))
    elif direction == 'vertical':
        # Make sure all images have the same width
        max_width = max(condition.shape[1], sample1.shape[1], sample2.shape[1])
        condition = cv2.resize(condition, (max_width, int(condition.shape[0] * max_width / condition.shape[1])))
        sample1 = cv2.resize(sample1, (max_width, int(sample1.shape[0] * max_width / sample1.shape[1])))
        sample2 = cv2.resize(sample2, (max_width, int(sample2.shape[0] * max_width / sample2.shape[1])))
        
        # Combine vertically
        combined_image = np.vstack((condition, sample1, sample2))
    else:
        raise ValueError("Invalid direction. Use 'horizontal' or 'vertical'.")

    # Save the combined image
    cv2.imwrite(outpath, combined_image)

# Example usage:
# Assuming `condition`, `sample1`, and `sample2` are already loaded images
# combine_three_and_save_images(condition, sample1, sample2, 'combined_image.jpg', 'horizontal')

# Example usage:
# Assuming `condition` and `sample` are already loaded images
# combine_and_save_images(condition, sample, 'combined_image.jpg', 'horizontal')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="./samples/905_test_full_shot_step50/",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--mvtec_path",
        type=str,
        default="/home/ubuntu/hdd1/yyk/ad_dataset/mvtec_anomaly_detection/",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/configs/mvtec.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/logs/rec/checkpoints/trainstep_checkpoints/epoch=000014-step=000002999.ckpt",
    )
    opt = parser.parse_args()
    ddim_eta = 0.0

    # import pdb
    # pdb.set_trace()
    dataset = MVTecDRAEMTestDataset(opt.mvtec_path, resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    print(f"Found {len(dataloader)} inputs.")

    config = OmegaConf.load(opt.config_path)
    
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.ckpt_path)["state_dict"],
                          strict=False)  #TODO: modify the ckpt path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = 2
    model = model.cuda(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    cnt = 0
    with torch.no_grad():
        with model.ema_scope():
            for i_batch, batch in enumerate(dataloader):

                c_outpath = os.path.join(opt.outdir, 'condition'+str(cnt)+'.jpg')
                outpath = os.path.join(opt.outdir, str(cnt)+'.jpg')
                # print(outpath)
                condition = batch["image"].cpu().numpy().transpose(0,2,3,1)[0]*255
                mask = batch["mask"].cpu().numpy().transpose(0,2,3,1)[0]*255
                cond_all = batch["cond"] # condition = cond_all["c_concat"]
                cond_all["c_concat"] = cond_all["c_concat"].cuda(device)
                # cv2.imwrite(c_outpath, condition)

                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning(cond_all["c_crossattn"])]
                cond["c_concat"] =  [model.encode_first_stage((cond_all["c_concat"])).mode()]

                c = cond["c_concat"][0]
                noise = torch.randn_like(c)
                t = torch.randint(400, 500, (c.shape[0],), device=device).long()
                c_noisy = model.q_sample(x_start=c, t=t, noise=noise)
                # cond["c_concat"][0] = c_noisy
                
                shape = c.shape[1:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=cond, # or conditioning=c_noisy
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                sample = x_samples_ddim.cpu().numpy().transpose(0,2,3,1)[0]*255
                # cv2.imwrite(outpath, sample)
                combine_three_and_save_images(condition, sample, mask,outpath)
                cnt+=1




