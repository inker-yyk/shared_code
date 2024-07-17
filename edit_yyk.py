# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import os
import math
import random
import sys
from argparse import ArgumentParser

import einops
import sys
sys.path.append("/hdd2/yyk/InstructDiffusion-main/src/k-diffusion")
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

import requests

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond \
            = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                image_cfg_scale * (out_cond - out_txt_cond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)

    print(m, u)
    return model

class dotdict(dict):
    """A dictionary that supports dot notation."""
    def __getattr__(self, item):
        return self.get(item)
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, item):
        del self[item]

def main(args):

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed

    if args.input.startswith("http"):
        input_image = Image.open(requests.get(args.input, stream=True).raw).convert("RGB")
    else:
        input_image = Image.open(args.input).convert("RGB")
    
    temp_image = input_image
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width_resize = int((width * factor) // 64) * 64
    height_resize = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
    

    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad(), autocast("cuda"):
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(model.parameters()).device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        print(x.shape)
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

        edited_image = ImageOps.fit(edited_image, (width, height), method=Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', (edited_image.width + temp_image.width, edited_image.height))
        new_image.paste(temp_image, (0, 0))
        new_image.paste(edited_image, (temp_image.width, 0))
        # new_image.save(output_dir+'/output_'+args.input.split('/')[-1].split('.')[0]+'_seed'+str(seed)+'.jpg')

        name = args.input.split('/')[-1].split('.')[0]
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
        new_image.save(os.path.join(output_dir, name, name + "_seed" + str(seed) + '.jpg'))

if __name__ == "__main__":
    args_dict = dotdict({
        "resolution": 512,
        "steps": 100,
        "config": "configs/instruct_diffusion.yaml",
        "ckpt": "checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt",
        "vae_ckpt": None,
        "input": "/hdd2/yyk/001.png",
        "outdir": "logs",
        "edit": "Use yellow to encircle the defective portion of this photo of carpet.",
        "cfg_text": 5.0,
        "cfg_image": 1.25,
        "seed": None
    })
    
    root_img_path = "/home/yyk/datasets/mvtec_anomaly_detection/carpet/test"
    save_root_path = "/hdd2/yyk/InstructDiffusion-main/logs/output"
    os.makedirs(save_root_path, exist_ok=True)
    cls_list = os.listdir(root_img_path)
    cls_list.reverse()
    
    for anomaly_cls in cls_list:
        cls_path = os.path.join(root_img_path, anomaly_cls)
        cls_save_path = os.path.join(save_root_path, anomaly_cls)
        os.makedirs(cls_save_path, exist_ok=True)
        
        args_dict["outdir"] = cls_save_path
        args_dict["edit"] = "Use yellow to encircle the defective portion of this photo of carpet."
        
        img_list = os.listdir(cls_path)
        img_list = [os.path.join(cls_path, one) for one in img_list]

        
        for img_path in img_list:
            for i in range(3):
                args_dict["input"] = img_path
                main(args_dict)
