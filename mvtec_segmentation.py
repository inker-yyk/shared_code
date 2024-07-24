# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Binxin Yang (tennyson@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations

import os
import random
import copy
import json
import math
from pathlib import Path
from typing import Any

import prompt 
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class MvtecDataset(Dataset):
    def __init__(
        self,
        path: str, # 
        name: list,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        test: bool = False,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.test = test

        self.item_list = []

        for cls_name in name:
            name_prompt = [one.format(cls_name) for one in prompt.instructions]
            ad_img_path = os.path.join(path, cls_name + "_anomaly", "ad_img")
            output_img_path = os.path.join(path, cls_name + '_anomaly', "output_img")
            source_img_path = os.path.join(path, cls_name + "_anomaly", "source")
            ad_item_list = [os.path.join(ad_img_path, one) for one in os.listdir(ad_img_path)]
            output_item_list = [os.path.join(output_img_path, one) for one in os.listdir(output_img_path)]
            source_item_list = [os.path.join(source_img_path, one) for one in os.listdir(source_img_path)]
            assert len(ad_item_list) == len(output_item_list)
            for one_prompt in name_prompt:
                for a, b in zip(ad_item_list, output_item_list):
                    self.item_list.append({"input": a, "output": b, "text": one_prompt})
                for a in source_item_list:
                    self.item_list.append({"input": a, "output": a, "text": one_prompt})
        
        
    def __len__(self) -> int:
        return len(self.item_list)
    

    def __getitem__(self, i:int) -> dict[str, Any]:
        item = self.item_list[i]
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = Image.open(item["input"]).convert("RGB"), Image.open(item["output"]).convert("RGB")
        image_prompt = item["text"]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        # crop = torchvision.transforms.RandomCrop(self.crop_res)
        # flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        # image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
