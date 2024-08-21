import os
from pathlib import Path
from PIL import Image

def merge_images(image_paths, output_path):
    # 打开四张图片
    images = [Image.open(path) for path in image_paths]
    
    # 获取单个图片的宽度和高度（假设所有图片的大小相同）
    width, height = images[0].size
    
    # 创建一个新的空白图像，大小为 2x2 布局
    merged_image = Image.new('RGB', (2 * width, 2 * height))
    
    # 将四张图片粘贴到新图像的四个区域
    merged_image.paste(images[0], (0, 0))
    merged_image.paste(images[1], (width, 0))
    merged_image.paste(images[2], (0, height))
    merged_image.paste(images[3], (width, height))
    
    # 保存合并后的图片
    merged_image.save(output_path)
    
def f1(path):
    names = ["gt_images{}.jpg", "rec_images{}.jpg", "in_masks{}.jpg", "out_masks{}.jpg"]
    path_list = [str(folder) for folder in Path(path).rglob('*') if folder.is_dir()]
    for path_item in path_list:
        img_names = os.listdir(path_item)
        img_len = 0
        for img_name in img_names:
            if img_name.startswith("gt_"):
                img_len += 1

        for i in range(img_len):
            input_paths = [os.path.join(path_item, name.format(str(i))) for name in names]
            output_path = os.path.join(path_item, str(i) + "_.jpg")
            merge_images(input_paths, output_path)
        
        for img_name in img_names:
            if img_name.startswith(("gt", "rec", "in_", "out_", "heat")):
                img_path = os.path.join(path_item, img_name)
                os.remove(img_path)

path = "/hdd2/yyk/DiffAD-main_4/logs/img_feature_817/final_output_v2"
f1(path)
