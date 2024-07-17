from PIL import Image
import os

def merge_images_horizontally(directory_path, temp_path):
    # 获取目录下所有jpg文件
    images = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]
    
    # 检查是否有图片
    if not images:
        print("No JPG images found in the directory.")
        return
    
    # 读取所有图片并存储为Image对象列表
    img_objects = []
    for img_file in images:
        img_path = os.path.join(directory_path, img_file)
        img = Image.open(img_path)
        img_objects.append(img)
    
    # 确保所有图片高度一致，如果高度不同，则调整高度
    max_height = max([img.height for img in img_objects])
    for i, img in enumerate(img_objects):
        if img.height != max_height:
            width = int(img.width * (max_height / img.height))
            img_objects[i] = img.resize((width, max_height), Image.ANTIALIAS)
    
    # 创建新图像，宽度为所有图像宽度之和，高度取最大高度
    total_width = sum([img.width for img in img_objects])
    new_img = Image.new('RGB', (total_width, max_height))
    
    # 将图像横向粘贴到新图像中
    x_offset = 0
    for img in img_objects:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # 保存新图像
    new_img.save(os.path.join(temp_path, os.path.basename(directory_path) + '_merged.jpg'))
    
    # 删除原始图片
    for img_file in images:
        os.remove(os.path.join(directory_path, img_file))
    
    print("Images merged and original images deleted.")


root_path = "/hdd2/yyk/InstructDiffusion-main/logs/output"
anomaly_cls = os.listdir(root_path)
anomaly_cls_path = [os.path.join(root_path, one) for one in anomaly_cls]

for a_path in anomaly_cls_path:
    a_list = os.listdir(a_path)
    a_list = [os.path.join(a_path, one) for one in a_list]
    
    for one in a_list:
        merge_images_horizontally(one, a_path)
