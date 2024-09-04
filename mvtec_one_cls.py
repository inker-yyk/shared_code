class MVTecDRAEMTrainDataset_one_cls(Dataset):

    def __init__(self, root_dir, anomaly_source_path, obj_name, resize_shape=None, k_shot=4, index_path=""):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.k_shot = k_shot
        self.index_path = index_path
        self.obj_name = obj_name

        # /data / zhangxinyi / dataset / mvtec / toothbrush / train / good
        # self.image_paths = []
        # for obj in obj_list:
        # obj = 'cable'
        # self.image_paths += sorted(glob.glob(root_dir + obj + "/train/good/*.png"))
        # print(len(self.image_paths))
        
        self.image_paths = []
        if not self.index_path:
            for obj in obj_list:
                images_path = sorted(glob.glob(os.path.join(root_dir, obj) + "/train/*/*.png"))
                random_path = random.sample(images_path, self.k_shot)
                for path in random_path:
                    self.image_paths.append({"path": path, "prompt": prompt_template.format(obj)})  
        else:
            with open(self.index_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    obj_name = line.rstrip().split("/")[-4]
                    if obj_name == self.obj_name:
                        self.image_paths.append({"path": line.rstrip(), "prompt": prompt_template.format(obj_name)})
        
        # new_lines = []
        # for dict_item in self.image_paths:
        #     new_lines.append(dict_item["path"] + "\n")
        # with open("/hdd2/yyk/DiffAD-main_2/trainset_822.txt", "w") as file:
        #     file.writelines(new_lines)
            
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        


    def __len__(self):
        return len(self.image_paths) * 60


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image. Please check the file path and try again.")

        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx_dict = self.image_paths[idx % len(self.image_paths)]
        path, prompt = idx_dict["path"], idx_dict["prompt"]
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(path, self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx, "prompt": prompt, 
                  "cond": {"c_concat": augmented_image, "c_crossattn": prompt}}
        return sample
