import os.path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm


class SDD_Dataset(Dataset):
    """ Semantic drone dataset https://www.tugraz.at/index.php?id=22387 """

    def __init__(self, root_dir: str, preload=False, transform=None):
        """
            root_dir: root dir of dataset where placed: RGB_color_image_masks/, semantic_drone_dataset/,
                      value_masks/ and class_dict_seg.csv
        """
        self.root_dir = root_dir
        self.preload = preload
        self.transform = transform
        self.original_images_dir = os.path.join(root_dir, "semantic_drone_dataset", "original_images")
        self.rgb_masks_dir = os.path.join(root_dir, "RGB_color_image_masks")
        self.value_masks_dir = os.path.join(root_dir, "value_masks")

        self.categories = ["default", "car", "person", "bicycle"]
        self.categories_2_label_map = {cat: i for i, cat in enumerate(self.categories)}
        self.labels_2_category_map = {i: cat for i, cat in enumerate(self.categories)}

        self.categories_2_bgr_map = {}
        tmp_df = pd.read_csv(os.path.join(self.root_dir, "class_dict_seg.csv"), index_col="name", sep=",")
        for category in self.categories[1:]:
            r, g, b = tmp_df.loc[category].values.tolist()
            self.categories_2_bgr_map[category] = (b, g, r)

        self.img_names = os.listdir(self.original_images_dir)
        self.images = [None] * len(self.img_names)
        self.value_masks = [None] * len(self.img_names)

        if self.preload:
            for i, img_name in tqdm(enumerate(self.img_names), desc="show_masks"):
                img = cv2.imread(os.path.join(self.original_images_dir, img_name))
                mask = cv2.imread(os.path.join(self.value_masks_dir, img_name.replace(".jpg", ".png")), 0)

                self.images[i] = img
                self.value_masks[i] = mask

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img = cv2.imread(os.path.join(self.original_images_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.value_masks_dir, img_name.replace(".jpg", ".png")), 0)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img_name, img, mask

    def generate_value_masks_from_bgr(self):
        for i, img_name in tqdm(enumerate(self.img_names), desc="generating value_masks from bgr_masks"):

            bgr_mask = cv2.imread(os.path.join(self.rgb_masks_dir, img_name.replace(".jpg", ".png")))
            mask = np.zeros(bgr_mask.shape[:2])
            for cat, cat_bgr in self.categories_2_bgr_map.items():
                cat_mask = (bgr_mask == cat_bgr).all(-1)
                mask[cat_mask] = self.categories_2_label_map[cat]

            cv2.imwrite(os.path.join(self.value_masks_dir, img_name.replace(".jpg", ".png")), mask)

    def show_masks(self, alpha):
        for i, img_name in tqdm(enumerate(self.img_names), desc="show_masks"):
            img = cv2.imread(os.path.join(self.original_images_dir, img_name))
            mask = cv2.imread(os.path.join(self.value_masks_dir, img_name.replace(".jpg", ".png")), 0)

            img = self.generate_masked_image(img, mask, alpha=alpha)
            img = cv2.resize(img, None, fx=0.15, fy=0.15)
            cv2.imshow(f"masked", img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyAllWindows()

    def generate_masked_image(self, img, mask, alpha=0.6):
        bgr_mask = np.zeros_like(img)
        for cat, cat_bgr in self.categories_2_bgr_map.items():
            bgr_mask[mask == self.categories_2_label_map[cat]] = cat_bgr

        img = cv2.addWeighted(img, 1, bgr_mask, alpha, 0)
        return img


if __name__ == '__main__':
    sdd_dataset = SDD_Dataset(root_dir="data/SDD")
    # sdd_dataset.generate_value_masks_from_bgr()
    sdd_dataset.show_masks(alpha=0.5)
    print(len(sdd_dataset.images), len(sdd_dataset.value_masks))
