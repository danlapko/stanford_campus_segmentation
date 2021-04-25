import os.path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm


class StanfordDataset(Dataset):
    """
    Stanford drone dataset https://cvgl.stanford.edu/projects/uav_data/
    Ground truth masks are obtained by subtracting the background in the bounding rectangles of this object.
    """

    def __init__(self, root_dir: str,
                 n_frame_samples=3, interframe_step=1,
                 mode="val", part_of_dataset_to_use=1,
                 dilate=False, transform=None):
        """
            root_dir: root dir of dataset where scenes all placed
            interframe_step - distance between consecutive frames get into sample,
        """
        assert n_frame_samples % 2 == 1
        self.n_frame_samples = n_frame_samples
        self.dilate = dilate
        self.part_of_dataset_to_use = part_of_dataset_to_use
        self.interframe_step = interframe_step
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.video_dirs = self._load_video_dirs()

        self.samples = self._prepare_samples()

        self.categories = ["default", "car", "person", "bicycle"]
        self.categories_2_label_map = {cat: i for i, cat in enumerate(self.categories)}
        self.labels_2_category_map = {i: cat for i, cat in enumerate(self.categories)}
        # self.categories_2_label_map = {"default": 0, "car": 1, "person": 2, "bicycle": 2}
        # self.labels_2_category_map = {0: "default", 1: "car", 2: "person"}

        self.categories_2_color_map = {"car": (0, 0, 255),  # blue
                                       "person": (0, 255, 0),  # green
                                       "bicycle": (255, 255, 0)  # yellow
                                       }

        # self.mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
        # self.mean = [0.485, 0.456, 0.406]
        # self.std = [0.229, 0.224, 0.225]
        self.mean = [0.456, ] * n_frame_samples
        self.std = [0.224, ] * n_frame_samples
        self.torch_transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        self.dilate_kernel = np.ones((5, 5), np.uint8)

    def _load_video_dirs(self):
        video_dirs = []
        scene_names = os.listdir(self.root_dir)
        if self.mode == "val":
            for scene_name in scene_names:
                if scene_name == "little":
                    for video_name in os.listdir(os.path.join(self.root_dir, scene_name)):
                        video_dirs.append(os.path.join(self.root_dir, scene_name, video_name))
                else:
                    video_dirs.append(os.path.join(self.root_dir, scene_name, "video0"))
        elif self.mode == "train":
            for scene_name in scene_names:
                if scene_name == "little":
                    continue
                else:
                    for video_name in os.listdir(os.path.join(self.root_dir, scene_name)):
                        if video_name != "video0":
                            video_dirs.append(os.path.join(self.root_dir, scene_name, video_name))
        else:
            raise NotImplemented(f"mode {self.mode} not implemented")
        return video_dirs

    def _prepare_samples(self):
        all_samples = []
        for video_dir in self.video_dirs:
            frame_names = os.listdir(os.path.join(video_dir, "frames"))
            frame_nums = list(sorted(map(lambda name: int(name.split(".")[0]), frame_names)))
            video_samples = []
            boxes_df = pd.read_csv(os.path.join(video_dir, "boxes.csv"))
            for i in range(len(frame_nums)):
                if i + (self.n_frame_samples - 1) * self.interframe_step >= len(frame_nums):
                    break
                sample_frames_ixs = [frame_nums[i + j * self.interframe_step] for j in range(self.n_frame_samples)]
                target_frame_ix = i + self.n_frame_samples // 2 * self.interframe_step
                sample_boxes_df = boxes_df[boxes_df["frame"] == target_frame_ix]
                sample_boxes = sample_boxes_df.drop(columns=["frame"]).values.tolist()

                if random.random() < self.part_of_dataset_to_use:
                    video_samples.append((video_dir, sample_frames_ixs, target_frame_ix, sample_boxes))

            all_samples.extend(video_samples)
        return all_samples

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_i):
        video_dir, sample_frames_ixs, target_frame_ix, boxes = self.samples[sample_i]
        sample_ims = []
        sample_masks = []
        for frame_ix in sample_frames_ixs:
            im = cv2.imread(os.path.join(video_dir, "frames", f"{frame_ix}.jpg"), 0)
            sample_ims.append(im)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(video_dir, "seg_masks", f"{frame_ix}.png"), 0)
            mask[mask == self.categories_2_label_map["bicycle"]] = self.categories_2_label_map["person"]
            if self.dilate:
                mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
            sample_masks.append(mask)

        stacked_sample_ims = np.stack(sample_ims, axis=-1)
        stacked_sample_masks = np.stack(sample_masks, axis=-1)

        # mask = cv2.imread(os.path.join(video_dir, "seg_masks", f"{target_frame_ix}.png"), 0)
        # mask[mask == self.categories_2_label_map["bicycle"]] = self.categories_2_label_map["person"]

        if self.transform is not None:
            aug = self.transform(image=stacked_sample_ims, mask=stacked_sample_masks)  # , bboxes=boxes)  # , box_mask=box_mask)
            stacked_sample_ims = aug['image']
            stacked_sample_masks = aug['mask']

        if self.transform is None:
            stacked_sample_ims = Image.fromarray(stacked_sample_ims)

        stacked_sample_ims = self.torch_transform(stacked_sample_ims)
        stacked_sample_masks = torch.from_numpy(stacked_sample_masks).long()
        return sample_i, stacked_sample_ims, stacked_sample_masks

    def show_masks(self, alpha):
        raise NotImplemented
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

    def generate_masked_image(self, img, mask, alpha=0.4, gray_img=False):
        if gray_img:
            img = np.stack((img,) * 3, axis=-1)
        bgr_mask = np.zeros_like(img)
        for cat, cat_bgr in self.categories_2_color_map.items():
            bgr_mask[mask == self.categories_2_label_map[cat]] = cat_bgr

        img = cv2.addWeighted(img, 1, bgr_mask, alpha, 0)
        return img


if __name__ == '__main__':
    sdd_dataset = StanfordDataset(root_dir="data/stanford_drone")
    # sdd_dataset.generate_value_masks_from_bgr()
    sdd_dataset.show_masks(alpha=1)
    print(len(sdd_dataset.images), len(sdd_dataset.value_masks))
