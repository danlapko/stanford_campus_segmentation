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
    One sample - n stacked grayscale sequential frames.
    """

    def __init__(self, root_dir: str,
                 n_frame_samples=3, interframe_step=1,
                 mode="val", part_of_dataset_to_use=1,
                 dilate=False, transform=None):
        """
            root_dir: root dir of dataset where scenes are placed;
            n_frame_samples: number of different frames get into one sample (sample - image with n-channels);
            interframe_step - distance between consecutive frames get into sample;
            mode - "val" or "train";
            part_of_dataset_to_use - float in range of 0..1, representing which part of origin dataset to use
                                    (samples get into the dataset randomly with probability of part_of_dataset_to_use);
            dilate - if to use cv2.dilate to enlarge masks area;
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

        self.categories = ["default", "person", "bicycle", "car", "skate"]
        self.categories_2_label_map = {cat: i for i, cat in enumerate(self.categories)}
        self.labels_2_category_map = {i: cat for i, cat in enumerate(self.categories)}

        self.categories_2_color_map = {
            "car": (0, 0, 255),  # blue
            "person": (0, 255, 0),  # green
            "bicycle": (255, 255, 0)  # yellow
        }

        if self.n_frame_samples == 3:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.456, ] * n_frame_samples
            self.std = [0.224, ] * n_frame_samples

        self.torch_transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        self.dilate_kernel = np.ones((5, 5), np.uint8)

    def _load_video_dirs(self):
        """ Loading video paths for current dataset: 'val' contains 'video0' from every scene and whole 'little' scene,
                                                    'train' contains all other videos """
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
        """ Generates samples consisting of lists with len==n_frame_samples of frame_paths, appropriate mask_paths
        and bounding boxes """
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
        """ To have ability to set different transforms for train and val subsets after retrieving them from
        random_split() """
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, sample_i):
        video_dir, sample_frames_ixs, target_frame_ix, boxes = self.samples[sample_i]
        sample_ims = []
        for frame_ix in sample_frames_ixs:
            im = cv2.imread(os.path.join(video_dir, "frames", f"{frame_ix}.jpg"), 0)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            sample_ims.append(im)

        stacked_sample_ims = np.stack(sample_ims, axis=-1)
        mask = cv2.imread(os.path.join(video_dir, "seg_masks", f"{target_frame_ix}.png"), 0)
        mask[mask == self.categories_2_label_map["skate"]] = self.categories_2_label_map[
            "person"]  # reduce 'skate' class to 'person'
        mask[mask == self.categories_2_label_map["car"]] = self.categories_2_label_map[
            "default"]  # reduce 'car' class to 'default'

        if self.dilate:
            mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        if self.transform is not None:
            aug = self.transform(image=stacked_sample_ims, mask=mask)  # , bboxes=boxes)  # , box_mask=box_mask)
            stacked_sample_ims = aug['image']
            mask = aug['mask']

        if self.transform is None:
            stacked_sample_ims = Image.fromarray(stacked_sample_ims)

        stacked_sample_ims = self.torch_transform(stacked_sample_ims)
        mask = torch.from_numpy(mask).long()
        return sample_i, stacked_sample_ims, mask

    def generate_masked_image(self, img, mask, alpha=0.4, gray_img=False):
        """ Generates images with transparent mask by given image and mask (alpha - mask opacity)
        img: 1-channel or 3-channel image;
        mask: 1-channel mask where classes encoded by appropriate int labels;
        gray_image: if to generate masked image for gray input image"""
        if gray_img:
            img = np.stack((img,) * 3, axis=-1)
        bgr_mask = np.zeros_like(img)
        for cat, cat_bgr in self.categories_2_color_map.items():
            bgr_mask[mask == self.categories_2_label_map[cat]] = cat_bgr

        img = cv2.addWeighted(img, 1, bgr_mask, alpha, 0)
        return img


if __name__ == '__main__':
    stanford_dataset = StanfordDataset(root_dir="data/stanford_drone/videos")
    print(len(stanford_dataset.samples))
