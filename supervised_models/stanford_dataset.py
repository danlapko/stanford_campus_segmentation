import os.path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A


class StanfordDataset(Dataset):
    """
    Stanford drone dataset https://cvgl.stanford.edu/projects/uav_data/
    Ground truth masks are obtained by subtracting the background in the bounding rectangles of this object.
    """

    def __init__(self, root_dir: str, interframe_step=1, mode="val", part_of_dataset_to_use=1, transform=None):
        """
            root_dir: root dir of dataset where scenes all placed
            interframe_step: 1 - consecutive 3 frames get into sample triplet,
                             >1 - (i_frame-interframe_step, i_frame, i_frame+interframe_step) frames get into triplet,
                             0 -  (i_frame, i_frame, i_frame) - triplet consists of same frames
        """
        self.part_of_dataset_to_use = part_of_dataset_to_use
        self.interframe_step = interframe_step
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.video_dirs = self._load_video_dirs()

        self.triplets = self._prepare_triplets()

        self.categories = ["default", "car", "person", "bicycle"]
        self.categories_2_label_map = {cat: i for i, cat in enumerate(self.categories)}
        self.labels_2_category_map = {i: cat for i, cat in enumerate(self.categories)}

        self.categories_2_color_map = {"car": (0, 0, 255),  # blue
                                       "person": (0, 255, 0),  # green
                                       "bicycle": (255, 255, 0)  # yellow
                                       }

        self.mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
        self.torch_transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

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

    def _prepare_triplets(self):
        all_triplets = []
        for video_dir in self.video_dirs:
            frame_names = os.listdir(os.path.join(video_dir, "frames"))
            frame_nums = list(sorted(map(lambda name: int(name.split(".")[0]), frame_names)))
            video_triplets = []
            boxes_df = pd.read_csv(os.path.join(video_dir, "boxes.csv"))
            for i in range(len(frame_nums)):
                if i + 2 * self.interframe_step >= len(frame_nums):
                    break
                # triplet = frame_nums[i:i + 2 * self.interframe_step + 1] #
                triplet = [frame_nums[i], frame_nums[i + self.interframe_step],
                           frame_nums[i + 2 * self.interframe_step]]
                triplet_boxes_df = boxes_df[boxes_df["frame"] == i]
                triplet_boxes = triplet_boxes_df.drop(columns=["frame"]).values.tolist()
                if random.random() < self.part_of_dataset_to_use:
                    video_triplets.append((video_dir, triplet, triplet_boxes))

            all_triplets.extend(video_triplets)
        return all_triplets

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        video_dir, triplet_nums, boxes = self.triplets[item]
        triplet_ims = []
        for triplet_num in triplet_nums:
            im = cv2.imread(os.path.join(video_dir, "frames", f"{triplet_num}.jpg"))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            triplet_ims.append(im)

        stacked_triplet_ims = np.concatenate(triplet_ims, axis=2)
        mask = cv2.imread(os.path.join(video_dir, "seg_masks", f"{triplet_nums[1]}.png"), 0)

        if self.transform is not None:
            # print(len(triplet_ims), triplet_ims[0].shape, stacked_triplet_ims.shape, mask.shape)
            # a_transform = self.transform[:]
            # if self.rotate_if_h_more_w and stacked_triplet_ims.shape[0] > stacked_triplet_ims.shape[1]:
            #     a_transform = [A.Rotate((90, 90), cv2.INTER_AREA, p=1)] + a_transform
            # a_transform = A.Compose(a_transform)

            aug = self.transform(image=stacked_triplet_ims, mask=mask)
            stacked_triplet_ims = aug['image']
            # stacked_triplet_ims = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            stacked_triplet_ims = Image.fromarray(stacked_triplet_ims)

        # print(stacked_triplet_ims.shape)
        stacked_triplet_ims = self.torch_transform(stacked_triplet_ims)
        # print(stacked_triplet_ims.shape)
        # if self.rotate_if_h_more_w and stacked_triplet_ims.shape[1] > stacked_triplet_ims.shape[2]:
        #     stacked_triplet_ims = torch.rot90(stacked_triplet_ims, 1, [1, 2])
        mask = torch.from_numpy(mask).long()
        # print(len(video_dir), len(triplet_nums), stacked_triplet_ims.shape, mask.shape, len(boxes))
        triplet_ix = item
        return triplet_ix, stacked_triplet_ims, mask

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

    def generate_masked_image(self, img, mask, alpha=0.4):
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
