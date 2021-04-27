import sys, os

from pytorch_lightning.metrics import IoU

from kornia.losses import focal_loss
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional import iou
from segmentation_models_pytorch import DeepLabV3

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Union, List
import albumentations as A

torch.manual_seed(0)
sys.path.append(os.getcwd())

from supervised_models.stanford_dataset import StanfordDataset


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class StanfordSegModel(pl.LightningModule):
    def __init__(self):
        super(StanfordSegModel, self).__init__()
        self.in_channels = 3
        self.interframe_step = 10
        self.num_classes = 3  # default - includes background, buses, cars and golf carts,
        # person - includes pedestrians and skaters,
        # bicycle - includes bikers

        self.input_size = np.array([640, 640, self.in_channels])  # 640, 640

        self.train_batch_size = 4
        self.val_batch_size = 16
        self.learning_rate = 1e-4

        self.part_of_train_dataset = 0.045
        self.part_of_val_dataset = 0.15

        self.transform_train = None
        self.transform_val = None
        self.train_dataset = None
        self.val_dataset = None
        self.unnormalizer = None

        self.iou = IoU(num_classes=self.num_classes, reduction="none")

        self.net = DeepLabV3('efficientnet-b0', in_channels=self.in_channels, classes=self.num_classes)

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_ix):
        sample_ixs, stacked_sample_ims, mask = batch

        out = self.forward(stacked_sample_ims)

        loss = F.cross_entropy(out, mask, reduction="mean")
        # loss = focal_loss(out, mask, alpha=1, gamma=2, reduction="mean")

        self.log('train_loss', loss)
        self.logger.experiment.add_scalars("train_iou", {
            cat_name: cat_iou for cat_name, cat_iou in zip(self.train_dataset.categories,
                                                           iou(F.softmax(out, dim=1), mask,
                                                               num_classes=self.num_classes,
                                                               reduction='none'))}, global_step=self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_ix):
        sample_ixs, stacked_sample_ims, mask = batch

        out = self.forward(stacked_sample_ims)

        loss = F.cross_entropy(out, mask, reduction="mean")
        # loss = focal_loss(out, mask, alpha=1, gamma=2, reduction="mean")

        self.iou.update(F.softmax(out, dim=1), mask)
        self.log('val_loss', loss)

        # log images with segmentation mask
        for sample_ix, sample_ims, out_mask in zip(sample_ixs, stacked_sample_ims, out):
            sample_ims = self.unnormalizer(sample_ims)
            im = sample_ims[len(sample_ims) // 2]  # peak central channel (central gray image)
            im = im.cpu().numpy()
            im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
                np.uint8)

            out_mask = out_mask.permute(1, 2, 0).cpu().numpy()
            out_mask = np.argmax(out_mask, axis=-1)

            masked_img = self.val_dataset.generate_masked_image(im, out_mask, gray_img=True)  # mask[0].cpu().numpy())
            self.logger.experiment.add_image(f"images/{self.current_epoch}", torch.tensor(masked_img).permute(2, 0, 1))
            break
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_iou = self.iou.compute()
        self.iou.reset()
        self.logger.experiment.add_scalars("val_iou", {cat_name: cat_iou for cat_name, cat_iou in
                                                       zip(self.val_dataset.categories, val_iou)},
                                           global_step=self.global_step)

    def forward_img(self, sample_ims: List[np.array], smallest_size=640):
        """ Process sample grayscale images (n_images == self.input_size) and generates one segmentation
        (for central input img) """
        transform = A.Compose([
            A.SmallestMaxSize(smallest_size, always_apply=True, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16,
                          border_mode=cv2.BORDER_CONSTANT,
                          always_apply=True, value=0)
        ])

        sample_ims = np.stack(sample_ims, axis=-1)
        sample_ims = transform(image=sample_ims)["image"]
        self.net.eval()
        with torch.no_grad():
            x = sample_ims
            x = self.val_dataset.torch_transform(x)
            x = torch.unsqueeze(x, 0)
            x = x.to(self.device)
            out_mask = self.forward(x)
            out_mask = out_mask[0].permute(1, 2, 0).cpu().numpy()
            out_mask = np.argmax(out_mask, axis=-1)
        masked_img = self.val_dataset.generate_masked_image(sample_ims[:, :, sample_ims.shape[-1] // 2], out_mask,
                                                            gray_img=True)
        return masked_img

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # sch = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=3e-5, max_lr=self.learning_rate, step_size_up=2000,
        #                                         mode="triangular2")
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=2, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': 'val_loss',
        }

    def setup(self, stage: str) -> None:
        self.transform_train = A.Compose([
            A.SmallestMaxSize(min(self.input_size[:2]), always_apply=True, interpolation=cv2.INTER_AREA),
            A.RandomCrop(self.input_size[0], self.input_size[1], always_apply=1),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(p=0.5),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0, 0.5), (0, 0.5), p=0.2),
            A.GaussNoise(p=0.2)],
            # additional_targets={'box_mask': 'mask'}, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )

        self.transform_val = A.Compose([
            A.SmallestMaxSize(min(self.input_size[:2]), always_apply=True, interpolation=cv2.INTER_AREA),
            A.RandomCrop(self.input_size[0], self.input_size[1], always_apply=1),
        ],
            # additional_targets={'box_mask': 'mask'}, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )

        # self.transform_test = A.Compose([
        #     A.SmallestMaxSize(min(self.input_size[:2]), always_apply=True, interpolation=cv2.INTER_AREA),
        #     # A.CenterCrop(height=self.input_size[0], width=self.input_size[1], always_apply=True)
        #     # A.PadIfNeeded(min_height=self.input_size[0], min_width=self.input_size[1], pad_height_divisor=16,
        #     #               pad_width_divisor=16, border_mode=cv2.BORDER_CONSTANT, always_apply=True, value=0)
        #     A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16,
        #                   border_mode=cv2.BORDER_CONSTANT,
        #                   always_apply=True, value=0)
        #     # A.RandomCrop(self.input_size[0], self.input_size[1], always_apply=1),
        # ])

        self.train_dataset = StanfordDataset("data/stanford_drone/videos",
                                             n_frame_samples=self.in_channels, interframe_step=self.interframe_step,
                                             mode="train", part_of_dataset_to_use=self.part_of_train_dataset,
                                             dilate=True, transform=self.transform_train)
        self.val_dataset = StanfordDataset("data/stanford_drone/videos",
                                           n_frame_samples=self.in_channels, interframe_step=self.interframe_step,
                                           mode="val", part_of_dataset_to_use=self.part_of_val_dataset,
                                           dilate=True, transform=self.transform_val)

        self.unnormalizer = UnNormalize(mean=self.train_dataset.mean, std=self.train_dataset.std)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=8,
                          pin_memory=True, drop_last=True)


def train():
    model = StanfordSegModel()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'checkpoints_stanford/deeplabv3_effnet-b0-{model.in_channels}ch/',
        save_top_k=2,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=1, callbacks=[lr_monitor],
                         checkpoint_callback=checkpoint_callback,
                         num_sanity_val_steps=20,
                         log_every_n_steps=4,
                         max_epochs=23,
                         resume_from_checkpoint="checkpoints_stanford/deeplabv3_effnet-b0-3ch/backup.epoch=22-step=15291.ckpt"
                         )
    trainer.fit(model)


if __name__ == '__main__':
    train()
