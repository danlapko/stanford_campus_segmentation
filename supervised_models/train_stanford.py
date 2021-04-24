import sys, os

import torch
from pytorch_lightning.metrics import IoU

torch.manual_seed(0)
from kornia.losses import focal_loss
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.metrics.functional import iou
from segmentation_models_pytorch import DeepLabV3Plus, DeepLabV3

# from torchmetrics import IoU

sys.path.append(os.getcwd())
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision
import pytorch_lightning as pl
from typing import Union, List
import albumentations as A

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
        self.train_batch_size = 2
        self.val_batch_size = 16
        self.part_of_train_dataset = 0.015
        self.part_of_val_dataset = 0.15

        self.learning_rate = 5e-4
        self.num_classes = 4
        self.val_split_part = 0.1
        self.input_size = np.array([736, 736, 3])  # 576, 864
        self.interframe_step = 4

        self.transform_train = None
        self.transform_val = None
        self.train_dataset = None
        self.val_dataset = None
        self.unnormalizer = None

        self.iou = IoU(num_classes=4, reduction="none")

        self.net = DeepLabV3('efficientnet-b2', in_channels=9, classes=self.num_classes, encoder_weights="imagenet")

    def forward(self, x):
        out = self.net(x)
        # print(x.shape, x.min(), x.max(), out.shape, out.min(), out.max())
        return out

    def training_step(self, batch, batch_ix):
        triplet_ixs, stacked_triplet_ims, mask = batch
        out = self.forward(stacked_triplet_ims)
        # print(out.shape)
        # print(mask.shape)
        loss = F.cross_entropy(out, mask)
        # loss = focal_loss(out, mask, alpha=1, gamma=2, reduction="mean")

        self.log('train_loss', loss)
        self.logger.experiment.add_scalars("train_iou", {
            cat_name: cat_iou for cat_name, cat_iou in zip(self.train_dataset.categories,
                                                           iou(F.softmax(out, dim=1), mask, num_classes=4,
                                                               reduction='none'))}, global_step=self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_ix):
        triplet_ixs, stacked_triplet_ims, mask = batch
        out = self.forward(stacked_triplet_ims)

        loss = F.cross_entropy(out, mask)
        # loss = focal_loss(out, mask_batch, alpha=0.5, gamma=2, reduction="mean")
        self.iou.update(F.softmax(out, dim=1), mask)
        self.log('val_loss', loss)
        # log images with segmentation mask
        for triplet_ix, triplet_ims, out_mask in zip(triplet_ixs, stacked_triplet_ims, out):
            im = triplet_ims[3:6]
            im = self.unnormalizer(im)
            im = im.permute(1, 2, 0).cpu().numpy()
            im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
                np.uint8)

            out_mask = out_mask.permute(1, 2, 0).cpu().numpy()
            out_mask = np.argmax(out_mask, axis=-1)

            masked_img = self.val_dataset.generate_masked_image(im, out_mask) #mask[0].cpu().numpy())
            self.logger.experiment.add_image(f"images/{self.current_epoch}", torch.tensor(masked_img).permute(2, 0, 1))
            break
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        val_iou = self.iou.compute()
        self.iou.reset()
        self.logger.experiment.add_scalars("val_iou", {cat_name: cat_iou for cat_name, cat_iou in
                                                       zip(self.val_dataset.categories, val_iou)},
                                           global_step=self.global_step)

    def forward_img(self, im_triplet):
        raise NotImplemented
        self.net.eval()
        with torch.no_grad():
            # img = self.transform_val(image=img)["image"]
            x = im_triplet
            x = self.val_dataset.torch_transform(x)
            x = torch.unsqueeze(x, 0)
            x = x.to(self.device)
            out_mask = self.forward(x)
            # print(x.shape, out_mask.shape)
            out_mask = out_mask[0].permute(1, 2, 0).cpu().numpy()
            out_mask = np.argmax(out_mask, axis=-1)
            # print(out_mask.max())

        masked_img = self.val_dataset.generate_masked_image(im_triplet[3:6], out_mask)
        return masked_img

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # sch = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=3e-5, max_lr=self.learning_rate, step_size_up=2000,
        #                                         mode="triangular2")
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=2)
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
            A.RandomBrightnessContrast((0, 0.5), (0, 0.5), p=0.5),
            A.GaussNoise(p=0.3)])
        self.transform_val = A.Compose([
            A.SmallestMaxSize(min(self.input_size[:2]), always_apply=True, interpolation=cv2.INTER_AREA),
            A.RandomCrop(self.input_size[0], self.input_size[1], always_apply=1),
            # A.HorizontalFlip(),
            # A.GridDistortion(p=0.2)
        ])
        self.train_dataset = StanfordDataset("data/stanford_drone/videos", interframe_step=self.interframe_step,
                                             mode="train", part_of_dataset_to_use=self.part_of_train_dataset,
                                             transform=self.transform_train)
        self.val_dataset = StanfordDataset("data/stanford_drone/videos", interframe_step=self.interframe_step,
                                           mode="val", part_of_dataset_to_use=self.part_of_val_dataset,
                                           transform=self.transform_val)
        self.unnormalizer = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=8,
                          pin_memory=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=8,
                          pin_memory=True)


def train():
    model = StanfordSegModel()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints_stanford/deeplabv3_effnet-b2/',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=1, callbacks=[lr_monitor],
                         checkpoint_callback=checkpoint_callback,
                         num_sanity_val_steps=-1,
                         log_every_n_steps=4,
                         # max_epochs=1,
                         # resume_from_checkpoint='checkpoints_stanford/deeplabv3_effnet-b2/epoch=4-step=2229.ckpt'
                         )
    trainer.fit(model)


if __name__ == '__main__':
    train()
