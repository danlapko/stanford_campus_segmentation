import sys, os

import torch

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

from supervised_models.sdd_dataset import SDD_Dataset


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


class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.batch_size = 3
        self.learning_rate = 1e-3
        self.num_classes = 4
        self.val_split_part = 0.1

        self.train_dataset = None
        self.val_dataset = None
        self.unnormalizer = None

        # self.net = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        # self.net.classifier = torchvision.models.segmentation.fcn.FCNHead(2048, self.num_classes)
        # self.net = UNet(num_classes=self.num_classes, bilinear=False)
        # self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        # self.net.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, self.num_classes)
        self.net = DeepLabV3('efficientnet-b2', in_channels=3, classes=self.num_classes)
        # self.net = DeepLabV3('mobilenet_v2', in_channels=3, classes=self.num_classes)
        # self.net = ENet(num_classes=19)

        # self.net = ENet(num_classes=19)

    def forward(self, x):
        # out = F.softmax(self.net(x)['out'], dim=1)
        # out = self.net(x)['out']
        out = self.net(x)
        # print(x.shape, x.min(), x.max(), out.shape, out.min(), out.max())
        return out

    def training_step(self, batch, batch_ix):
        img_name, img, mask = batch
        # img = img.float()
        # mask = mask.long()
        out = self.forward(img)
        # print(out.shape)
        # print(mask.shape)
        loss = F.cross_entropy(out, mask)
        # loss = focal_loss(out, mask, alpha=1, gamma=2, reduction="mean")

        self.log('train_loss', loss)
        self.log('lr', self.learning_rate)
        self.logger.experiment.add_scalars("train_iou", {
            cat_name: cat_iou for cat_name, cat_iou in zip(self.train_dataset.dataset.categories,
                                                           iou(F.softmax(out, dim=1), mask, num_classes=4,
                                                               reduction='none'))}, global_step=self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_ix):
        im_name_batch, im_batch, mask_batch = batch
        out = self.forward(im_batch)

        loss = F.cross_entropy(out, mask_batch)
        # loss = focal_loss(out, mask_batch, alpha=0.5, gamma=2, reduction="mean")

        self.log('val_loss', loss)

        # log images with segmentation mask
        for im_name, im, out_mask in zip(im_name_batch, im_batch, out):
            im = self.unnormalizer(im)
            im = im.permute(1, 2, 0).cpu().numpy()
            im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
                np.uint8)

            out_mask = out_mask.permute(1, 2, 0).cpu().numpy()
            out_mask = np.argmax(out_mask, axis=-1)

            masked_img = self.val_dataset.dataset.generate_masked_image(im, out_mask)
            self.logger.experiment.add_image(f"images/{self.current_epoch}", torch.tensor(masked_img).permute(2, 0, 1))
            break
        return {'val_loss': loss, 'pred': out.cpu(), 'target': mask_batch.cpu()}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['pred'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        val_iou = iou(F.softmax(preds, dim=1), targets, num_classes=4, reduction='none')
        self.logger.experiment.add_scalars("val_iou", {cat_name: cat_iou for cat_name, cat_iou in
                                                       zip(self.train_dataset.dataset.categories, val_iou)},
                                           global_step=self.global_step)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=1e-5)
        # sch = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=3e-5, max_lr=self.learning_rate, step_size_up=2000,
        #                                         mode="triangular2")
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=2)
        return {
            'optimizer': opt,
            'lr_scheduler': sch,
            'monitor': 'val_loss',
        }

    def setup(self, stage: str) -> None:
        t_train = A.Compose([A.RandomCrop(2000, 3000, p=0.5), A.Resize(512, 768,  # 576, 864
                                                                       interpolation=cv2.INTER_AREA),
                             A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(p=0.5),
                             A.GridDistortion(p=0.2),
                             A.RandomBrightnessContrast((0, 0.5), (0, 0.5), p=0.5),
                             A.GaussNoise(p=0.3)])
        t_val = A.Compose([A.Resize(512, 768, interpolation=cv2.INTER_AREA),
                           # A.HorizontalFlip(),
                           # A.GridDistortion(p=0.2)
                           ])
        dataset = SDD_Dataset("data/SDD", preload=False, transform=None)

        n_val = int(len(dataset) * self.val_split_part)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_ds.dataset.set_transform(t_train)
        val_ds.dataset.set_transform(t_val)

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.unnormalizer = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


def train():
    model = SegModel()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints/deeplabv3_effnet-b2/',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=1, callbacks=[lr_monitor], checkpoint_callback=checkpoint_callback,
                         num_sanity_val_steps=-1,
                         log_every_n_steps=4,
                         # max_epochs=1,
                         # resume_from_checkpoint='checkpoints/deeplabv3/epoch=31-step=2879.ckpt'
                         )
    trainer.fit(model)


if __name__ == '__main__':
    train()
