"""
这是定义retinaface的脚本
"""
import sys
import collections

sys.path.append("./")

import timm
import torch

from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from loss import MultiBoxLoss
from models import Anchors, ClassHead, BboxHead, LandmarkHead, SSH
from data import DataGenerator, detection_collate


class RetinaFace(pl.LightningModule):

    def __init__(self, lr=1, backbone=None,input_size=None, num_workers=None, batch_size=None, data_dir=None) -> None:
        super().__init__()

        # 定义backbone
        if backbone == "mobilenetv2_100":
            self.backbone = timm.create_model(backbone, features_only=True, pretrained=True)
            self.fpn = FeaturePyramidNetwork([32, 96, 320], 64, extra_blocks=LastLevelMaxPool())
        elif backbone == "tf_mobilenetv3_small_minimal_100":
            self.backbone = timm.create_model(backbone, features_only=True, pretrained=True)
            self.fpn = FeaturePyramidNetwork([24, 48, 576], 64, extra_blocks=LastLevelMaxPool())
        elif backbone == "efficientnet_b3":
            self.backbone = timm.create_model(backbone, features_only=True, pretrained=True)
            self.fpn = FeaturePyramidNetwork([48, 136, 384], 64, extra_blocks=LastLevelMaxPool())
        elif backbone == "efficientnet_es":
            self.backbone = timm.create_model(backbone, features_only=True, pretrained=True)
            self.fpn = FeaturePyramidNetwork([48, 144, 192], 64, extra_blocks=LastLevelMaxPool())

        # 定义fpn与sshd
        self.ssh1 = SSH(64, 64)
        self.ssh2 = SSH(64, 64)
        self.ssh3 = SSH(64, 64)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=64)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=64)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=64)

        # 初始化损失函数
        self.loss_func = MultiBoxLoss(2, 0.35, 7, [0.1, 0.2])
        self.anchors = Anchors(image_size=input_size).get_anchors().cuda()

        # 初始化超参数
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.input_size = input_size

        # 保存超参数
        self.save_hyperparameters()

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):

        out = collections.OrderedDict()
        for i,feature in enumerate(self.backbone(inputs)[-3:]):
            out[i] = feature
        
        fpn = self.fpn.forward(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)
        output = (bbox_regressions, classifications, ldm_regressions)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "monitor": "val_loss"}}

    def prepare_data(self):
        self.train_data = DataGenerator(self.data_dir + "train/train.txt", self.input_size)
        self.val_data = DataGenerator(self.data_dir + "train/val.txt", self.input_size)

    def train_dataloader(self):

        return DataLoader(self.train_data,
                          shuffle=True,
                          drop_last=True,
                          batch_size=self.batch_size,
                          collate_fn=detection_collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val_data,
                          shuffle=True,
                          drop_last=True,
                          batch_size=self.batch_size,
                          collate_fn=detection_collate,
                          num_workers=self.num_workers)

    def on_epoch_start(self):
        self.val_r_loss = []
        self.val_c_loss = []
        self.val_landm_loss = []
        self.val_loss = []

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if not len(images) == 0:
            out = self.forward(images)
            self.train_r_loss, self.train_c_loss, self.train_landm_loss = self.loss_func(out, self.anchors, labels)
            self.train_loss = 2 * self.train_r_loss + self.train_c_loss + self.train_landm_loss
            self.log_dict({
                "Regression Loss": self.train_r_loss,
                "Conf Loss": self.train_c_loss,
                "LandMark Loss": self.train_landm_loss
            },
                          prog_bar=True,
                          logger=False)

            return {"loss": self.train_loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if not len(images) == 0:
            out = self.forward(images)
            r_loss, c_loss, landm_loss = self.loss_func(out, self.anchors, labels)
            loss = 2 * r_loss + c_loss + landm_loss
            self.val_r_loss.append(r_loss)
            self.val_c_loss.append(c_loss)
            self.val_landm_loss.append(landm_loss)
            self.val_loss.append(loss)
            self.log("val_loss", loss, logger=False)


    def on_train_epoch_end(self):
        # 记录当前epoch的train损失
        self.logger[0].experiment.add_scalar("loss", self.train_loss, global_step=self.current_epoch)
        self.logger[0].experiment.add_scalar("Regression Loss", self.train_r_loss, global_step=self.current_epoch)
        self.logger[0].experiment.add_scalar("Conf Loss", self.train_c_loss, global_step=self.current_epoch)
        self.logger[0].experiment.add_scalar("LandMark Loss", self.train_landm_loss, global_step=self.current_epoch)

    def on_validation_epoch_end(self):
        # 记录当前epoch的val的平均损失
        self.logger[1].experiment.add_scalar("loss", torch.mean(torch.stack(self.val_loss)), global_step=self.current_epoch)
        self.logger[1].experiment.add_scalar("Regression Loss", torch.mean(torch.stack(self.val_r_loss)), global_step=self.current_epoch)
        self.logger[1].experiment.add_scalar("Conf Loss", torch.mean(torch.stack(self.val_c_loss)), global_step=self.current_epoch)
        self.logger[1].experiment.add_scalar("LandMark Loss", torch.mean(torch.stack(self.val_landm_loss)), global_step=self.current_epoch)



        

