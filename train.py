"""
这是用于训练retinaface的模块
"""

import sys
import warnings
import time

sys.path.append("./")
warnings.filterwarnings("ignore")

import argparse
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from retinaface import RetinaFace


# 定义训练进度条样式
class LitProgressBar(RichProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self._leave = True

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Retinaface")
    parser.add_argument('--backbone',
                        default="efficientnet_es",
                        help="choose from mobilenetv2_100;tf_mobilenetv3_small_minimal_100;efficientnet_b3;efficientnet_es")
    parser.add_argument('--auto_batch_size', default=False)
    parser.add_argument('--auto_lr', default=False)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--input_size', default=[840, 840])
    parser.add_argument('--batch_size', default=12)
    parser.add_argument('--num_workers', default=6)
    parser.add_argument('--lr', default=1)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--data_dir', default="../dataset/widerface/")
    parser.add_argument('--checkpoint_dir', default=None)
    opt = parser.parse_args()

    # 实例化模型
    Train_model = RetinaFace(backbone=opt.backbone,
                             lr=opt.lr,
                             input_size=opt.input_size,
                             num_workers=opt.num_workers,
                             batch_size=opt.batch_size,
                             data_dir=opt.data_dir)

    # 初始化训练器
    lr_monitor = LearningRateMonitor(logging_interval='epoch',)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=15)
    current_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))

    # 定义记录器及检查点

    if opt.checkpoint_dir:
        logger0 = TensorBoardLogger(save_dir=opt.checkpoint_dir.split("/checkpoints")[0], name="train", version="")
        logger1 = TensorBoardLogger(save_dir=opt.checkpoint_dir.split("/checkpoints")[0], name="val", version="")
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          save_top_k=5,
                                          filename='{epoch}-{val_loss:.4f}',
                                          dirpath=opt.checkpoint_dir.split("/epoch")[0])
    else:
        logger0 = TensorBoardLogger(save_dir="lightning_logs/" + current_time, name="train", version="")
        logger1 = TensorBoardLogger(save_dir="lightning_logs/" + current_time, name="val", version="")
        checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          save_top_k=5,
                                          filename='{epoch}-{val_loss:.4f}',
                                          dirpath="lightning_logs/" + current_time + "/checkpoints")

    # 定义进度条
    bar = LitProgressBar()

    # 训练前的debug测试
    if opt.debug:
        trainer = pl.Trainer(gpus=opt.gpus, callbacks=[early_stop_callback, lr_monitor], fast_dev_run=True)
        trainer.fit(Train_model)

    # find最佳batch_size
    if opt.auto_batch_size:
        trainer = pl.Trainer(gpus=opt.gpus,
                             auto_scale_batch_size="power",
                             default_root_dir="lr_batch_size_logs",
                             callbacks=[early_stop_callback, lr_monitor])
        Train_model.batch_size = trainer.tuner.scale_batch_size(Train_model)

    # find最佳学习率
    if opt.auto_lr:
        trainer = pl.Trainer(default_root_dir="lr_batch_size_logs",
                             gpus=opt.gpus,
                             auto_lr_find=True,
                             callbacks=[early_stop_callback, lr_monitor])
        lr_finder = trainer.tuner.lr_find(Train_model)
        Train_model.lr = lr_finder.suggestion()


    # 定义完整训练器
    trainer = pl.Trainer(
        gpus=opt.gpus,
        max_epochs=150,
        callbacks=[early_stop_callback, lr_monitor, bar, checkpoint_callback],
        logger=[logger0, logger1],
        precision=16,
        # limit_train_batches=0.01,
        # limit_val_batches=0.01,
    )

    # 加载checkpoint并开始训练
    trainer.fit(Train_model,ckpt_path=opt.checkpoint_dir)


