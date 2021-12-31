"""
这是转为onnx的模块
"""

import argparse
import warnings

# warnings.filterwarnings("ignore")

import torch

from retinaface import RetinaFace

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Retinaface mobilenetv2")
    parser.add_argument('--input_size', default=[840, 840])
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--onnx_path',default="test.onnx")
    parser.add_argument('--checkpoint_dir', default="lightning_logs/2021-12-21-17:54:43/checkpoints/epoch=0-val_loss=36.8446.ckpt")
    opt = parser.parse_args()

    model = RetinaFace(
        input_size=opt.input_size,
        batch_size=opt.batch_size,
    )
    model.load_from_checkpoint(opt.checkpoint_dir)
    input_sample = torch.randn((opt.batch_size,3,opt.input_size[0],opt.input_size[1]))
    model.to_onnx(opt.onnx_path, input_sample, export_params=True,)