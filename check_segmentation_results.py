import argparse
import torchvision.transforms as transforms
from skimage import io
import numpy as np
import pathlib
import torch
import matplotlib.pyplot as plt
from ImFusionScripts.fix_pythonpath import *
from pytorch_lightning.utilities.cloud_io import load as pl_load
import models
import modules
import imfusion

imfusion.init()
import ImFusionPlugins

file = imfusion.open("D:\\Maria\\DataBases\\SpineIFL\\VertebraeDb\\sweeps\\MariaT.imf")[1]
imfusion.executeAlgorithm('Load Labels', [file])

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

data_path = "D:\\Maria\\DataBases\\SpineIFL\\VertebraeDb\\sweeps\\MariaT.imf"
ckpt_path = "D:/NAS/output/43005/43005/checkpoints/" \
            "43005_BoneSegmentation_BoneSegmentationModule_unet_2d_UNet2D-epoch=106-val_loss=0.04.ckpt"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--bilinear', type=int, default=True)

# loading model checkpoints
hparams = parser.parse_args()
model_handler = models.UNet2D(hparams)
module = modules.BoneSegmentation(hparams, model_handler)
ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
module.load_state_dict(ckpt['state_dict'])

# defining input transform
transform = transforms.ToTensor()

# Loading input sweep
shared_image = imfusion.open(data_path)
sweep = shared_image[0]

out_label = np.zeros([777, 516, 544])


for i, bmode in enumerate(sweep):

    if i>20:
        continue

    bmode_array = np.squeeze(np.array(bmode))

    arr = np.array(bmode)
    input_image = transform(bmode_array)
    input_batch = input_image.unsqueeze(0)

    module.eval()
    output = module.forward(input_batch)
    output = torch.sigmoid(output)

    labels = np.squeeze(output.detach().to("cpu").numpy())
    out_label[i, :, :] = labels
np.save("tmp.npy", out_label)

