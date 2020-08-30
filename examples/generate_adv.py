""" Script for generating adversarial exmaples 
    and visualize intermediate layers.
"""
import sys
sys.path.append('/home/yilan/yantao/workspace/projects/dr_cvpr20/')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision.models as models
import torch
from tqdm import tqdm

from attacks.dispersion import DispersionAttack_gpu
from models.vgg import Vgg16
from utils.image_utils import load_image
from utils.torch_utils import numpy_to_variable

import pdb


dataset_dir = "images"
outout_dir = "images_adv"
images_name = os.listdir(dataset_dir)

model = Vgg16()
internal = [i for i in range(29)]

attack = DispersionAttack_gpu(model, epsilon=16./255, step_size=1./255, steps=1000)

for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_img_name_noext = os.path.splitext(temp_image_name)[0]
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    img_size = image_np.shape[1:]
    image_var = numpy_to_variable(image_np)
    features_ori, _ = model.prediction(image_var, internal=internal)
    adv_var = attack(image_var, 
                 attack_layer_idx=14, 
                 internal=internal, 
                )
    features_adv, _ = model.prediction(adv_var, internal=internal)

    img_ori_pil = Image.fromarray((np.transpose(image_np, (1, 2, 0)) * 255).astype(np.uint8))
    img_ori_pil.save(os.path.join(outout_dir, '{}_img_ori.png').format(temp_img_name_noext))
    img_adv_pil = Image.fromarray((np.transpose(adv_var[0].cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8))
    img_adv_pil.save(os.path.join(outout_dir, '{}_img_adv.png').format(temp_img_name_noext))


