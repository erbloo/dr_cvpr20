import sys
sys.path.append('/home/yantao/workspace/projects/baidu/CVPR2019_workshop')
import torchvision.models as models
import torch
from tqdm import tqdm
from PIL import Image
from utils.image_utils import load_image
from utils.torch_utils import numpy_to_variable
from models.vgg import Vgg16
from attacks.dispersion import DispersionAttack_gpu
import numpy as np
import os


dataset_dir = "images"
outout_dir = "images_adv"
images_name = os.listdir(dataset_dir)

model = Vgg16()
internal = [i for i in range(29)]

attack = DispersionAttack_gpu(model, epsilon=16./255, step_size=1./255, steps=1000)

for idx, temp_image_name in enumerate(tqdm(images_name)):
    temp_image_path = os.path.join(dataset_dir, temp_image_name)
    image_np = load_image(data_format='channels_first', abs_path=True, fpath=temp_image_path)
    image = numpy_to_variable(image_np)

    adv = attack(image, 
                 attack_layer_idx=14, 
                 internal=internal, 
                )
    adv_image = (np.transpose(adv[0].cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    Image.fromarray(adv_image).save(os.path.join(outout_dir, temp_image_name))
    