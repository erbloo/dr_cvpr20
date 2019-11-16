import sys
sys.path.append('/home/yantao/workspace/projects/baidu/CVPR2019_workshop')
import torchvision.models as models
import torch
from PIL import Image
from utils.image_utils import load_image
from utils.torch_utils import numpy_to_variable
import numpy as np
import torchvision
from tqdm import tqdm
import os
import shutil            


dataset_dir = "./"

dataset_dir_ori = os.path.join(dataset_dir, 'images')
dataset_dir_adv = os.path.join(dataset_dir, 'images_adv')

images_name = os.listdir(dataset_dir_ori)

test_model = torchvision.models.densenet121(pretrained='imagenet').cuda().eval()

for idx, temp_image_name in enumerate(tqdm(images_name)):
    total_samples = len(images_name)
    ori_img_path = os.path.join(dataset_dir_ori, temp_image_name)
    adv_img_path = os.path.join(dataset_dir_adv, temp_image_name)

    image_ori_np = load_image(data_format='channels_first', shape=(224, 224), bounds=(0, 1), abs_path=True, fpath=ori_img_path)
    image_ori_var = numpy_to_variable(image_ori_np)
    gt_out = test_model(image_ori_var).detach().cpu().numpy()
    gt_label = np.argmax(gt_out)
    
    image_adv_np = load_image(data_format='channels_first', shape=(224, 224), bounds=(0, 1), abs_path=True, fpath=adv_img_path)
    image_adv_var = numpy_to_variable(image_adv_np)
    pd_out = test_model(image_adv_var).detach().cpu().numpy()
    pd_label = np.argmax(pd_out)

    print('idx: ', idx)
    print('ground truth: ', gt_label)
    print('prediction: ', pd_label)