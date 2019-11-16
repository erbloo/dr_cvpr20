import numpy as np 
import os
import shutil
import glob

import pdb

def generate_imagenet_testdata(input_dir, output_dir, num_imgs=100):
    class_dir_list = os.listdir(input_dir)
    selected_image_names = np.random.choice(class_dir_list, num_imgs)
    for temp_image_name in selected_image_names:
        shutil.copyfile(os.path.join(input_dir, temp_image_name), os.path.join(output_dir, temp_image_name))

def generate_FDDB_testdata(input_dir, output_dir, num_imgs=100):
    files_path = glob.glob(input_dir + '*/*/*/big/*.jpg')
    selected_image_paths = np.random.choice(files_path, num_imgs)
    for idx, temp_path in enumerate(selected_image_paths):
        shutil.copyfile(temp_path, os.path.join(output_dir, 'img_choice_{0:03d}.jpg'.format(idx)))
    

if __name__ == "__main__":
    generate_imagenet_testdata("/home/yantao/datasets/ILSVRC/Data/DET/test/", "/home/yantao/datasets/imagenet_100image/")
    #generate_FDDB_testdata("/home/yantao/datasets/FDDB_face/images/", "/home/yantao/datasets/FDDB_100image/")