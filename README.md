# Dispersion Reduction Attack


Code for CVPR2020 paper [Enhancing Cross-task Black-Box Transferability of Adversarial Examples with Dispersion Reduction](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Enhancing_Cross-Task_Black-Box_Transferability_of_Adversarial_Examples_With_Dispersion_Reduction_CVPR_2020_paper.pdf).

### Docker Quick Start
Build the docker image and all dependencies will be installed automatically.
```
docker pull erbloo/baidu_aisec:yantao_v1.0.0
```
Get access to the container and clone the repository into it using command:
```
docker exec -ti |container_name| bash
git clone https://github.com/erbloo/dr_cvpr20.git
```

### Experimental Data
Experimental data is included in [dr_images_cvpr20](https://github.com/erbloo/dr_images_cvpr20).


### Please find the attack example in `example.ipynb`


### Citing this work
If you find this work is useful in your research, please consider citing:
```
    @inproceedings{lu2020enhancing,
      title={Enhancing cross-task black-box transferability of adversarial examples with dispersion reduction},
      author={Lu, Yantao and Jia, Yunhan and Wang, Jianyu and Li, Bai and Chai, Weiheng and Carin, Lawrence and Velipasalar, Senem},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={940--949},
      year={2020}
    }
```
