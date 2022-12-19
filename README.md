
# Adaptive Uncertainty Distribution in Deep Learning for Unsupervised Underwater Image Enhancement (UDnet) ([Paper](https://arxiv.org/))

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


The Pytorch Implementation of ''Adaptive Uncertainty Distribution in Deep Learning for Unsupervised Underwater Image Enhancement''. 

<div align=center><img src="model_utils/fig_1.png" height = "60%" width = "60%"/></div>

## Introduction
<div align=center><img src="model_utils/fig_8.png"></div>
We present a fully trainable framework to enhance underwater images without ground truth. 
We demonstrate that our proposed model outperforms ten popular underwater image enhancement methods on seven common  metrics, 
in both paired and unpaired settings.


## Requirement
In this project, we use Linux host with a single NVidia GeForce RTX 2080 Ti GPU with 11 GB of memory, Python 3.7, Pytorch 1.9.1.

## Running

### Inference

Download the pretrained model [pretrained model](https://cloudstor.aarnet.edu.au/plus/s/sO7yrv6LeEwna7K).

Check  the model and image pathes in Inference.py, and then run:

```
python Inference.py  
```

### Training

To train the model, you need to download Underwater Dataset [dataset](https://github.com/xahidbuffon/Awesome_Underwater_Datasets).

Check  the dataset path in Train.py, and then run:
```
python Train.py   
```

## Bibtex

If you find UDnet is useful in your research, please cite our paper:


```
{ Adaptive Uncertainty Distribution in Deep Learning for Unsupervised Underwater Image Enhancement	
https://github.com/alzayats/UDnet
}
```

## License
The code is made available for academic research purpose only. This project is open sourced under MIT license.

## Credit
https://github.com/zhenqifu/PUIE-Net \
https://github.com/deepxzy/USLN \
https://github.com/xueleichen/PyTorch-Underwater-Image-Enhancement
