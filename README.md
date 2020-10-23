# DCGAN implementation using Pytorch
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python3](https://img.shields.io/badge/python->=3-green.svg)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[Paper link](https://arxiv.org/pdf/1511.06434.pdf)

Implementation of Deep Convolutional GAN (DCGAN) from scratch using Pytorch.

## Libraries required

* pytorch
* matplotlib
* tqdm

## Dataset
 MNIST
 (dataset will be automatically dowloaded when using this repository)

 ## Training

 To train DCGAN model just run

**For CPU**
```
python train.py --epochs=10 --batch_size=32 --device="cpu"
```

**For GPU**
```
python train.py --epochs=10 --batch_size=64 --device="gpu"
```

# Output Images during Training DCGAN

When step = 500


<p align="center">
<img src="https://github.com/AbhishekSalian/DCGAN-using-Pytorch/blob/main/images/generated_step500.png?raw=true"></a>
</p>


When step = 1000
<p align="center">
<img src="https://github.com/AbhishekSalian/DCGAN-using-Pytorch/blob/main/images/generated_step1000.png?raw=true"></a>
</p>


When step = 2000

<p align="center">
<img src="https://github.com/AbhishekSalian/DCGAN-using-Pytorch/blob/main/images/generated_step2000.png?raw=true"></a>
</p>

When steps = 23000

<p align="center">
<img src="https://github.com/AbhishekSalian/DCGAN-using-Pytorch/blob/main/images/generated_step23k.png?raw=true"></a>
</p>


