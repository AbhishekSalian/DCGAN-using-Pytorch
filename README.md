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

When steps is 500

<<<<<<< HEAD
=======
![image1]("./images/generated_step500.png")
>>>>>>> 3ea165206d4c9e8b76901d4b3daa22c461dbd091

<p align="center">
<img src="../DCGAN-using-Pytorch/images/generated_step500.png"></a>
</p>


<<<<<<< HEAD
When steps is 1000
<p align="center">
<img src="../DCGAN-using-Pytorch/images/generated_step1000.png"></a>
</p>

When steps is 1500

<p align="center">
<img src="../DCGAN-using-Pytorch/images/generated_step1500.png"></a>
</p>

When steps is 2000

<p align="center">
<img src="../DCGAN-using-Pytorch/images/generated_step2000.png"></a>
</p>

When steps is 23k

<p align="center">
<img src="../DCGAN-using-Pytorch/images/generated_step23k.png"></a>
</p>
=======
![image1]("./images/generated_step1000.png")

When steps is 1500

![image1]("./images/generated_step1500.png")

When steps is 2000

![image1]("./images/generated_step2000.png")

When steps is 23k

![image1]("./images/generated_step23k.png")
>>>>>>> 3ea165206d4c9e8b76901d4b3daa22c461dbd091


