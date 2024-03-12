# Diffusion-Models
Diffusion models from scratch
1) Results
2) Architecture
3) Experiments

**1) Results**

 **Forward Diffusion Process**

![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/forward_process.jpg?raw=true "results")

 **Reverse Diffusion Process**

Below are some of the results showing **Progressive Generation** on **CIFAR-10** Dataset. The model was trained for **100 epochs** on P100 GPU. The model used was **Residual-Attention-UNet**. We can see some of the CIFAR-10 classes here such Ship, Dog, Frog etc.

![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/reverse_process.jpg?raw=true "results")

Some **Generated Samples**

![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/sampled_images.png?raw=true "results")


**2) Architecture**

 DDPM paper uses model similar to Residual Attention UNet. Implemented multiple different architectures, but here have uploaded 3 architectures namely **Convolutional 
 UNet** (Vanilla with tweeks!!) , **Residual UNet** and **Residual Attention UNet**. Below is the diagram i have drawn for Residual UNet. Residual Attention UNet diagram is **work in progress :)**

 ![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/resunet_components.jpg?raw=true "architecture")
 ![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/resunet.png?raw=true "architecture")


 **3) Experiments**
