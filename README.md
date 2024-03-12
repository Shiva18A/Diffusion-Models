# Diffusion-Models
Diffusion models from scratch
1) Results
2) Architecture
3) Additional Experiments

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


 **3) Additional Experiments** - 40 Epochs
 
  [Notebook 2](diffusion_experiments_wandb.ipynb/)
 
 Used **WandB** (Weights & Biases) for the experiments. Below are the experiment results for only the above 3 uploaded model, namely a **Conv UNet**, **Residual UNet** (see 
 diagram), Residual Attention Unet ([Notebook 1](DDPM_diffusion_model_scratch.ipynb/)). All the models were trained for just **40 epochs** on P100 GPU. As we can see the 
 Conv UNet fails to generate any images. This is due to the fact that model is very deep with large channel sizes and has no residual connections. Skip connections in UNet 
 were not sufficient in this case. Using a smaller model works but output image generation quality is severely impacted. Currently working on some other ways to make this 
 work. In the same model, adding residual blocks (**see diagram**) works wonders and model's performance and image generation quality leaps significantly. We can see the 
 result in below displayed images. Lastly Residual Attention UNet performs the best and gives the best results. The difference is marginal when training for 40 Epochs but as 
 we reach 100 Epochs or so we can see some differences. Objects are more clearly defined rather than diffused looking, as compared to Residual UNet.

 <br>
 
![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/W%26B%20Chart%203_11_2024%2C%206_47_18%20PM.png?raw=true "results")

 **Residual Attention UNet** 
 
 ![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/res_att_unet_40.png?raw=true "results")

 <br>
 
 **Residual UNet**
 
 ![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/res_unet_40.png?raw=true "results")

 <br>
 
 **Conv UNet**
 
 ![image](https://github.com/Shiva18A/Diffusion-Models/blob/main/imgs/conv_unet_40.png?raw=true "results")
