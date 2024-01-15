# NeRF-VPT

# Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.1** 

## Software

* Clone this repo by `git clone git@github.com:Freedomcls/NeRF-VPT.git`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n nerf_vpt python=3.6` to create a conda environment and activate it by `conda activate nerf_vpt`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    * Install `torchsearchsorted` by `cd torchsearchsorted` then `pip install .`
    
# Training

### Data download
Download data from here.(Baidu Netdisk link:`https://pan.baidu.com/s/1Z-hyU7iLW4imccumzpMTGg?pwd=1234`, and extraction code: `1234`)
Unzip the data and place it in the current directory.


### Training Stage 0
First, it is necessary to train NeRF as Stage 0.  
```
sh run_stage0.sh
```


# Inference
Render images of the training, validation, and test datasets from the corresponding viewpoints as view prompts.  

```
sh run_stage0_render.sh
```

This script will output the PSNR metric of the model on the dataset, and the rendered images will be saved in  `results/replica/replica_stage0_xx`.

### Training Stage 1~N
Proceed with training the NeRF-VPT model next.  

```
sh run_multi_stage.sh
```
  
# Inference NeRF-VPT
Render images of the training, validation, and test datasets from the corresponding viewpoints as view prompts in Stage 1~N.  

```
sh run_mul_stage_render.sh
```
You can continue the next stage of training using `run_multi_stage.sh`, but you would need to modify the corresponding parameters within it.
For example, if training stage2, you need to modify `--exp_name replica_stage1` to `--exp_name replica_stage2`.
The same in inference phase, you need to modify:
1.all the `--scene_name replica_stage1_xxx` to `--scene_name replica_stage2_xxx`;
2.all the `--ckpt_path ckpts/replica_stage1/xxx` to `--ckpt_path ckpts/replica_stage2/xxx`;
3.all the `--prompt_path results/replica/replica_stage0` to `--prompt_path results/replica/replica_stage1`.

Also, the script will output the PSNR metric of the model.