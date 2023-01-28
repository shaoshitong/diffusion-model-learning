
### Install
First create the conda environment:
```bash
conda create -n diffusion python=3.7.0
source activate diffsuion 
```

Second install torch:
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c https://mirrors.ustc.edu.cn/anaconda/cloud/pytorch/
```

Third install diffusers:
```bash
pip install --upgrade pip
pip install --upgrade diffusers[torch] transformers accelerate
pip install --upgrade "jax[cpu]"
```

### DEMO
demo explanation:
|  demo   | explanation  |
|  ----  | ----  |
| uncond_dpm_solver_demo.py  | Call dpm solver for generation by importing the unet used by `ddim` and `scheduler` in `diffusers`.  |
| classifier_guidance_dpm_solver_demo.py  | Call dpm solver for generation by importing the unet used by `ddim` and `scheduler` in `guided-diffusion`[link](https://github.com/openai/guided-diffusion). This is a classifier-guidance scenario.|
| classifier_guidance_dpm_solver_demo_2.py | Solve constant \mathcal{Z} is not a constant. Detail can be found in yuque [link](https://www.yuque.com/u29155493/ru454g/imahsg0kzzt5xo5a#Ug61t). | 
#### dpm solver in uncond scenario

1. load unet in code line 28-34.
2. load betas and noise scheduler in code line 48-50.
3. translate unet from diffusers to dpm solver format in code line 55-65.
4. initialize x_t in code line 84-85.
5. generate x_0 in code line 86-116.


#### dpm solver in classifier-guidance scenario

1. load unet in code line 30-51.
2. load classifier in code line 70-80.
3. load betas and noise scheduler in code line 85-98.
4. translate unet from diffusers to dpm solver format in code line 102-112.
5. initialize x_t in code line 132-148.
6. generate x_0 in code line 149-165.

#### dpm solver in classifier-guidance scenario (constant \mathcal{Z})

1. load unet in code line 30-53.
2. define \tau in code line 59.
3. load classifier in code line 70-80.
4. load betas and noise scheduler in code line 85-98.
5. translate unet from diffusers to dpm solver format in code line 102-112.
6. initialize x_t in code line 132-147.
7. generate x_0 in code line 148-165.