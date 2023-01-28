from schedule import NoiseScheduleVP, model_wrapper, DPM_Solver

## You need to firstly define your model and the extra inputs of your model,
## And initialize an `x_T` from the standard normal distribution.
## `model` has the format: model(x_t, t_input, **model_kwargs).
## If your model has no extra inputs, just let model_kwargs = {}.

## If you use discrete-time DPMs, you need to further define the
## beta arrays for the noise schedule.

## For classifier guidance, you need to further define a classifier function,
## a guidance scale and a condition variable.


# model = ....
# model_kwargs = {...}
# x_T = ...
# condition = ...
# betas = ....
# classifier = ...
# classifier_kwargs = {...}
# guidance_scale = ...

# TODO: Definition
import torch
import numpy as np
BATCHSIZE = 4
label = torch.ones(BATCHSIZE).long().cuda() * 30

# TODO: I. define model
from model import create_uncond_unet, create_classifier_unet
from utils import model_defaults
import os, sys

model_hyperparameter = model_defaults()
model_hyperparameter.update(dict(learn_sigma=True))
_model_fn = create_uncond_unet(
    **model_hyperparameter
)
if not os.path.exists("256x256_diffusion.pt"):
    os.system("wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt")
model_path = "256x256_diffusion.pt"
_model_fn.load_state_dict(torch.load(model_path, map_location="cpu"))
_model_fn.cuda()

@torch.no_grad()
def model(x, t, **kwargs):
    B, C = x.shape[:2]
    model_output = _model_fn(x, t, **kwargs)
    model_output, model_var_values = torch.split(model_output, C, dim=1)
    return model_output


# TODO: II. define model_kwargs
model_kwargs = {"y": label}

# TODO: III. define condition
import torch.nn.functional as F
tau = 1.0
condition = lambda x, y=label: F.log_softmax(x/tau, dim=-1)[range(x.shape[0]), y.view(-1)]

# TODO: IV. define unconditional_condition
unconditional_condition = None  # Nothing to do with guidance-classifier scenarios

# TODO: V. define guidance_scale
from utils import classifier_defaults

guidance_scale = 1.0  # default

# TODO: VI. define classifier
classifier_hyperparameter = classifier_defaults()
classifier_fn = create_classifier_unet(
    **classifier_hyperparameter
)
if not os.path.exists("256x256_classifier.pt"):
    os.system("wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt")
classifier_path = "256x256_classifier.pt"
classifier_fn.load_state_dict(torch.load(classifier_path, map_location="cpu"))
classifier_fn.cuda()
classifier = lambda x, t, cond: cond(classifier_fn(x, t))
# TODO: VII. define classifier_kwargs
classifier_kwargs = {}  # Nothing to do with uncond scenarios

# TODO: VIII. define betas
from utils import get_named_beta_schedule
betas = torch.from_numpy(get_named_beta_schedule("linear", 1000)).cuda()
# alphas = 1.0 - betas
# last_alpha_cumprod = 1.0
# new_betas = []
# timestep_map = []
# for i, alpha_cumprod in enumerate(torch.cumprod(alphas, dim=0)):
#     if i in [j for j in range(20)]:
#         new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
#         last_alpha_cumprod = alpha_cumprod
#         timestep_map.append(i)
# betas = torch.Tensor(new_betas)
## 1. Define the noise schedule.
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
## 2. Convert your discrete-time `model` to the continuous-time
## noise prediction model. Here is an example for a diffusion model
## `model` with the noise prediction type ("noise") .
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type="noise",  # or "x_start" or "v" or "score"
    model_kwargs=model_kwargs,
    guidance_type="classifier",
    condition=condition,
    guidance_scale=guidance_scale,
    classifier_fn=classifier,
    classifier_kwargs=classifier_kwargs,
)

## 3. Define dpm-solver and sample by multistep DPM-Solver.
## (We recommend multistep DPM-Solver for conditional sampling)
## You can adjust the `steps` to balance the computation
## costs and the sample quality.

dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

## If the DPM is defined on pixel-space images, you can further
## set `correcting_x0_fn="dynamic_thresholding"`. e.g.:

# dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
#   correcting_x0_fn="dynamic_thresholding")


## You can use steps = 10, 12, 15, 20, 25, 50, 100.
## Empirically, we find that steps in [10, 20] can generate quite good samples.
## And steps = 20 can almost converge.

image_shape = (BATCHSIZE, 3, 256, 256)
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
set_random_seed(0)
if not os.path.exists("x_T.pt"):
    x_T = torch.randn(image_shape).cuda()
    torch.save(x_T,"x_T.pt")
else:
    x_T  = torch.load("x_T.pt")
x_sample = dpm_solver.sample(
    x_T,
    steps=20,
    order=3,
    skip_type="logSNR",
    method="multistep",
)

from PIL import Image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


for i in range(x_sample.shape[0]):
    sub_image = x_sample[i]
    sub_image = (sub_image / 2 + 0.5).clamp(0, 1)
    sub_image = sub_image.cpu().permute(1, 2, 0).numpy()
    numpy_to_pil(sub_image)[0].save(f"sample_tau_is_{tau}_{i}.png")
