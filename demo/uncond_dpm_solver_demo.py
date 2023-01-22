from schedule import NoiseScheduleVP, model_wrapper, DPM_Solver
import torch
from diffusers import DDIMPipeline


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
# TODO: I. define model
model_id = "google/ddpm-cifar10-32"
pipeline = DDIMPipeline.from_pretrained(model_id)
unet_fn = pipeline.unet.cuda()
model = lambda x,t:unet_fn(x,t).sample
# TODO: II. define model_kwargs
model_kwargs = {}
# TODO: III. define condition
condition = None # Nothing to do with uncond scenarios
# TODO: IV. define unconditional_condition
unconditional_condition = None # Nothing to do with uncond scenarios
# TODO: V. define guidance_scale
guidance_scale = 1. # Nothing to do with uncond scenarios
# TODO: VI. define classifier
classifier = None # Nothing to do with uncond scenarios
# TODO: VII. define classifier_kwargs
classifier_kwargs = {} # Nothing to do with uncond scenarios
# TODO: VIII. define betas
betas = pipeline.scheduler.betas


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
    guidance_type="uncond",
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
image_shape = (16, unet_fn.in_channels, unet_fn.sample_size, unet_fn.sample_size)
x_T  = torch.randn(image_shape).cuda()
x_sample = dpm_solver.sample(
    x_T,
    steps=100,
    order=1,
    skip_type="time_uniform",
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
    sub_image = sub_image.cpu().permute(1,2,0).numpy()
    numpy_to_pil(sub_image)[0].save(f"sample_{i}.png")