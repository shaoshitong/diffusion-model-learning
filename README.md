# Diffusion Model Learning
Here I will add some of my own notes, insights and demos about learning the diffusion model.

## Docs
In the docs folder, I have my notes on diffusion model, including papers and good blogs, ODE review, some derivation of SDE in diffusion model, and a series of understanding of diffusion model code. However, since the pdf exported by `yuque` is really unforgiving, there may be a lot of typographical problems. However, I have opened access to the connection at: [yuque link](https://www.yuque.com/u29155493/ru454g), password is `vghi`.

## Papers

Papers are list in [paper](./papers//README.md):

<!-- 
- Chung H, Ryu D, McCann M T, et al. Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models[J]. arXiv preprint arXiv:2211.10655, 2022.[https://arxiv.org/abs/2211.10655](https://arxiv.org/abs/2211.10655)
- Ajay A, Du Y, Gupta A, et al. Is Conditional Generative Modeling all you need for Decision-Making?[J]. arXiv preprint arXiv:2211.15657, 2022.[https://arxiv.org/abs/2211.15657](https://arxiv.org/abs/2211.15657)
- Ho J, Salimans T. Classifier-free diffusion guidance[J]. arXiv preprint arXiv:2207.12598, 2022.[https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598)
- Poole B, Jain A, Barron J T, et al. Dreamfusion: Text-to-3d using 2d diffusion[J]. arXiv preprint arXiv:2209.14988, 2022.[https://arxiv.org/abs/2209.14988](https://arxiv.org/abs/2209.14988)
- Hertz A, Mokady R, Tenenbaum J, et al. Prompt-to-prompt image editing with cross attention control[J]. arXiv preprint arXiv:2208.01626, 2022.[https://arxiv.org/abs/2208.01626](https://arxiv.org/abs/2208.01626)
- Couairon G, Verbeek J, Schwenk H, et al. Diffedit: Diffusion-based semantic image editing with mask guidance[J]. arXiv preprint arXiv:2210.11427, 2022.[https://arxiv.org/abs/2210.11427](https://arxiv.org/abs/2210.11427)
- Dhariwal P, Nichol A. Diffusion models beat gans on image synthesis[J]. Advances in Neural Information Processing Systems, 2021, 34: 8780-8794.[https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)
- Liu X, Park D H, Azadi S, et al. More control for free! image synthesis with semantic diffusion guidance[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023: 289-299.[https://openaccess.thecvf.com/content/WACV2023/html/Liu_More_Control_for_Free_Image_Synthesis_With_Semantic_Diffusion_Guidance_WACV_2023_paper.html](https://openaccess.thecvf.com/content/WACV2023/html/Liu_More_Control_for_Free_Image_Synthesis_With_Semantic_Diffusion_Guidance_WACV_2023_paper.html)
- Rasul K, Seward C, Schuster I, et al. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting[C]//International Conference on Machine Learning. PMLR, 2021: 8857-8868.[http://proceedings.mlr.press/v139/rasul21a.html](http://proceedings.mlr.press/v139/rasul21a.html)
- Nie W, Guo B, Huang Y, et al. Diffusion models for adversarial purification[J]. arXiv preprint arXiv:2205.07460, 2022.[https://arxiv.org/abs/2205.07460](https://arxiv.org/abs/2205.07460)
<a name="A6T1n"></a> -->

## Demo
In the demo folder, a series of sample demos exists related to dpm solver/dpm solver++. Detail can be found in [readme.md](./demo/README.md).

## Model
UNet copy from [ADM](https://github.com/openai/guided-diffusion).

## Schedule
Scheduler copy from [DPM Solver/DPM Solver++](https://github.com/LuChengTHU/dpm-solver)