# Diffusion Model Learning
Here I will add some of my own notes, insights and demos about learning the diffusion model.

## docs
In the docs folder, I have my notes on diffusion model, including papers and good blogs, ODE review, some derivation of SDE in diffusion model, and a series of understanding of diffusion model code. However, since the pdf exported by `yuque` is really unforgiving, there may be a lot of typographical problems. However, I have opened access to the connection at: [yuque link](https://www.yuque.com/u29155493/ru454g), password is `vghi`.

---

### Papers are list in [paper/README.md](./papers/README.md):

#### Diffusion Survey:

- Cao H, Tan C, Gao Z, et al. A survey on generative diffusion model[J]. arXiv preprint arXiv:2209.02646, 2022.[https://arxiv.org/abs/2209.02646](https://arxiv.org/abs/2209.02646)
- Kazerouni A, Aghdam E K, Heidari M, et al. Diffusion models for medical image analysis: A comprehensive survey[J]. arXiv preprint arXiv:2211.07804, 2022.[https://arxiv.org/abs/2211.07804](https://arxiv.org/abs/2211.07804)
<a name="tRrXx"></a>
#### Diffusion Theory:

- Mittal G, Engel J, Hawthorne C, et al. Symbolic music generation with diffusion models[J]. arXiv preprint arXiv:2103.16091, 2021.[https://arxiv.org/abs/2103.16091](https://arxiv.org/abs/2103.16091)
- Lee S, Kim B, Ye J C. Minimizing Trajectory Curvature of ODE-based Generative Models[J]. arXiv preprint arXiv:2301.12003, 2023.[https://arxiv.org/abs/2301.12003](https://arxiv.org/abs/2301.12003)
- Lee H, Lu J, Tan Y. Convergence of score-based generative modeling for general data distributions[C]//International Conference on Algorithmic Learning Theory. PMLR, 2023: 946-985.[https://proceedings.mlr.press/v201/lee23a.html](https://proceedings.mlr.press/v201/lee23a.html)
- Chen S, Chewi S, Li J, et al. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions[J]. arXiv preprint arXiv:2209.11215, 2022.[https://arxiv.org/abs/2209.11215](https://arxiv.org/abs/2209.11215)
- Lipman Y, Chen R T Q, Ben-Hamu H, et al. Flow matching for generative modeling[J]. arXiv preprint arXiv:2210.02747, 2022.[https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- Liu X, Gong C, Liu Q. Flow straight and fast: Learning to generate and transfer data with rectified flow[J]. arXiv preprint arXiv:2209.03003, 2022.[https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003)
- Song J, Meng C, Ermon S. Denoising diffusion implicit models[J]. arXiv preprint arXiv:2010.02502, 2020.[https://arxiv.org/abs/2010.02502](https://arxiv.org/abs/2010.02502)
- Bao F, Li C, Sun J, et al. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models[J]. arXiv preprint arXiv:2206.07309, 2022.[https://arxiv.org/abs/2206.07309](https://arxiv.org/abs/2206.07309)
- Bao F, Li C, Zhu J, et al. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models[J]. arXiv preprint arXiv:2201.06503, 2022.[https://arxiv.org/abs/2201.06503](https://arxiv.org/abs/2201.06503)
- Kingma D, Salimans T, Poole B, et al. Variational diffusion models[J]. Advances in neural information processing systems, 2021, 34: 21696-21707.[https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html)
- Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International Conference on Machine Learning. PMLR, 2015: 2256-2265.[http://proceedings.mlr.press/v37/sohl-dickstein15.html](http://proceedings.mlr.press/v37/sohl-dickstein15.html)
- Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in Neural Information Processing Systems, 2020, 33: 6840-6851.[https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)
- Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.[https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
- Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.[https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html)
<a name="LJfyE"></a>
#### Diffusion Application:

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
<a name="A6T1n"></a>
#### Diffusion Compression:

- Meng C, Gao R, Kingma D P, et al. On distillation of guided diffusion models[J]. arXiv preprint arXiv:2210.03142, 2022.[https://arxiv.org/abs/2210.03142](https://arxiv.org/abs/2210.03142)
- Yang X, Zhou D, Feng J, et al. Diffusion Probabilistic Model Made Slim[J]. arXiv preprint arXiv:2211.17106, 2022.[https://arxiv.org/abs/2211.17106](https://arxiv.org/abs/2211.17106)
- Theis L, Salimans T, Hoffman M D, et al. Lossy compression with gaussian diffusion[J]. arXiv preprint arXiv:2206.08889, 2022.[https://arxiv.org/abs/2206.08889](https://arxiv.org/abs/2206.08889)
- Luhman E, Luhman T. Knowledge distillation in iterative generative models for improved sampling speed[J]. arXiv preprint arXiv:2101.02388, 2021.[https://arxiv.org/abs/2101.02388](https://arxiv.org/abs/2101.02388)
- Salimans T, Ho J. Progressive distillation for fast sampling of diffusion models[J]. arXiv preprint arXiv:2202.00512, 2022.[https://arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512)
<a name="iZP3s"></a>
#### Diffusion Accelerate:

- Deng Y, Kojima N, Rush A M. Markup-to-Image Diffusion Models with Scheduled Sampling[J]. arXiv preprint arXiv:2210.05147, 2022.[https://arxiv.org/abs/2210.05147](https://arxiv.org/abs/2210.05147)
- Luzi L, Siahkoohi A, Mayer P M, et al. Boomerang: Local sampling on image manifolds using diffusion models[J]. arXiv preprint arXiv:2210.12100, 2022.[https://arxiv.org/abs/2210.12100](https://arxiv.org/abs/2210.12100)
- Wizadwongsa S, Suwajanakorn S. Accelerating Guided Diffusion Sampling with Splitting Numerical Methods[J]. arXiv preprint arXiv:2301.11558, 2023.[https://arxiv.org/abs/2301.11558](https://arxiv.org/abs/2301.11558)
- Zhang Q, Chen Y. Fast sampling of diffusion models with exponential integrator[J]. arXiv preprint arXiv:2204.13902, 2022.[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)
- Tachibana H, Go M, Inahara M, et al. Quasi-Taylor Samplers for Diffusion Generative Models based on Ideal Derivatives[J].[https://openreview.net/forum?id=7ks5PS09q1](https://openreview.net/forum?id=7ks5PS09q1)
- Kim B, Ye J C. Denoising MCMC for Accelerating Diffusion-Based Generative Models[J]. arXiv preprint arXiv:2209.14593, 2022.[https://arxiv.org/abs/2209.14593](https://arxiv.org/abs/2209.14593)
- Zheng H, He P, Chen W, et al. Truncated diffusion probabilistic models and diffusion-based adversarial auto-encoders[J]. arXiv preprint arXiv:2202.09671, 2022.[https://arxiv.org/abs/2202.09671](https://arxiv.org/abs/2202.09671)
- Lin X, Jwalapuram P, Joty S. Dynamic Scheduled Sampling with Imitation Loss for Neural Text Generation[J]. arXiv preprint arXiv:2301.13753, 2023.[https://arxiv.org/abs/2301.13753](https://arxiv.org/abs/2301.13753)
- Lu C, Zhou Y, Bao F, et al. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models[J]. arXiv preprint arXiv:2211.01095, 2022.[https://arxiv.org/abs/2211.01095](https://arxiv.org/abs/2211.01095)
- Lu C, Zhou Y, Bao F, et al. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps[J]. arXiv preprint arXiv:2206.00927, 2022.[https://arxiv.org/abs/2206.00927](https://arxiv.org/abs/2206.00927)
<a name="VX71f"></a>

## demo
In the demo folder, a series of sample demos exists related to dpm solver/dpm solver++. Detail can be found in [readme.md](./demo/README.md).

## model
UNet copy from [ADM](https://github.com/openai/guided-diffusion).

## schedule
Scheduler copy from [DPM Solver/DPM Solver++](https://github.com/LuChengTHU/dpm-solver)