## Diffusion Model Papers

- [Diffusion Model Survey](#diffusion-model-survey)
- [Diffusion Model Theory](#diffusion-model-theory)
    - [Crucial](#crucial)
    - [SDE/ODE/Flow](#sdeodeflow)
    - [Architecture](#architecture)
- [Diffusion Model Application](#diffusion-model-application)
    - [Crucial](#crucial-1)
    - [Long-tail Distribution](#long-tail-distribution)
    - [2D/3D Vision-Language](#2d3d-vision-language)
    - [Self-supervised Learning](#self-supervised-learning)
    - [Data Augmentation/Expansion](#data-augmentationexpansion)
    - [Object Detection](#object-detection)
    - [Visual Tracking](#visual-tracking)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Instance Segmentation](#instance-segmentation)
    - [Panoptic Segmentation](#panoptic-segmentation)
    - [Medical](#medical)
    - [Image Editing](#image-editing)
    - [Low-level Vision](#low-level-vision)
    - [Image Matting](#image-matting)
    - [Deblur](#deblur)
    - [3D-point Cloud](#3d-point-cloud)
    - [Video Generation/Understanding/Prediction](#video-generationunderstandingprediction)
    - [Action Detection](#action-detection)
    - [Dataset Compression](#dataset-compression)
    - [Knowledge Distillation](#knowledge-distillation)
    - [Parameter Pruning](#parameter-pruning)
    - [Model Quantization](#model-quantization)
    - [Depth Estimation](#depth-estimation)
    - [Text Detection](#text-detection)
    - [Anomaly Detection](#anomaly-detection)
    - [3D Reconstruction](#3d-reconstruction)
    - [Trajectory Prediction](#trajectory-prediction)
    - [Lane Detection](#lane-detection)
    - [Image Captioning](#image-captioning)
    - [Visual Question Answering](#visual-question-answering)
    - [Sign Language Recognition](#sign-language-recognition)
    - [Novel View Synthesis](#novel-view-synthesis)
    - [Zero/Few-Shot Learning](#zerofew-shot-learning)
    - [Transfer Learning](#transfer-learning)
    - [Stereo Matching](#stereo-matching)
    - [Scene Graph Generation](#scene-graph-generation)
    - [Image Quality Assessment](#image-quality-assessment)
    - [Music](#music)
    - [Speech](#speech)
    - [GflowNet](#gflownet)
- [Diffusion Model Compression/Accelerated Sampling](#diffusion-model-compressionaccelerated-sampling)
    - [Accelerated Sampling](#accelerated-sampling)
    - [Model Compression](#model-compression)

### Diffusion Model Survey:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
|  A survey on generative diffusion model  | Hanqun Cao  | Arxiv | [paper](https://arxiv.org/abs/2209.02646) | [code](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model) |
| Diffusion Models for Time Series Applications: A Survey | Lequan Lin | Arxiv | [paper](https://arxiv.org/abs/2305.00624) | - |
| Diffusion models for medical image analysis: A comprehensive survey | Amirhossein Kazerouni | Arxiv | [paper](https://arxiv.org/abs/2211.07804) | [code](https://github.com/amirhossein-kz/awesome-diffusion-models-in-medical-imaging) |
| A Comprehensive Survey on Knowledge Distillation of Diffusion Models | Weijian Luo | Arxiv | [paper](https://arxiv.org/abs/2304.04262) | - |
| A Survey on Graph Diffusion Models: Generative AI in Science for Molecule, Protein and Material | Mengchun Zhang | Arxiv | [paper](https://arxiv.org/abs/2304.01565) | - |
| Audio Diffusion Model for Speech Synthesis: A Survey on Text To Speech and Speech Enhancement in Generative AI | Chenshuang Zhang | Arxiv | [paper](https://arxiv.org/abs/2303.13336) | - |
| Diffusion Models in NLP: A Survey | Hao Zou | Arxiv | [paper](https://arxiv.org/abs/2305.14671) | - |
| Text-to-image Diffusion Models in Generative AI: A Survey | Chenshuang Zhang | Arxiv | [paper](https://arxiv.org/abs/2303.07909) | - |
| Diffusion Models for Non-autoregressive Text Generation: A Survey | Yifan Li | IJCAI survey track | [paper](https://arxiv.org/abs/2303.06574) | [code](https://github.com/rucaibox/awesome-text-diffusion-models) |
| Diffusion Models in Bioinformatics: A New Wave of Deep Learning Revolution in Action | Zhiye Guo | Arxiv | [paper](https://arxiv.org/abs/2302.10907) | - |
| Generative Diffusion Models on Graphs: Methods and Applications | Chengyi Liu | Arxiv | [paper](https://arxiv.org/abs/2302.02591) | - |
| Efficient Diffusion Models for Vision: A Survey | Anwaar Ulhaq | Arxiv | [paper](https://arxiv.org/abs/2210.09292) | - |
| Diffusion Models in Vision: A Survey | Florinel-Alin Croitoru | TPAMI | [paper](https://ieeexplore.ieee.org/abstract/document/10081412/) | - |
| A Survey on Generative Diffusion Model | Hanqun Cao | Arxiv | [paper](https://arxiv.org/abs/2209.02646) | - |
| Diffusion Models: A Comprehensive Survey of Methods and Applications | Ling Yang | Arxiv | [paper](https://arxiv.org/abs/2209.00796) | [code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy) |
| Ai-generated content (aigc): A survey | Jiayang Wu | Arxiv | [paper](https://arxiv.org/abs/2304.06632) | - |
| A Complete Survey on Generative AI (AIGC): Is ChatGPT from GPT-4 to GPT-5 All You Need? | Chaoning Zhang | Arxiv | [paper](https://arxiv.org/abs/2303.11717) | - |

### Diffusion Model Theory:

#### Crucial:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Denoising Diffusion Probabilistic Models | Jonathan Ho | NIPS | [paper](https://arxiv.org/abs/2006.11239) | [code](https://github.com/hojonathanho/diffusion) | 
| Score-Based Generative Modeling through Stochastic Differential Equations | Yang Song | ICLR best paper | [paper](https://arxiv.org/abs/2011.13456) | [code](https://github.com/yang-song/score_sde) |
| Denoising Diffusion Implicit Models | Jiaming Song | ICLR | [paper](https://arxiv.org/abs/2010.02502) | [code](https://github.com/ermongroup/ddim) |
|  Consistency Models  | Yang Song  | ICML | [paper](https://arxiv.org/abs/2303.01469) | [code](https://github.com/openai/consistency_models) |
| Variational diffusion models | Diederik Kingma  | NIPS | [paper](https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html) | [code](https://github.com/google-research/vdm) |
| Deep Unsupervised Learning using Nonequilibrium Thermodynamics | Jascha Sohl-Dickstein | PMLR | [paper](http://proceedings.mlr.press/v37/sohl-dickstein15.html) | [code](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models) |
| Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models | Fan Bao | ICLR | [paper](https://openreview.net/forum?id=0xiJLKH-ufZ) | [code](https://github.com/baofff/Analytic-DPM) |
| Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow | Xingchao Liu | ICLR | [paper](https://arxiv.org/abs/2209.03003) | [code](https://github.com/gnobitab/RectifiedFlow) |
| Elucidating the Design Space of Diffusion-Based Generative Models | Tero Karras | NIPS | [paper](https://arxiv.org/abs/2206.00364) | [code](https://github.com/nvlabs/edm) |  
| Improved Denoising Diffusion Probabilistic Models | Alex Nichol | ICML | [paper](https://proceedings.mlr.press/v139/nichol21a.html) | [code](https://github.com/openai/improved-diffusion) |
| Maximum likelihood training of score-based diffusion models | Yang Song | NIPS | [paper](https://arxiv.org/abs/2101.09258) | [code](https://github.com/yang-song/score_flow) |


#### SDE/ODE/Flow

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| On enhancing the robustness of Vision Transformers: Defensive Diffusion | Raza Imam | Arxiv | [paper](https://arxiv.org/abs/2305.08031) | [code](https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion) |
| Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model | Xiulong Yang | Arxiv | [paper](https://arxiv.org/abs/2208.07791) | [code](https://github.com/sndnyang/Diffusion_ViT) |
| Your Diffusion Model is Secretly a Zero-Shot Classifier | Alexander C. Li | Arxiv | [paper](https://arxiv.org/abs/2303.16203) | [code](https://github.com/diffusion-classifier/diffusion-classifier) |
| SVDiff: Compact Parameter Space for Diffusion Fine-Tuning | Ligong Han | Arxiv | [paper](https://arxiv.org/abs/2303.11305) | [code](https://github.com/phymhan/SVDiff) |
| Stochastic Interpolants: A Unifying Framework for Flows and Diffusions | Michael S. Albergo | Arxiv | [paper](https://arxiv.org/abs/2303.08797) | - |
| Rectified Flow: A Marginal Preserving Approach to Optimal Transport | Qiang Liu | Arxiv | [paper](https://arxiv.org/abs/2209.14577) | - |
| Minimizing Trajectory Curvature of ODE-based Generative Models | Sangyun Lee | ICML | [paper](https://arxiv.org/abs/2301.12003) | [code](https://github.com/sangyun884/fast-ode) |
| Convergence of score-based generative modeling for general data distributions | Holden Lee | NIPS workshop | [paper](https://openreview.net/forum?id=Sg19A8mu8sv) | - |
| Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions | Sitian Chen | ICLR | [paper](https://arxiv.org/abs/2209.11215) | - |
| Flow Matching for Generative Modeling | Yaron Lipman | Arxiv | [paper](https://arxiv.org/abs/2210.02747) | - |
| Estimating the Optimal Covariance with Imperfect Mean in Diffusion Probabilistic Models | Fan Bao | ICML | [paper](https://arxiv.org/abs/2206.07309) | [code](https://github.com/baofff/Extended-Analytic-DPM) |
| Expressiveness Remarks for Denoising Diffusion Models and Samplers | Francisco Vargas | Arxiv | [paper](https://arxiv.org/abs/2305.09605) | - |
| The Score-Difference Flow for Implicit Generative Modeling | Romann M. Weber | Arxiv | [paper](https://arxiv.org/abs/2304.12906) | - |
| Diffusion Models for Constrained Domains | Nic Fishman | Arxiv | [paper](https://arxiv.org/abs/2304.05364) | - |
| Diffusion Bridge Mixture Transports, Schrödinger Bridge Problems and Generative Modeling | Stefano Peluchetti | Arxiv | [paper](https://arxiv.org/abs/2304.00917) | - |
| Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models | Anant Raj | Arxiv | [paper](https://arxiv.org/abs/2303.17109) | - |
| Diffusion Schrödinger Bridge Matching | Yuyang Shi | Arxiv | [paper](https://arxiv.org/abs/2303.16852) | - |
| Restoration-Degradation Beyond Linear Diffusions: A Non-Asymptotic Analysis For DDIM-Type Samplers | Sitan Chen | Arxiv | [paper](https://arxiv.org/abs/2303.03384) | - |
| Diffusion Models are Minimax Optimal Distribution Estimators | Kazusato Oko | ICLR workshop | [paper](https://openreview.net/forum?id=6961CeTSFA) | - |
| Understanding the Diffusion Objective as a Weighted Integral of ELBOs | Diederik P. Kingma | Arxiv | [paper](https://arxiv.org/abs/2303.00848) | - |
| Continuous-Time Functional Diffusion Processes | Giulio Franzese | Arxiv | [paper](https://arxiv.org/abs/2303.00800) | - |
| Denoising Diffusion Samplers | Francisco Vargas | Arxiv | [paper](https://openreview.net/forum?id=8pvnfTAbu1f) | - |
| Infinite-Dimensional Diffusion Models for Function Spaces | Jakiw Pidstrigach | Arxiv | [paper](https://arxiv.org/abs/2302.10130) | - |
| Score-based Diffusion Models in Function Space | Jae Hyun Lim | Arxiv | [paper](https://arxiv.org/abs/2302.07400) | - |
| Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data | Minshuo Chen | Arxiv | [paper](https://arxiv.org/abs/2302.07194) | - |
| Stochastic Modified Flows, Mean-Field Limits and Dynamics of Stochastic Gradient Descent | Benjamin Gess | Arxiv | [paper](https://arxiv.org/abs/2302.07125) | - |
| Monte Carlo Neural Operator for Learning PDEs via Probabilistic Representation | Rui Zhang | Arxiv | [paper](https://arxiv.org/abs/2302.05104) | - |
| Example-Based Sampling with Diffusion Models | Bastien Doignies | Arxiv | [paper](https://arxiv.org/abs/2302.05116) | - |
| Information-Theoretic Diffusion | Xianghao Kong | ICLR | [paper](https://openreview.net/forum?id=UvmDCdSPDOW) | [code](https://github.com/kxh001/ITdiffusion) |
| Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport | Alexander Tong | Arxiv | [paper](https://arxiv.org/abs/2302.00482) | [code](https://github.com/atong01/conditional-flow-matching) |
| Transport with Support: Data-Conditional Diffusion Bridges | Ella Tamir | Arxiv | [paper](https://arxiv.org/abs/2301.13636) | - |
| Understanding and contextualising diffusion models | Stefano Scotta | Arxiv | [paper](https://arxiv.org/abs/2302.01394) | [code](https://github.com/stefanoscotta/1-d-generative-diffusion-model) |
| On the Mathematics of Diffusion Models | David McAllester | Arxiv | [paper](https://arxiv.org/abs/2301.11108) | - |
| Mathematical analysis of singularities in the diffusion model under the submanifold assumption | Yubin Lu | Arxiv | [paper](https://arxiv.org/abs/2301.07882) | - |
| Thompson Sampling with Diffusion Generative Prior | Yu-Guan Hsieh | Arxiv | [paper](https://arxiv.org/abs/2301.05182) | - |
| Your diffusion model secretly knows the dimension of the data manifold | Jan Stanczuk | Arxiv | [paper](https://arxiv.org/abs/2212.12611) | - |
| Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance | Dohyun Kwon | NIPS | [paper](https://arxiv.org/abs/2212.06359) | [code](https://github.com/uw-madison-lee-lab/score-wasserstein) |
| Nonlinear controllability and function representation by neural stochastic differential equations | Tanya Veeravalli | Arxiv | [paper](https://arxiv.org/abs/2212.00896) | - |
| Diffusion Generative Models in Infinite Dimensions | Gavin Kerrigan | AISTATS | [paper](https://arxiv.org/abs/2212.00886) | [code](https://github.com/gavinkerrigan/functional_diffusion) |
| Neural Langevin Dynamics: towards interpretable Neural Stochastic Differential Equations | Simon M. Koop | Arxiv | [paper](https://arxiv.org/abs/2211.09537) | - |
| Improved Analysis of Score-based Generative Modeling: User-Friendly Bounds under Minimal Smoothness Assumptions | Hongrui Chen | Arxiv | [paper](https://arxiv.org/abs/2211.01916) | - |
| Categorical SDEs with Simplex Diffusion | Pierre H. Richemond | Arxiv | [paper](https://arxiv.org/abs/2210.14784) | - |
| Diffusion Models for Causal Discovery via Topological Ordering | Pedro Sanchez | Arxiv | [paper](https://arxiv.org/abs/2210.06201) | [code](https://github.com/vios-s/diffan) |
| Regularizing Score-based Models with Score Fokker-Planck Equations | Chieh-Hsin Lai | NIPS workshop | [paper](https://openreview.net/forum?id=WqW7tC32v8N) | - |
| Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models | Louis Sharrock | Arxiv | [paper](https://arxiv.org/abs/2210.04872) | - |
| Analyzing diffusion as serial reproduction  | Raja Marjieh | Arxiv | [paper](https://arxiv.org/abs/2209.14821) | - |
| Convergence of score-based generative modeling for general data distributions | Holden Lee | NIPS workshop | [paper](https://arxiv.org/abs/2209.12381) | - |
| Riemannian Diffusion Models | Chin-Wei Huang | NIPS | [paper](https://arxiv.org/abs/2208.07949) | - |
| Convergence of denoising diffusion models under the manifold hypothesis | Valentin De Bortoli | TMLR | [paper](https://openreview.net/forum?id=MhK5aXo3gB) | - |
| Neural Diffusion Processes | Vincent Dutordoir | Arxiv | [paper](https://arxiv.org/abs/2206.03992) | [code](https://github.com/vdutor/neural-diffusion-processes) |
| Theory and Algorithms for Diffusion Processes on Riemannian Manifolds | Bowen Jing | NIPS | [paper](https://arxiv.org/abs/2206.01729) | [code](https://github.com/gcorso/torsional-diffusion) |
| Riemannian Score-Based Generative Modelling | Valentin De Bortoli | NIPS | [paper](https://arxiv.org/abs/2202.02763) | [code](https://github.com/oxcsml/riemannian-score-sde) |
| Interpreting diffusion score matching using normalizing flow | Wenbo Gong | ICLR workshop | [paper](https://arxiv.org/abs/2107.10072) | - |
| Diffusion normalizing flow | Zhang, Qinsheng | NIPS | [paper](https://arxiv.org/abs/2110.07579) | [code](https://github.com/qsh-zh/DiffFlow) |
| Maximum Likelihood Training of Implicit Nonlinear Diffusion Models | Dongjun Kim | NIPS | [paper](https://openreview.net/forum?id=TQn44YPuOR2) | [code](https://github.com/byeonghu-na/INDM) |
| Score-based generative modeling in latent space | Vahdat, Arash | NIPS | [paper](https://arxiv.org/abs/2106.05931) | [code](https://github.com/NVlabs/LSGM) |
| Maximum Likelihood Training of Parametrized Diffusion Model | Dongjun Kim | ICLR reject | [paper](https://openreview.net/forum?id=1v1N7Zhmgcx) | - |
| Soft truncation: A universal training technique of score-based diffusion model for high precision score estimation | Dongjun Kim | ICML | [paper](https://proceedings.mlr.press/v162/kim22i.html) | [code](https://github.com/Kim-Dongjun/Soft-Truncation) |
| Robust Classification via a Single Diffusion Model | Huanran Chen | Arxiv | [paper](https://arxiv.org/abs/2305.15241) | - |
| Maximum Likelihood Training for Score-Based Diffusion ODEs by High-Order Denoising Score Matching | Cheng Lu | ICML | [paper](https://arxiv.org/abs/2206.08265) | [code](https://github.com/luchengthu/mle_score_ode) |
| Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs | Kaiwen Zheng | ICML | [paper](https://arxiv.org/abs/2305.03935) | - |

#### Architecture

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| All are Worth Words: A ViT Backbone for Diffusion Models | Fan Bao | CVPR | [paper](https://arxiv.org/abs/2209.12152) | [code](https://github.com/baofff/U-ViT) |

### Diffusion Model Application:

** Note: In this title, we just summarize works that use diffusion models for their specific domains, rather than using their specific domains for diffusion models. Thus, no works can be listed in parameter pruning, model quantization.**

** Note: For a field, if there is no relevant work listed in it, most likely because I haven't put it up yet.**

#### Crucial:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Diffusion Models Beat GANs on Image Synthesis | Prafulla Dhariwal | NIPS | [paper](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html) | [code](https://github.com/openai/guided-diffusion)|
| Classifier-Free Diffusion Guidance | Jonathan Ho | NIPS workshop | [paper](https://arxiv.org/abs/2207.12598) | [code](https://github.com/Michedev/DDPMs-Pytorch) |
| DreamFusion: Text-to-3D using 2D Diffusion | Ben Poole | ICLR | [paper](https://arxiv.org/abs/2209.14988) | [code](https://github.com/ashawkey/stable-dreamfusion) |

#### NeRF:C Bodnar 

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| DreamFusion: Text-to-3D using 2D Diffusion | Ben Poole | ICLR | [paper](https://arxiv.org/abs/2209.14988) | [code](https://github.com/ashawkey/stable-dreamfusion) |
| Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation | Haochen Wang | CVPR | [paper](https://arxiv.org/abs/2212.00774) | [code](https://github.com/pals-ttic/sjc/) |
| Magic3D: High-Resolution Text-to-3D Content Creation | Chen-Hsuan Lin | CVPR | [paper](https://arxiv.org/abs/2211.10440) | [code](https://github.com/chinhsuanwu/dreamfusionacc) |
| Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation | Rui Chen | Arxiv | [paper](https://arxiv.org/abs/2303.13873) | [code](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) |
| Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures | Gal Metzer | CVPR | [paper](https://arxiv.org/abs/2211.07600) | [code](https://github.com/eladrich/latent-nerf) | 
| ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation | Zhengyi Wang | Arxiv | [paper](https://arxiv.org/abs/2305.16213) | [code](https://github.com/thu-ml/prolificdreamer) |

#### GNN:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Grand: Graph neural diffusion | B Chamberlain | NIPS workshop/ICML | [paper](http://proceedings.mlr.press/v139/chamberlain21a.html) | [code](https://github.com/twitter-research/graph-neural-pde) |
| Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in gnns | C Bodnar | NIPS | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html) | - |
| Graph Neural Networks as Gradient Flows: understanding graph convolutions via energy |Francesco Di Giovanni | Arxiv | [paper](https://arxiv.org/abs/2206.10991) | - |

#### Reid:
 
|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Person image synthesis via denoising diffusion model | AK Bhunia | CVPR | [paper](Bhunia_Person_Image_Synthesis_via_Denoising_Diffusion_Model_CVPR_2023_paper) | [code](https://github.com/ankanbhunia/PIDM) |


#### Long-tail Distribution:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| DiffBFR: Bootstrapping Diffusion Model Towards Blind Face Restoration | Xinmin Qiu | Arxiv | [paper](https://arxiv.org/abs/2305.04517) | - |
| Class-Balancing Diffusion Models | Yiming Qin | CVPR | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Qin_Class-Balancing_Diffusion_Models_CVPR_2023_paper.html) | - |

#### 2D/3D Vision-Language:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| DreamFusion: Text-to-3D using 2D Diffusion | Ben Poole | ICLR | [paper](https://arxiv.org/abs/2209.14988) | [code](https://github.com/ashawkey/stable-dreamfusion) |
| Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation | Haochen Wang | CVPR | [paper](https://arxiv.org/abs/2212.00774) | [code](https://github.com/pals-ttic/sjc/) |
| Magic3D: High-Resolution Text-to-3D Content Creation | Chen-Hsuan Lin | CVPR | [paper](https://arxiv.org/abs/2211.10440) | [code](https://github.com/chinhsuanwu/dreamfusionacc) |
| Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation | Rui Chen | Arxiv | [paper](https://arxiv.org/abs/2303.13873) | [code](https://github.com/Gorilla-Lab-SCUT/Fantasia3D) |
| Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures | Gal Metzer | CVPR | [paper](https://arxiv.org/abs/2211.07600) | [code](https://github.com/eladrich/latent-nerf) | 
| ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation | Zhengyi Wang | Arxiv | [paper](https://arxiv.org/abs/2305.16213) | [code](https://github.com/thu-ml/prolificdreamer) |

#### Self-supervised Learning:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Self-Score: Self-Supervised Learning on Score-Based Models for MRI Reconstruction | Zhuo-Xu Cui | Arxiv | [paper](https://arxiv.org/abs/2209.00835) | - |
| Diffusion adversarial representation learning for self-supervised vessel segmentation | Boah Kim | ICLR | [paper](https://arxiv.org/abs/2209.14566) | - |
| DDM: Self-Supervised Diffusion MRI Denoising with Generative Diffusion Models | Tiange Xiang | ICLR | [paper](https://arxiv.org/abs/2302.03018) | [code](https://github.com/stanfordmimi/ddm2) |
| Learning 3D Photography Videos via Self-supervised Diffusion on Single Images | Xiaodong Wang | Arxiv | [paper](https://arxiv.org/abs/2302.10781) | - |
| DDS2M: Self-Supervised Denoising Diffusion Spatio-Spectral Model for Hyperspectral Image Restoration | Yuchun Miao | Arxiv | [paper](https://arxiv.org/abs/2303.06682) | - |
| Data-Centric Learning from Unlabeled Graphs with Diffusion Model | Gang Liu | Arxiv | [paper](https://arxiv.org/abs/2303.10108) | [code](https://github.com/liugangcode/data_centric_transfer) |
| Denoising Diffusion Autoencoders are Unified Self-supervised Learners | Weilai Xiang | Arxiv | [paper](https://arxiv.org/abs/2303.09769) | - |

#### Data Augmentation/Expansion:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Effective Data Augmentation With Diffusion Models | Brandon Trabucco | Arxiv | [paper](https://arxiv.org/abs/2302.07944) | [code](https://github.com/brandontrabucco/da-fusion) |
| Deep Data Augmentation for Weed Recognition Enhancement: A Diffusion Probabilistic Model and Transfer Learning Based Approach | Dong Chen | Arxiv | [paper](https://arxiv.org/abs/2210.09509) | [code](https://github.com/dongchen06/dmweeds) |
| Diffusion-based Data Augmentation for Skin Disease Classification: Impact Across Original Medical Datasets to Fully Synthetic Images | Mohamed Akrout | Arxiv | [paper](https://arxiv.org/abs/2301.04802) | - |
| A data augmentation perspective on diffusion models and retrieval | Max F. Burg | Arxiv | [paper](https://arxiv.org/abs/2304.10253) | - |
| Multimodal Data Augmentation for Image Captioning using Diffusion Models | Changrong Xiao | Arxiv | [paper](https://arxiv.org/abs/2305.01855) | [code](https://github.com/xiaochr/multimodal-augmentation-image-captioning) |
| DiffuseExpand: Expanding dataset for 2D medical image segmentation using diffusion models | Shitong Shao | Arxiv | [paper](https://arxiv.org/abs/2304.13416) | [code](https://github.com/shaoshitong/DiffuseExpand) |
| AugDiff: Diffusion based Feature Augmentation for Multiple Instance Learning in Whole Slide Image | Zhuchen Shao | Arxiv | [paper](https://arxiv.org/abs/2303.06371) | - | 

#### Object Detection:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Diffusiondet: Diffusion model for object detection | Shoufa Chen | Arxiv | [paper](https://arxiv.org/abs/2211.09788) | [code](https://github.com/ShoufaChen/DiffusionDet) |

#### Visual Tracking:

#### Semantic Segmentation:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Segdiff: Image segmentation with diffusion probabilistic models | Tomer Amit | Arxiv | [paper](https://arxiv.org/abs/2112.00390) | [code](https://github.com/tomeramit/SegDiff) |
| Ambiguous Medical Image Segmentation Using Diffusion Models | Aimon Rahman | CVPR | [paper](https://arxiv.org/abs/2304.04745) | [code](https://github.com/aimansnigdha/ambiguous-medical-image-segmentation-using-diffusion-models) |

#### Instance Segmentation:

#### Panoptic Segmentation:

#### Medical:

#### Image Editing:

#### Low-level Vision:

#### Image Matting:

#### Deblur:

#### 3D-point Cloud:

#### Video Generation/Understanding/Prediction:

#### Action Detection:

#### Dataset Compression:

#### Knowledge Distillation:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Knowledge Diffusion for Distillation | Tao Huang | Arxiv | [paper](https://arxiv.org/abs/2305.15712) | [code](https://github.com/hunto/diffkd) |

#### Parameter Pruning:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |

#### Model Quantization:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |

#### Depth Estimation:

#### Text Detection:

#### Anomaly Detection:

#### 3D Reconstruction:

#### Trajectory Prediction:

#### Lane Detection:

#### Image Captioning:

#### Visual Question Answering:

#### Sign Language Recognition:

#### Novel View Synthesis:

#### Zero/Few-Shot Learning:

#### Transfer Learning:

#### Stereo Matching:

#### Scene Graph Generation:

#### Image Quality Assessment:

#### Music:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Symbolic Music Generation with Diffusion Models | Gautam Mittal | ISMIR | [paper](https://arxiv.org/abs/2103.16091) | [code](https://github.com/magenta/symbolic-music-diffusion) |
| Generating symbolic music using diffusion models | Lilac Atassi | Arxiv | [paper](https://arxiv.org/abs/2303.08385) | [code](https://github.com/lilac-code/music-diffusion) |
| Solving Audio Inverse Problems with a Diffusion Model | Eloi Moliner | ICASSP | [paper](https://ieeexplore.ieee.org/abstract/document/10095637) | - |
| DiffuseRoll: Multi-track multi-category music generation based on diffusion model | Hongfei Wang | Arxiv | [paper](https://arxiv.org/abs/2303.07794) | - |
| MAID: A Conditional Diffusion Model for Long Music Audio Inpainting | Kaiyang Liu | ICASSP | [paper](https://ieeexplore.ieee.org/abstract/document/10095769) | - |
| EDGE: Editable Dance Generation From Music | Jonathan Tseng | CVPR | [paper](https://arxiv.org/abs/2211.10658) | [code](https://github.com/Stanford-TML/EDGE) |

#### Speech:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Prodiff: Progressive fast diffusion model for high-quality text-to-speech | Rongjie Huang | ACMMM | [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547855) | [code](https://github.com/Rongjiehuang/ProDiff) |
| Diff-TTS: A Denoising Diffusion Model for Text-to-Speech | Myeonghun Jeong | Arxiv | [paper](https://arxiv.org/abs/2104.01409) | [code](https://github.com/keonlee9420/DiffSinger) | 
| Fastdiff: A fast conditional diffusion model for high-quality speech synthesis | Rongjie Huang | IJCAI | [paper](https://arxiv.org/abs/2204.09934) | [code](https://github.com/Rongjiehuang/FastDiff) |
| Guided-tts: A diffusion model for text-to-speech via classifier guidance | Heeseung Kim | ICML | [paper](https://arxiv.org/abs/2111.11755) | - |
| Restoring degraded speech via a modified diffusion model | Jianwei Zhang | Arxiv | [paper](https://arxiv.org/abs/2104.11347) | - |
| DiffMotion: Speech-Driven Gesture Synthesis Using Denoising Diffusion Model | Fan Zhang | MMM | [paper](https://link.springer.com/chapter/10.1007/978-3-031-27077-2_18) | [code](https://github.com/zf-CUZ/DiffMotion) |

#### GflowNet:

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Generative Flow Networks for Discrete Probabilistic Modeling | Dinghuai Zhang | NIPS | [paper](https://arxiv.org/abs/2202.01361) | [code](https://github.com/GFNOrg/EB_GFN) |
| ROBUST SCHEDULING WITH GFLOWNETS | David W. Zhang | ICLR | [paper](https://arxiv.org/abs/2302.05446) | [code](https://github.com/saleml/torchgfn) |
| Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation | Emmanuel Bengio | NIPS | [paper](https://arxiv.org/abs/2106.04399) | [code](https://github.com/GFNOrg/gflownet) | 
| Biological Sequence Design with GFlowNets | Moksh Jain | ICML | [paper](https://proceedings.mlr.press/v162/jain22a.html) | [code](https://github.com/MJ10/BioSeq-GFN-AL) |
| Bayesian Structure Learning with Generative Flow Networks | Tristan Deleu | NIPS | [paper](https://arxiv.org/abs/2202.13903) | [code](https://github.com/tristandeleu/jax-dag-gflownet) | 
| Trajectory balance: Improved credit assignment in GFlowNets | Nikolay Malkin | NIPS | [paper](https://arxiv.org/abs/2201.13259) | [code](https://gist.github.com/malkin1729/9a87ce4f19acdc2c24225782a8b81c15) |
| A theory of continuous generative flow networks | Salem Lahlou | ICML | [paper](https://arxiv.org/abs/2301.12594) | [code](https://github.com/saleml/continuous-gfn) |
| Better Training of GFlowNets with Local Credit and Incomplete Trajectories | Ling Pan | Arxiv | [paper](https://arxiv.org/abs/2302.01687) | - |
| Unifying Generative Models with GFlowNets and Beyond | Dinghuai Zhang | ICML workshop | [paper](https://arxiv.org/abs/2209.02606) | - |
| A Max-Flow Based Approach for Neural Architecture Search | Chao Xue | ECCV | [paper](https://link.springer.com/chapter/10.1007/978-3-031-20044-1_39) | - |
| Multi-Objective GFlowNets | Moksh Jain | Arxiv | [paper](https://arxiv.org/abs/2210.12765) | - |
| Towards Understanding and Improving GFlowNet Training | Max W. Shen | ICML | [paper](https://arxiv.org/abs/2305.07170) | [code](https://github.com/maxwshen/gflownet) |
| torchgfn: A PyTorch GFlowNet library | Salem Lahlou | Arxiv | [paper](https://arxiv.org/abs/2305.14594) | [code](https://github.com/saleml/torchgfn) |

### Diffusion Model Compression/Accelerated Sampling:

#### Accelerated Sampling

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Fast Sampling of Diffusion Models with Exponential Integrator | Qinsheng Zhang | ICLR | [paper](https://arxiv.org/abs/2204.13902) | [code](https://github.com/qsh-zh/deis) |
| Fast Sampling of Diffusion Models via Operator Learning | Hongkai Zheng | NIPS workshop | [paper](https://openreview.net/forum?id=XrhofG6qg7Y) | - |
| Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models | Muyang Li | NIPS | [paper](https://arxiv.org/abs/2211.02048) | [code](https://github.com/lmxyy/sige) |
| DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps | Cheng Lu | NIPS | [paper](https://arxiv.org/abs/2206.00927) | [code](https://github.com/luchengthu/dpm-solver) |
| BOOMERANG: LOCAL SAMPLING ON IMAGE MANIFOLDS USING DIFFUSION MODELS | Lorenzo Luzi | Arxiv | [paper](https://arxiv.org/abs/2210.12100) | - |
| Accelerating Guided Diffusion Sampling with Splitting Numerical Methods | Suttisak Wizadwongsa | ICLR | [paper](https://openreview.net/forum?id=F0KTk2plQzO) | [code](https://github.com/sWizad/split-diffusion) |
| Quasi-Taylor Samplers for Diffusion Generative Models based on Ideal Derivatives | Hideyuki Tachibana | Arxiv | [paper](https://arxiv.org/abs/2112.13339) | - |
| Denoising MCMC for Accelerating Diffusion-Based Generative Models | Beomsu Kim | Arxiv | [paper](https://arxiv.org/abs/2209.14593) | [code](https://github.com/1202kbs/DMCMC) |
| DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models | Cheng Lu | Arxiv | [paper](https://arxiv.org/abs/2211.01095) | [code](https://github.com/luchengthu/dpm-solver) |
|  Consistency Models  | Yang Song  | ICML | [paper](https://arxiv.org/abs/2303.01469) | [code](https://github.com/openai/consistency_models) |
| Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling | Shitong Shao | Arxiv | [paper](https://arxiv.org/abs/2305.10769) | [code](https://github.com/shaoshitong/Catch-Up-Distillation) |
| Progressive Distillation for Fast Sampling of Diffusion Models | Tim Salimans | ICLR | [paper](https://arxiv.org/abs/2202.00512) | [code](https://github.com/google-research/google-research/tree/master/diffusion_distillation) |
| ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech | Rongjie Huang | ACMMM | [paper](https://arxiv.org/abs/2207.06389) | [code](https://github.com/Rongjiehuang/ProDiff) |
| Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed | Eric Luhman | Arxiv | [paper](https://arxiv.org/abs/2101.02388) | [code](https://github.com/tcl9876/Denoising_Student) |
| Accelerating Diffusion Models via Early Stop of the Diffusion Process | Zhaoyang Lyu | Arxiv | [paper](https://arxiv.org/abs/2205.12524) | [code](https://github.com/ZhaoyangLyu/Early_Stopped_DDPM) |
| Truncated diffusion probabilistic models | Huangjie Zheng | ICLR | [paper](https://openreview.net/forum?id=HDxgaKk956l) | [code](https://github.com/jegzheng/truncated-diffusion-probabilistic-models) |
| How Much is Enough? A Study on Diffusion Times in Score-based Generative Models | Giulio Franzese | Arxiv | [paper](https://arxiv.org/abs/2206.05173) | - |
| gDDIM: Generalized denoising diffusion implicit models | Qinsheng Zhang | ICLR | [paper](https://openreview.net/forum?id=1hKE9qjvz-) | [code](https://github.com/qsh-zh/gDDIM) |
| Pseudo Numerical Methods for Diffusion Models on Manifolds | Luping Liu | ICLR | [paper](https://arxiv.org/abs/2202.09778) | [code](https://github.com/luping-liu/PNDM) |
| Gotta Go Fast When Generating Data with Score-Based Models | Alexia Jolicoeur-Martineau | Arxiv | [paper](https://arxiv.org/abs/2105.14080) | [code](https://github.com/AlexiaJM/score_sde_fast_sampling) |
| Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality | Daniel Watson | ICLR | [paper](https://arxiv.org/abs/2202.05830) | - |
| Learning to Efficiently Sample from Diffusion Probabilistic Models | Daniel Watson | Arxiv | [paper](https://arxiv.org/abs/2106.03802) | - |
| On Distillation of Guided Diffusion Models | Chenlin Meng | CVPR | [paper](https://arxiv.org/abs/2210.03142) | - |


#### Model Compression

|  Title   | First Author | Conference/Journal | Link | Code |
|  ----  | ----  | ---- | ---- | ---- |
| Diffusion Probabilistic Model Made Slim | Xingyi Yang | CVPR | [paper](https://arxiv.org/pdf/2211.17106.pdf) | - |
| LOSSY COMPRESSION WITH GAUSSIAN DIFFUSION | Lucas Theis | ICLR | [paper](https://arxiv.org/abs/2206.08889) | - |
| PTQD: Accurate Post-Training Quantization for Diffusion Models | Yefei He | Arxiv | [paper](https://arxiv.org/abs/2305.10657) | - |
| Compressing and Accelerating Stable Diffusion | Alex Kashi | Arxiv | [paper](https://alexkashi.com/resources/projects/CS_242_Final_Project.pdf) | - |
| Q-Diffusion: Quantizing Diffusion Models | Xiuyu Li | Arxiv | [paper](https://arxiv.org/abs/2302.04304) | - |
| Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations | Yu-Hui Chen | Arxiv | [paper](https://arxiv.org/abs/2304.11267) | - |
| Token Merging for Fast Stable Diffusion | Daniel Bolya | Arxiv | [paper](https://arxiv.org/abs/2303.17604) | [code](https://github.com/dbolya/tomesd) |
| Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models | Weijia Wu | Arxiv | [code](https://arxiv.org/abs/2303.11681) | - | 
