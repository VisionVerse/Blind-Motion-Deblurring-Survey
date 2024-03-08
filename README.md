# [Deep Learning in Motion Deblurring: Current Status, Benchmarks and Future Prospects](https://arxiv.org/pdf/2401.05055.pdf) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)![Stars](https://img.shields.io/github/stars/VisionVerse/Blind-Motion-Deblurring-Survey)

:fire::fire: In this review, we have systematically examined **over 150 papers** :page_with_curl::page_with_curl::page_with_curl:, summarizing and analyzing :star2:**more than 30** blind motion deblurring methods. 

:fire::fire::fire: Extensive qualitative and quantitative comparisons have been conducted against the current SOTA methods on four datasets, highlighting their limitations and pointing out future research directions.



![avatar](/time_sequence.jpg)
**Fig 1.** Overview of deep learning methods for blind motion deblurring.


## Content:

1. <a href="#survey"> Related Reviews and Surveys to Deblurring </a>
2. <a href="#cnnmodels"> CNN-based  Blind Motion Deblurring Models </a>
3.  <a href="#rnnmodels"> RNN-based  Blind Motion Deblurring Models </a>
4. <a href="#ganmodels"> GAN-based  Blind Motion Deblurring Models </a>
5. <a href="#tmodels"> Transformer-based  Blind Motion Deblurring Models </a>
6. <a href="#tmodels"> Diffusion-based  Blind Motion Deblurring Models </a>
7. <a href="#datasets"> Motion Deblurring Datasets </a>
8. <a href="#evaluation"> Evaluation </a>
9. <a href="#citation"> Citation </a>

------

# Related Reviews and Surveys to Deblurring:  <a id="survey" class="anchor" href="#survey" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2023-12-28) :balloon:
**No.** | **Year** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-: 
01 | 2021 | CDS  |  A Survey on Single Image Deblurring | [Paper](https://ieeexplore.ieee.org/abstract/document/9463161)/Project
02 | 2021 | CVIU |  Single-image deblurring with neural networks: A comparative survey | [Paper](https://www.sciencedirect.com/science/article/pii/S1077314220301533)/Project
03 | 2022 | IJCV   |  Deep Image Deblurring: A Survey | [Paper](https://arxiv.org/abs/2201.10522)/Project
04 | 2022 | arXiv  |  Blind Image Deblurring: A Review | [Paper](https://arxiv.org/pdf/1604.07090.pdf)/Project
05 | 2023 | CVMJ  |  A survey on facial image deblurring| [Paper](https://arxiv.org/abs/2302.05017)/Project
06 | 2023 | arXiv  |  A Comprehensive Survey on Deep Neural Image Deblurring | [Paper](https://arxiv.org/abs/2310.04719)/Project

------












# CNN-based Blind Motion Deblurring Models:  <a id="cnnmodels" class="anchor" href="#CNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-08) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2017 | DeepDeblur| CVPR | Deep multi-scale convolutional neural network for dynamic scene deblurring | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)/[Project](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
02 | 2019 | DMPHN | CVPR | Deep stacked hierarchical multi-patch network for image deblurring | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deep_Stacked_Hierarchical_Multi-Patch_Network_for_Image_Deblurring_CVPR_2019_paper.pdf)/[Project](https://github.com/HongguangZhang/DMPHN-cvpr19-master)
03 | 2019 | PSS-NSC | CVPR | Dynamic scene deblurring with parameter selective sharing and nested skip connections | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Dynamic_Scene_Deblurring_With_Parameter_Selective_Sharing_and_Nested_Skip_CVPR_2019_paper.pdf)/[Project](https://github.com/firenxygao/deblur)
04 | 2020 | DGN | TIP | Dynamic scene deblurring by depth guided model | [Paper](https://ieeexplore.ieee.org/abstract/document/9043904)/Project
05 | 2020 | MSCAN | TCSVT | Deep convolutional-neural-network-based channel attention for single image dynamic scene blind deblurring | [Paper](https://ieeexplore.ieee.org/document/9247132)/[Project](https://github.com/karentwan/mscan)
06 | 2021 | SDWNet | ICCVW | Sdwnet: A straight dilated network with wavelet transformation for image deblurring | [Paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Zou_SDWNet_A_Straight_Dilated_Network_With_Wavelet_Transformation_for_Image_ICCVW_2021_paper.pdf)/[Project](https://github.com/FlyEgle/SDWNet)
07 | 2021 | MIMOU-Net+| ICCV | Rethinking coarse-to-fine approach in single image deblurring | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf)/[Project](https://github.com/chosj95/MIMO-UNet)
08 | 2021 | MPRNet | CVPR | Multi-stage progressive image restoration | [Paper](http://cg.postech.ac.kr/Research/MSSNet/MSSNet.pdf)/[Project](https://github.com/swz30/MPRNet)
09 | 2022 | MSSNet | ECCVW | Mssnet: Multi-scale-stage network for single image deblurring | [Paper]()/[Project](https://github.com/kky7/MSSNet)
10 | 2022 | HINet | CVPRW | Hinet: Half instance normalization network for image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Chen_HINet_Half_Instance_Normalization_Network_for_Image_Restoration_CVPRW_2021_paper.pdf)/[Project](https://github.com/megvii-model/HINet)
11 | 2022 | BANet | TIP | Banet: a blur-aware attention network for dynamic scene deblurring | [Paper](https://ieeexplore.ieee.org/document/9930938/)/[Project](https://github.com/pp00704831/BANet-TIP-2022)
12 | 2022 | IRNeXt | ICML | Irnext: Rethinking convolutional network design for image restoration | [Paper](https://openreview.net/pdf?id=MZkbgahv4a)/[Project](https://github.com/c-yn/IRNeXt)
13 | 2023 | ReLoBlur | AAAI | Real-World Deep Local Motion Deblurring | [Paper](https://arxiv.org/abs/2204.08179)/[Project](https://github.com/LeiaLi/ReLoBlur)
















# RNN-based Blind Motion Deblurring Models:  <a id="rnnmodels" class="anchor" href="#RNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-06) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2018 | SVRNN | CVPR | Dynamic scene deblurring using spatially variant recurrent neural networks | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Dynamic_Scene_Deblurring_CVPR_2018_paper.pdf)/[Project](https://github.com/zhjwustc/cvpr18_rnn_deblur_matcaffe)
02 | 2018 | SRN | CVPR| Scale-recurrent network for deep image deblurring | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tao_Scale-Recurrent_Network_for_CVPR_2018_paper.pdf)/[Project](https://github.com/jiangsutx/SRN-Deblur)
03 | 2023 | MT-RNN | ECCV | Multi-temporal recurrent neural networks for progressive non-uniform single image deblurring with incremental temporal training | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_20)/[Project](https://github.com/Dong1P/MTRNN)







# GAN-based Blind Motion Deblurring Models:  <a id="ganmodels" class="anchor" href="#GANmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-07) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2018 | DeblurGAN | CVPR | Deblurgan: Blind motion deblurring using conditional adversarial networks | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf)/[Project](https://github.com/KupynOrest/DeblurGAN)
02 | 2019 | DeblurGAN-V2 | ICCV | Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kupyn_DeblurGAN-v2_Deblurring_Orders-of-Magnitude_Faster_and_Better_ICCV_2019_paper.pdf)/[Project](https://github.com/VITA-Group/DeblurGANv2)
03 | 2020 | DBGAN| CVPR | Distribution-induced Bidirectional GAN for Graph Representation Learning | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Distribution-Induced_Bidirectional_Generative_Adversarial_Network_for_Graph_Representation_Learning_CVPR_2020_paper.pdf)/[Project](https://github.com/SsGood/DBGAN)
04 | 2021 | CycleGAN| ICCV | Unpaired image-to-image translation using cycle-consistent adversarial networks | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)/[Project](https://github.com/junyanz/CycleGAN)
05 | 2022 | FCLGAN| ACM | Unpaired image-to-image translation using cycle-consistent adversarial networks | [Paper](https://arxiv.org/pdf/2204.07820.pdf)/[Project](https://github.com/suiyizhao/FCL-GAN)
06 | 2022 | Ghost-DeblurGAN| IROS | Application of Ghost-DeblurGAN to Fiducial Marker Detection | [Paper](https://arxiv.org/pdf/2109.03379.pdf)/[Project](https://github.com/York-SDCNLab/Ghost-DeblurGAN)














# Transformer-based Blind Motion Deblurring Models:  <a id="tmodels" class="anchor" href="#Tmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-07) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2021 |Uformer| CVPR | Uformer: A general u-shaped transformer for image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/ZhendongWang6/Uformer)
02 | 2022 | Restormer | CVPR | Restormer: Efficient transformer for high-resolution image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/swz30/Restormer)
03 | 2022 | Stripformer | ECCV | Stripformer: Strip transformer for fast image deblurring | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_9)/[Project](https://github.com/pp00704831/Stripformer-ECCV-2022-)
04 | 2022 | Stoformer | NeurIPS | Stochastic Window Transformer for Image Restoration | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ca6d336ddaa316a6ae953a20b9477cf-Paper-Conference.pdf)/[Project](https://github.com/jiexiaou/Stoformer)
05 | 2023 | Sharpformer | TIP | SharpFormer: Learning Local Feature Preserving Global Representations for Image Deblurring | [Paper](https://ieeexplore.ieee.org/document/10124841)/[Project](https://github.com/qingsenyangit/SharpFormer)

06 | 2023 | FFTformer | CVPR | Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)/[Project](https://github.com/kkkls/fftformer)
07 | 2023 | BiT | CVPR | Blur Interpolation Transformer for Real-World Motion from Blur | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Blur_Interpolation_Transformer_for_Real-World_Motion_From_Blur_CVPR_2023_paper.pdf)/[Project](https://github.com/zzh-tech/bit)
08 | 2024 | | CVPR | Efficient Multi-scale Network with Learnable Discrete Wavelet Transform for Blind Motion Debluring| [Paper]/[Project]









# Diffusion-based Blind Motion Deblurring Models:  <a id="tmodels" class="anchor" href="#Tmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-07) :balloon:

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
01 | 2024 |ID-Blau| CVPR | ID-Blau: Image Deblurring by Implicit Diffusion-based reBLurring AUgmentation | [Paper]/[Project]
02 | 2024 || CVPR | Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring | [Paper]/[Project]



------













#  Motion Deblurring Datasets:  <a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-08) :balloon:
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Train/Val/Test**  | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :- | :-:
01   | [**Köhler at al.**](https://link.springer.com/chapter/10.1007/978-3-642-33786-4_3)   | 2012 | ECCV | 4 sharp, 48 blur | Synthetic | -  | [link]()
02   | [**GoPro**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)   | 2017 | CVPR | 3214 | Synthetic | 2103/0/1111  | [link](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
03 | [**HIDE**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)   | 2019 | CVPR | 8422 | Synthetic | 6397/0/2025  | [link](https://github.com/joanshen0508/HA_deblur)
04 | [**Blur-DVS**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Learning_Event-Based_Motion_Deblurring_CVPR_2020_paper.pdf) | 2020 | CVPR | 13358 | Real | 8878/1120/3360  | [link]
05 | [**RealBlur**](https://link.springer.com/chapter/10.1007/978-3-030-58595-2_12)   | 2020 | ECCV   | 4738 | Real | 3758/0/980  | [link](https://github.com/rimchang/RealBlur)
06 | [**RsBlur**](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670481.pdf)   | 2022 | ECCV | 13358 | Real | 8878/1120/3360 | [link](https://github.com/rimchang/RSBlur)
07 | [**ReLoBlur**](https://arxiv.org/abs/2204.08179) | 2023 | AAAI | 2405 | Real | 2010/0/395 | [link](https://github.com/LeiaLi/ReLoBlur)

------

# Evaluation:  <a id="evaluation" class="anchor" href="#evaluation" aria-hidden="true"><span class="octicon octicon-link"></span></a>  

* For evaluation on **GoPro** results in MATLAB, modify './out/...' to the corresponding path
```matlab
evaluation_GoPro.m
```
* For evaluation on **HIDE** results in MATLAB, modify './out/...' to the corresponding path
```matlab
evaluation_HIDE.m
```
* For evaluation on **RealBlur_J** results, modify './out/...' to the corresponding path
```python
python evaluate_RealBlur_J.py
```
* For evaluation on **RealBlur_R** results, modify './out/...' to the corresponding path
```python
python evaluate_RealBlur_R.py
```
    

------

# Citation: <a id="citation" class="anchor" href="#citation" aria-hidden="true"><span class="octicon octicon-link"></span></a> 

If you find our survey paper and evaluation code are useful, please cite the following paper:
```BibTeX
@article{xiang2024application,
      title={Application of Deep Learning in Blind Motion Deblurring: Current Status and Future Prospects}, 
      author={Yawen Xiang and Heng Zhou and Chengyang Li and Fangwei Sun and Zhongbo Li and Yongqiang Xie},
      year={2024},
      journal={arXiv preprint arXiv:2401.05055},
}
```


--------------------------------------------------------------------------------------

# :clap::clap::clap: Thanks to the above authors for their excellent work！
