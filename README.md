# Application of Deep Learning in Blind Motion Deblurring: Current Status and Future Prospects

This is a survey that reviews deep learning models and benchmark datasets related to blind motion deblurring and provides a comprehensive evaluation of these models.

![avatar](/time_sequence.jpg)
**Fig 1.** Overview of deep learning methods for blind motion deblurring.

## Content:

1. <a href="#survey"> Related Reviews and Surveys to Deblurring </a>
2. <a href="#cnnmodels"> CNN-based  Blind Motion Deblurring Models </a>
3.  <a href="#rnnmodels"> RNN-based  Blind Motion Deblurring Models </a>
4. <a href="#ganmodels"> GAN-based  Blind Motion Deblurring Models </a>
5. <a href="#tmodels"> Transformer-based  Blind Motion Deblurring Models </a>
6. <a href="#datasets"> Motion Deblurring Datasets </a>
8. <a href="#evaluation"> Evaluation </a>
9. <a href="#citation"> Citation </a>

------

# Related Reviews and Surveys to Deblurring:  <a id="survey" class="anchor" href="#survey" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2023-12-28)
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
:rocket::rocket::rocket:Update (in 2024-01-08)

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
3 | 2021 | MIMOU-Net+| ICCV | Rethinking coarse-to-fine approach in single image deblurring | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf)/[Project](https://github.com/chosj95/MIMO-UNet)
3 | 2021 | MIMOU-Net+| ICCV | Rethinking coarse-to-fine approach in single image deblurring | [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cho_Rethinking_Coarse-To-Fine_Approach_in_Single_Image_Deblurring_ICCV_2021_paper.pdf)/[Project](https://github.com/chosj95/MIMO-UNet)
2 | 2019 | DMPHN | CVPR | Deep stacked hierarchical multi-patch network for image deblurring | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deep_Stacked_Hierarchical_Multi-Patch_Network_for_Image_Deblurring_CVPR_2019_paper.pdf)/[Project](https://github.com/HongguangZhang/DMPHN-cvpr19-master)
1 | 2017 | DeepDeblur| CVPR | Deep multi-scale convolutional neural network for dynamic scene deblurring | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)/[Project](https://github.com/SeungjunNah/DeepDeblur-PyTorch)












# RNN-based Blind Motion Deblurring Models:  <a id="rnnmodels" class="anchor" href="#RNNmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-06)

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
3 | 2023 | XMSNet| ACM MM | Objexxx Semantics | [Paper](https://arxiv.org/pdf/2305.10469.pdf)/[Project](https://github.com/Zongwei97/XMSNet)
2 | 2018 |RNN_deblur| CVPR| Dynamic scene deblurring using spatially variant recurrent  neural networks | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Dynamic_Scene_Deblurring_CVPR_2018_paper.pdf)/[Project](https://github.com/zhjwustc/cvpr18_rnn_deblur_matcaffe)
1 | 2023 | MT-RNN | ECCV | Multi-temporal recurrent neural networks for progressive non-uniform single image deblurring with incremental temporal training | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58539-6_20)/[Project](https://github.com/Dong1P/MTRNN)








# GAN-based Blind Motion Deblurring Models:  <a id="ganmodels" class="anchor" href="#GANmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-06)

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
6 | 2022 | Ghost-DeblurGAN| IROS | Application of Ghost-DeblurGAN to Fiducial Marker Detection | [Paper](https://arxiv.org/pdf/2109.03379.pdf)/[Project](https://github.com/York-SDCNLab/Ghost-DeblurGAN)
5 | 2022 | FCLGAN| ACM | Unpaired image-to-image translation using cycle-consistent adversarial networks | [Paper](https://arxiv.org/pdf/2204.07820.pdf)/[Project](https://github.com/suiyizhao/FCL-GAN)
4 | 2021 | CycleGAN| ICCV | Unpaired image-to-image translation using cycle-consistent adversarial networks | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)/[Project](https://github.com/junyanz/CycleGAN)
3 | 2020 | DBGAN| CVPR | Distribution-induced Bidirectional GAN for Graph Representation Learning | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Distribution-Induced_Bidirectional_Generative_Adversarial_Network_for_Graph_Representation_Learning_CVPR_2020_paper.pdf)/[Project](https://github.com/SsGood/DBGAN)
2 | 2019 | DeblurGAN-V2 | ICCV | Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kupyn_DeblurGAN-v2_Deblurring_Orders-of-Magnitude_Faster_and_Better_ICCV_2019_paper.pdf)/[Project](https://github.com/VITA-Group/DeblurGANv2)
1 | 2018 | DeblurGAN | CVPR | Deblurgan: Blind motion deblurring using conditional adversarial networks | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf)/[Project](https://github.com/KupynOrest/DeblurGAN)










# Transformer-based Blind Motion Deblurring Models:  <a id="tmodels" class="anchor" href="#Tmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-06)

**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
7 | 2023 | BiT | CVPR | Blur Interpolation Transformer for Real-World Motion from Blur | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhong_Blur_Interpolation_Transformer_for_Real-World_Motion_From_Blur_CVPR_2023_paper.pdf)/[Project](https://github.com/zzh-tech/bit)
6 | 2023 | FFTformer | CVPR | Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)/[Project](https://github.com/kkkls/fftformer)
5 | 2023 | Sharpformer | TIP | SharpFormer: Learning Local Feature Preserving Global Representations for Image Deblurring | [Paper](https://ieeexplore.ieee.org/document/10124841)/[Project](https://github.com/qingsenyangit/SharpFormer)
4 | 2022 | Stoformer | NeurIPS | Stochastic Window Transformer for Image Restoration | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3ca6d336ddaa316a6ae953a20b9477cf-Paper-Conference.pdf)/[Project](https://github.com/jiexiaou/Stoformer)
3 | 2022 | Stripformer | ECCV | Stripformer: Strip transformer for fast image deblurring | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_9)/[Project](https://github.com/pp00704831/Stripformer-ECCV-2022-)
2 | 2022 | Restormer | CVPR | Restormer: Efficient transformer for high-resolution image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/swz30/Restormer)
1 | 2021 |Uformer| CVPR | Uformer: A general u-shaped transformer for image restoration | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf)/[Project](https://github.com/ZhendongWang6/Uformer)

------













#  Motion Deblurring Datasets:  <a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:rocket::rocket::rocket:Update (in 2024-01-08)
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Resolution** | **Download**
:-: | :-: | :-: | :-  | :-  | :-:| :-: | :-:
1   | [**GoPro**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)   |2017 | CVPR   | 3214 | Synthetic | [251-1200] * [222-900] | [link](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
2 | [**HIDE**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)   |2019 |CVPR   | 8422 | Synthetic | [251-1200] * [222-900] | [link](https://github.com/joanshen0508/HA_deblur)
3 | [**RealBlur**](https://link.springer.com/chapter/10.1007/978-3-030-58595-2_12)   |2020 | ECCV   | 4738 | Real | [251-1200] * [222-900] | [link]()


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

```


--------------------------------------------------------------------------------------

# :clap::clap::clap: Thanks to the above authors for their excellent work！
