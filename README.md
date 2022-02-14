# STMask

The code is implmented for our paper in CVPR2021:
 - [STMask: Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation](http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm)

![image](https://github.com/MinghanLi/STMask/blob/main/images/overall1.png)

# News
- [27/06/2021] **!Important issue:** For previous results of YTVIS2021 and OVIS datasets, we use the bounding boxes with normalization in the function bbox_feat_extractor() of track_to_segmetn_head.py by mistake. _However, the bounding boxes in bbox_feat_extractor() function should not be normalized._ We update the results and trained models for YTVIS2021 and OVIS datasets. Apologize for our negligence.
- [12/06/2021] Update the solution for the error in deform_conv_cuda.cu 
- [22/04/2021] Add experimental results on [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/) datasets
- [14/04/2021] Release code on Github and paper on arxiv

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/MinghanLi/STMask.git
   cd STMask
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
     - conda activate STMask-env
   - Manually with pip
     - Set up a Python3 environment.
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
       
 - Install mmcv and mmdet
    - According to your Cuda and pytorch version to install mmcv or mmcv-full from [here](https://github.com/open-mmlab/mmcv). Here my cuda and torch version are 10.1 and 1.5.0 respectively. 
      ```Shell
      pip install mmcv-full==1.1.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
      ```
    - install cocoapi and a customized COCO API for YouTubeVIS dataset from [here](https://github.com/youtubevos/cocoapi)
      ```Shell
      pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
      git clone https://github.com/youtubevos/cocoapi
      cd cocoapi/PythonAPI
      # To compile and install locally 
      python setup.py build_ext --inplace
      # To install library to Python site-packages 
      python setup.py build_ext install
      ```

 - Install spatial-correlation-sampler 
      ```Shell
      pip install spatial-correlation-sampler
      ```
 
 - Complie DCNv2 code (see [Installation](https://github.com/dbolya/yolact#installation))
   - Download code for deformable convolutional layers from [here](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)
     ```Shell
     git clone https://github.com/CharlesShang/DCNv2.git
     cd DCNv2
     python setup.py build develop
     ```

- Modify mmcv/ops/deform_conv.py to handle deformable convolution with different height and width (like 3 * 5) in FCB(ali) or FCB(ada)
  - Open the file deform_conv.py 
    ```Shell
    vim /your_conda_env_path/mmcv/ops/deform_conv.py
    ```
  - Replace padW=ctx.padding[1], padH=ctx.padding[0] with padW=ctx.padding[0], padH=ctx.padding[1], taking Line 81-89 as an example:
    ```Shell
    ext_module.deform_conv_forward(
            input,
            weight,
            offset,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[0],
            padH=ctx.padding[1],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            im2col_step=cur_im2col_step)
    ```

# Dataset
 - If you'd like to train STMask, please download the datasets from the official web: [YTVIS2019](https://youtube-vos.org/dataset/), [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).


# Evaluation 
The input size on all VIS benchmarks is 360*640 here.
## Quantitative Results on YTVIS2019 ((trained with 12 epoches))
Here are our STMask models (released on April, 2021) along with their FPS on a 2080Ti and mAP on `valid set`, where mAP and mAP* are obtained under cross class fast nms and fast nms respectively. 
Note that FCB(ali) and FCB(ada) are only executed on the classification branch. 

| Backbone      | FCA  | FCB      | TF | FPS  | mAP  | mAP* | Weights |                                                                                                        
|:-------------:|:----:|:----:    |----|------|------|------|-----------------------------------------------------------------------------------------------------------|
| R50-DCN-FPN   | FCA  | -        | TF | 29.3 | 32.6 | 33.4 | [STMask_plus_resnet50.pth](https://drive.google.com/file/d/14RHpTHA5GZGbuyHzc3bgn0luTsl4STuo/view?usp=sharing) | 
| R50-DCN-FPN   | FCA  | FCB(ali) | TF | 27.8 | -    | 32.1 | [STMask_plus_resnet50_ali.pth](https://drive.google.com/file/d/1ZWIH0oFL8kNU4roe-kwiL9Q8lGgZ8MwJ/view?usp=sharing) | 
| R50-DCN-FPN   | FCA  | FCB(ada) | TF | 28.6 | 32.8 | 33.0 | [STMask_plus_resnet50_ada.pth](https://drive.google.com/file/d/1fxkEtjiIwwqgc-wh-e-7OPLBNjDu-faA/view?usp=sharing) |
| R101-DCN-FPN  | FCA  | -        | TF | 24.5 | 36.0 | 36.3 | [STMask_plus_base.pth](https://drive.google.com/file/d/1qgq8yC8otUMJMsffsaC288YOAwYf3OIz/view?usp=sharing) |    
| R101-DCN-FPN  | FCA  | FCB(ali) | TF | 22.1 | 36.3 | 37.1 | [STMask_plus_base_ali.pth](https://drive.google.com/file/d/1ZWIH0oFL8kNU4roe-kwiL9Q8lGgZ8MwJ/view?usp=sharing)  |   
| R101-DCN-FPN  | FCA  | FCB(ada) | TF | 23.4 | 36.8 | 37.9 | [STMask_plus_base_ada.pth](https://drive.google.com/file/d/1Y5TZZVY9BF2Jq1F2_g4ZRfrmy6B6hEVn/view?usp=sharing)  |   
****
## Quantitative Results on YTVIS2021 (trained with 12 epoches)
| Backbone      | FCA  | FCB      | TF | mAP* | Weights | Results |                                                                                                         
|:-------------:|:----:|:----:    |----|------|-----------------------------------------------------------------------------------------------------------|------|
| R50-DCN-FPN   | FCA  | -        | TF | 30.6 | [STMask_plus_resnet50_YTVIS2021.pth](https://drive.google.com/file/d/1i_Wy2z2H_Z9vSf8oNqikhVNdVqSomoVP/view?usp=sharing) | - |
| R50-DCN-FPN   | FCA  | FCB(ada) | TF | 31.1 | [STMask_plus_resnet50_ada_YTVIS2021.pth](https://drive.google.com/file/d/1XxP8rVwjH2-aMfKtX3gW_89Z7-5ZiNjt/view?usp=sharing) | [stdout.txt](https://drive.google.com/file/d/1jPVMy5CdrClti0XvhKUUz11QFa-W5Zpd/view?usp=sharing) |
| R101-DCN-FPN  | FCA  | -        | TF | 33.7 | [STMask_plus_base_YTVIS2021.pth](https://drive.google.com/file/d/1gXyo25muXlFxjuXKO_MblKsvDramBGC8/view?usp=sharing) | - |
| R101-DCN-FPN  | FCA  | FCB(ada) | TF | 34.6 | [STMask_plus_base_ada_YTVIS2021.pth](https://drive.google.com/file/d/1EcZcvSdPR6aMl1hpBIGY2S9IUHmonzR5/view?usp=sharing)  |  [stdout.txt](https://drive.google.com/file/d/1i3cd28-FbIu6eXZUbgIZMO72IxRQZi7h/view?usp=sharing)


## Quantitative Results on OVIS (trained with 20 epoches)
| Backbone      | FCA  | FCB      | TF | mAP* | Weights | Results|                                                                                                        
|:-------------:|:----:|:----:    |----|------|-----------------------------------------------------------------------------------------------------------|------|
| R50-DCN-FPN   | FCA  | -        | TF | 15.4 | [STMask_plus_resnet50_OVIS.pth](https://drive.google.com/file/d/18aij_6YboxVzScc0Nmcb50nb0uczal6P/view?usp=sharing) | - |
| R50-DCN-FPN   | FCA  | FCB(ada) | TF | 15.4 | [STMask_plus_resnet50_ada_OVIS.pth](https://drive.google.com/file/d/10qx2dBeksHlNxG5nmPaNOyz35Bl6u2uQ/view?usp=sharing) | [stdout.txt](https://drive.google.com/file/d/1jD_O1YMERQqXwi1m90PnfUj3NNduWz1W/view?usp=sharing) |
| R101-DCN-FPN  | FCA  | -        | TF | 17.3 | [STMask_plus_base_OVIS.pth](https://drive.google.com/file/d/1lw2YSEO58kDOgqtg6tlZueCCQkLMV1QX/view?usp=sharing) | [stdout.txt](https://drive.google.com/file/d/1-1z8hPzMPWZp_-Rk9kxILCSkB8Ec7OM0/view?usp=sharing) |
| R101-DCN-FPN  | FCA  | FCB(ada) | TF | 15.8 | [STMask_plus_base_ada_OVIS.pth](https://drive.google.com/file/d/1zJazhMlrYHydnHggKhBjiICVVWpr5djS/view?usp=sharing)  | - |  


To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `STMask_plus_base` for `STMask_plus_base.pth`). 
Here all STMask models are trained based on `yolact_plus_base_54_80000.pth` or `yolact_plus_resnet_54_80000.pth` from Yolact++ [here](https://github.com/dbolya/yolact). 

## Quantitative Results on COCO

We also provide quantitative results of Yolcat++ with our proposed feature calibration for anchors and boxes on COCO (w/o temporal fusion module). Here are the results on COCO valid set.

| Image Size        | Backbone      | FCA  | FCB      | B_AP | M_AP | Weights |                                                                                                         
|:----------:       |:-------------:|:----:|:----:    |------|------|---------------------------------------------------------------------------------------------------------------|
| [550,550]         | R50-DCN-FPN   | FCA  | -        | 34.5 | 32.9 |[yolact_plus_resnet50_54.pth](https://drive.google.com/file/d/18bGj_pgKGojtnn8ni5XPbAUBNWGHkQbN/view?usp=sharing) |
| [550,550]         | R50-DCN-FPN   | FCA  | FCB(ali) | 34.6 | 33.3 |[yolact_plus_resnet50_ali_54.pth](https://drive.google.com/file/d/1iHefY01fhLE3OqMeb20guGD4U3HsxHmR/view?usp=sharing) | 
| [550,550]         | R50-DCN-FPN   | FCA  | FCB(ada) | 34.7 | 33.2 |[yolact_plus_resnet50_ada_54.pth](https://drive.google.com/file/d/12nEvCra-nU2nPQn0RN5OT_NXF-in1VzL/view?usp=sharing)  |
| [550,550]         | R101-DCN-FPN  | FCA  | -        | 35.7 | 33.3 |[yolact_plus_base_54.pth](https://drive.google.com/file/d/1TwtfP89h4-UJawsOetvSkVJmwZbjWHkk/view?usp=sharing) |    
| [550,550]         | R101-DCN-FPN  | FCA  | FCB(ali) | 35.6 | 34.1 |[yolact_plus_base_ali_54.pth](https://drive.google.com/file/d/1wvCSvRyMDKfxf1an9xTzquUQpX_azalR/view?usp=sharing)  |   
| [550,550]         | R101-DCN-FPN  | FCA  | FCB(ada) | 36.4 | 34.8 |[yolact_plus_baseada_54.pth](https://drive.google.com/file/d/1xpIeTe2kUMcyw0Ud0nbHZJlXHhBywrfM/view?usp=sharing)  |   


# Inference
```Shell
# Output a YTVOSEval json to submit to the website.
# This command will create './weights/results.json' for instance segmentation.
python eval.py --config=STMask_plus_base_ada_config --trained_model=weights/STMask_plus_base_ada.pth --mask_det_file=weights/results.json
```

```Shell
# Output a visual segmentation results
python eval.py --config=STMask_plus_base_ada_config --trained_model=weights/STMask_plus_base_ada.pth --mask_det_file=weights/results.json --display
```

# Training
By default, we train on YouTubeVOS2019 dataset. Make sure to download the entire dataset using the commands above.
 - To train, grab an COCO-pretrained model and put it in `./weights`.
   - [Yolcat++]: For Resnet-50/-101, download `yolact_plus_base_54_80000.pth` or `yolact_plus_resnet_54_80000.pth` from Yolact++ [here](https://github.com/dbolya/yolact).
   - [Yolcat++ & FC]: Alternatively, you can use those Yolact++ with FC models on Table. 2 for training, which can obtain a relative higher performance than that of Yolact++ models.


- Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains STMask_plus_base_config with a batch_size of 8.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config=STMask_plus_base_config --batch_size=8 --lr=1e-4 --save_folder=weights/weights_r101


# Resume training STMask_plus_base with a specific weight file and start from the iteration specified in the weight file's name.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config=STMask_plus_base_config --resume=weights/STMask_plus_base_10_32100.pth 
```

# Citation
If you use STMask or this code base in your work, please cite
```
@inproceedings{STMask-CVPR2021,
  author    = {Minghan Li and Shuai Li and Lida Li and Lei Zhang},
  title     = {Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation},
  booktitle = {CVPR},
  year      = {2021},
}
```


# Contact
For questions about our paper or code, please contact Li Minghan (liminghan0330@gmail.com or minghancs.li@connect.polyu.hk).
