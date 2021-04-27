# STMask

The code is implmented for our paper in CVPR2021:
 - [STMask: Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation](http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm)

![image](https://github.com/MinghanLi/STMask/blob/main/images/overall1.png)

# News
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

# Dataset
 - If you'd like to train STMask, please download the datasets from the official web: [YTVIS2019](https://youtube-vos.org/dataset/), [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).


# Evaluation 
## Quantitative Results on YTVIS2019
Here are our STMask models (released on April, 2021) along with their FPS on a 2080Ti and mAP on `valid set`, where mAP and mAP* are obtained under cross class fast nms and fast nms respectively. 
Note that FCB(ali) and FCB(ada) are only executed on the classification branch.

| Image Size       | Backbone      | FCA  | FCB      | TF | FPS  | mAP  | mAP* | Weights |                                                                                                         
|:----------:      |:-------------:|:----:|:----:    |----|------|------|------|-----------------------------------------------------------------------------------------------------------|
| [384,640]        | R50-DCN-FPN   | FCA  | -        | TF | 29.3 | 32.6 | 33.4 | [STMask_plus_resnet50.pth](https://drive.google.com/file/d/1R_SturnDgIPqPp8L5m6BUT44QO2QvsW6/view?usp=sharing) |
| [384,640]        | R50-DCN-FPN   | FCA  | FCB(ali) | TF | 27.8 | -    | 32.1 | [STMask_plus_resnet50_ali.pth](https://drive.google.com/file/d/1J9L2oDNqm40wwzKn1iIwvXnPIr5n1kQP/view?usp=sharing) | 
| [384,640]        | R50-DCN-FPN   | FCA  | FCB(ada) | TF | 28.6 | 32.8 | 33.0 | [STMask_plus_resnet50_ada.pth](https://drive.google.com/file/d/1HbtRX3sH_3CZTAjuIIMv8hIdTuqItfwq/view?usp=sharing) |
| [384,640]        | R101-DCN-FPN  | FCA  | -        | TF | 24.5 | 36.0 | 36.3 | [STMask_plus_base.pth](https://drive.google.com/file/d/1R_SturnDgIPqPp8L5m6BUT44QO2QvsW6/view?usp=sharing) |    
| [384,640]        | R101-DCN-FPN  | FCA  | FCB(ali) | TF | 22.1 | 36.3 | 37.1 | [STMask_plus_base_ali.pth](https://drive.google.com/file/d/1Cza-I9xAqkzXKlaTJrSIzlP4pcC0og4f/view?usp=sharing)  |   
| [384,640]        | R101-DCN-FPN  | FCA  | FCB(ada) | TF | 23.4 | 36.8 | 37.9 | [STMask_plus_base_ada.pth](https://drive.google.com/file/d/1ZjQWiURoHZnyafWaPzgvYSEibk77rGDa/view?usp=sharing)  |   

## Quantitative Results on YTVIS2021 
| Image Size       | Backbone      | FCA  | FCB      | TF | mAP* | Weights |                                                                                                         
|:----------:      |:-------------:|:----:|:----:    |----|------|-----------------------------------------------------------------------------------------------------------|
| [384,640]        | R50-DCN-FPN   | FCA  | -        | TF | 29.2 | [STMask_plus_resnet50_YTVIS2021.pth](https://drive.google.com/file/d/1Rmz2-qtMlHPrTRVTU1gByHaBM0608u15/view?usp=sharing) |
| [384,640]        | R50-DCN-FPN   | FCA  | FCB(ada) | TF | 31.1 | [STMask_plus_resnet50_ada_YTVIS2021.pth](https://drive.google.com/file/d/1q50kkB-GY30Gx6o278a0mxHmz_0Yz3t_/view?usp=sharing) |
| [384,640]        | R101-DCN-FPN  | FCA  | -        | TF | 32.4 | [STMask_plus_base_YTVIS2021.pth](https://drive.google.com/file/d/1iBtnE1vX3-8hV69lL4RYfYkz2R8ZZj6J/view?usp=sharing) |    
| [384,640]        | R101-DCN-FPN  | FCA  | FCB(ada) | TF | 32.7 | [STMask_plus_base_ada_YTVIS2021.pth](https://drive.google.com/file/d/1o99-Cg8L7MIzAP-Rjm06G93HitGadJMi/view?usp=sharing)  |   


## Quantitative Results on OVIS 
| Image Size       | Backbone      | FCA  | FCB      | TF | mAP* | Weights |                                                                                                         
|:----------:      |:-------------:|:----:|:----:    |----|------|-----------------------------------------------------------------------------------------------------------|
| [384,640]        | R50-DCN-FPN   | FCA  | -        | TF | 13.1 | [STMask_plus_resnet50_OVIS.pth](https://drive.google.com/file/d/1PDLajHIyzsTxu3dZwnUXJReI0mXYsSQq/view?usp=sharing) |
| [384,640]        | R50-DCN-FPN   | FCA  | FCB(ada) | TF | 13.0 | [STMask_plus_resnet50_ada_OVIS.pth](https://drive.google.com/file/d/10qx2dBeksHlNxG5nmPaNOyz35Bl6u2uQ/view?usp=sharing)|
| [384,640]        | R101-DCN-FPN  | FCA  | -        | TF | 15.1 | [STMask_plus_base_OVIS.pth](https://drive.google.com/file/d/1lw2YSEO58kDOgqtg6tlZueCCQkLMV1QX/view?usp=sharing) |    
| [384,640]        | R101-DCN-FPN  | FCA  | FCB(ada) | TF | 13.9 | [STMask_plus_base_ada_OVIS.pth](https://drive.google.com/file/d/1lDzOe1ASykeOco0M-h7XWsbPEYStuTzK/view?usp=sharing)  |   


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
