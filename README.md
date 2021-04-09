# STMask

The code is implmented for our paper in CVPR2021:
 - [STMask: Spatial Feature Calibration and Temporal Fusion for Effective One-stage Video Instance Segmentation](http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm)

![image](https://github.com/MinghanLi/STMask/blob/main/images/overall1.png)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/MinghanLi/STMask.git
   cd STMask
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f env.yml`
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
    - install mmcv or mmcv-full from [here](https://github.com/open-mmlab/mmcv)
      ```Shell
      pip install mmcv-full==1.1.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
      ```
    - install cocoapi
      ```Shell
      pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
      cd cocoapi/PythonAPI
      pip install -v -e .  # or "python setup.py develop"
      ```
 
 - Complie DCNv2 code (see [Installation](https://github.com/dbolya/yolact#installation))
   - Download code for deformable convolutional layers from [here](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)
     ```Shell
     git clone https://github.com/CharlesShang/DCNv2.git
     cd DCNv2
     python setup.py build develop
     ```

# Dataset
 - If you'd like to train STMask, download the YTVOS2019 dataset from [the official web](https://youtube-vos.org/dataset/).


# Evaluation
Here are our STMask models (released on April, 2021) along with their FPS on a 2080Ti and mAP on `validset`:

| Image Size       | Backbone      | FCA  | FCB      | TF | FPS  | mAP  | Weights |                                                                                                         
|:----------:      |:-------------:|:----:|:----:    |----|------|------|-----------------------------------------------------------------------------------------------------------|
| [384,640]        | Resnet50-FPN  | FCA  | -        | TF | 29.3 | 31.6 | [STMask_r50_FCA+TF.pth](https://drive.google.com/file/d/1TCiW-EQLEh1SrN-o7cOvKEFQy2WhkeSL/view?usp=sharing) |
| [384,640]        | Resnet50-FPN  | FCA  | FCB(ali) | TF | -    | -    | [STMask_r50_ali.pth]() | 
| [384,640]        | Resnet50-FPN  | FCA  | FCB(ada) | TF | 28.6 | 33.5 | [STMask_r50_ada.pth]()  |
| [384,640]        | Resnet101-FPN | FCA  | -        | TF | 24.5 | 36.0 | [STMask_r101_FCA+TF.pth](https://drive.google.com/file/d/1qgq8yC8otUMJMsffsaC288YOAwYf3OIz/view?usp=sharing) |    
| [384,640]        | Resnet101-FPN | FCA  | FCB(ali) | TF | 22.1 | 36.3 | [STMask_r101_ali.pth]()  |   
| [384,640]        | Resnet101-FPN | FCA  | FCB(ada) | TF | 23.4 | 36.8 | [STMask_r101_ada.pth]()  |   

To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `STMask_plus_base` for `STMask_r101_FCA+TF.pth`).
## Quantitative Results on YTVOS2019
```Shell
# Output a YTVOSEval json to submit to the website.
# This command will create './weights/results.json' for instance segmentation.
python eval.py --trained_model=weights/STMask_r101_FCA+TF.pth --mask_det_file=weights/results.json
```

# Training
By default, we train on YouTubeVOS2019 dataset. Make sure to download the entire dataset using the commands above.
 - To train, grab an COCO-pretrained model and put it in `./weights`.
   - For Resnet50, download `yolact_plus_resnet50_54.pth` from [here](https://drive.google.com/file/d/18bGj_pgKGojtnn8ni5XPbAUBNWGHkQbN/view?usp=sharing).
   - For Resnet101, download `yolact_plus_base_54_80000.pth` from [here](https://github.com/dbolya/yolact).
- Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains yolact_base_config with a batch_size of 8.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config=STMask_plus_base_config --batch_size=8 --save_folder=weights/weights_r101


# Resume training STMask_base with a specific weight file and start from the iteration specified in the weight file's name.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config=STMask_plus_base_config --resume=weights/STMask_plus_base_10_32100.pth 
```

# Citation
If you use STMask or this code base in your work, please cite
```
@inproceedings{STMask-CVPR2021,
  author    = {Minghan Li and Shuai Li and Lida Li and Lei Zhang},
  title     = {Spatial Feature Calibration and Temporal Fusion for Effective \\ One-stage Video Instance Segmentation},
  booktitle = {CVPR},
  year      = {2021},
}
```


# Contact
For questions about our paper or code, please contact Li Minghan (liminghan0330@gmail.com or minghancs.li@connect.polyu.hk).
