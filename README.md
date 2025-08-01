# CTM-UNet
This is This is the official code for the paper "CTM-UNet: A medical image segmentation network based on multi-scale information awareness and semantic enhancement


**Data Format**

Refer to the UNeXt codes if you run into problems.
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...

            
**Using the code**

The code is stable while using Python 3.6.13, CUDA >=10.1


**If you prefer pip, install following versions:**

timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
