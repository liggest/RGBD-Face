from RGBDFace.utils.image import prepareRaw
import os

'''
数据集目录结构：
datasets              根目录
  |-xxx               数据集目录
    |-rgb             彩色图像
    |-depth           深度图像
    |-intrinsic.json  图像尺寸和相机内参

数据样例参见：https://github.com/tencent-ailab/hifi3dface
拍摄用APP：https://github.com/lxk121lalala/RGBD_data_capture

对于拍得的原始数据，可将其置入数据集目录，使用如下函数
'''

datasetBase="./datasets/"
dataset="xxx"
color="rgb"
depth="depth"
prepareRaw(os.path.join(datasetBase,dataset),color,depth)


