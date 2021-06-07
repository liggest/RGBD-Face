from .general import createPath
import numpy as np
import cv2
import open3d as o3d
import os
from PIL import Image

from skimage.color import rgb2gray
from skimage import img_as_ubyte,morphology

from .depth import get16BitDepth

def getImg(path,useO3D=True):
    '''
        使用open3d读取指定路径的图像，返回ndarray或open3d的Image
    '''
    img=o3d.io.read_image(path)
    return img if useO3D else np.asarray(img)

def getImgPair(colorPath,depthPath,depth16bit=True,useO3D=True):
    '''
        得到一对彩色、深度图，depth16bit=True则将深度图处理为int16
    '''
    color=getImg(colorPath,useO3D)
    depth=getImg(depthPath,useO3D)
    #color=o3d.io.read_image(colorPath)
    #depth=o3d.io.read_image(depthPath)
    if depth16bit:
        depth=get16BitDepth(depth)
    return color,depth

def getConvexHullMask(img,points):
    '''
        得到点集的凸包，返回和图像大小相同的遮罩
    '''
    mask=np.zeros_like(img,dtype=np.ubyte)
    hull=cv2.convexHull(points)
    return cv2.fillConvexPoly(mask, hull, (1,))

def getMaskedImg(img,mask,reverse=False):
    '''
        将遮罩应用到图（靠对应下标赋值）    
        mask像素少的时候较快
    '''
    imMasked=np.zeros_like(img)
    if reverse:
        idxs=np.where(mask==0)
    else:
        idxs=np.where(mask==1)
    imMasked[idxs]=img[idxs]
    return imMasked

def getMaskedImg2(img,mask,reverse=False):
    '''
        将遮罩应用到图（靠对应项相乘）    
        mask像素多的时候稍快
    '''
    if reverse:
        mask=~mask
    if img.ndim>2:
        c=img.shape[-1]
        mask=np.tile(np.expand_dims(mask,2),(1,1,c))
    return img * mask

def imgAsIntGray(img):
    '''
        尝试把图像转换为uint8的灰度图
    '''
    img=np.asarray(img)
    if img.ndim>2: #非灰度图转成灰度图
        img=rgb2gray(img)
    if img.max()<=1: #float的转成uint8
        img=img_as_ubyte(img)
    return img

def maskDilation(mask,size=2):
    '''
        使用size大小的圆膨胀mask
    '''
    disk=morphology.disk(size)
    maskDila=morphology.binary_dilation(mask,selem=disk)
    return maskDila

def maskErosion(mask,size=2):
    '''
        使用size大小的圆腐蚀mask
    '''
    disk=morphology.disk(size)
    maskEros=morphology.binary_erosion(mask,selem=disk)
    return maskEros

def maskOpening(mask,size=2):
    '''
        使用size大小的圆对mask做开运算
    '''
    disk=morphology.disk(size)
    maskOpen=morphology.binary_opening(mask,disk)
    return maskOpen

def maskClosing(mask,size=2):
    '''
        使用size大小的圆对mask做闭运算
    '''
    disk=morphology.disk(size)
    maskClose=morphology.binary_closing(mask,disk)
    return maskClose

def prepareRaw(datasetPath,color,depth,depthfirst=True):
    '''
        处理APP拍摄的原始图像，将其转正、把彩色图和深度图区分开
        
        depthfirst = True 即深度图序号在彩色图前
    '''
    if not os.path.exists(datasetPath):
        raise FileNotFoundError("没找到数据集目录")
    datalist=sorted( os.listdir(datasetPath) )
    depthFlag=depthfirst
    colorPath=os.path.join(datasetPath,color)
    depthPath=os.path.join(datasetPath,depth)
    createPath(colorPath)
    createPath(depthPath)
    for f in datalist:
        fl=f.lower()
        if fl.endswith("jpg") or fl.endswith("png") or fl.endswith("jpeg"):
            im=Image.open(os.path.join(datasetPath,f))
            im=im.transpose(Image.ROTATE_270) # ← ==> ↑
            if depthFlag:
                im.save(os.path.join(depthPath,f))
            else:
                im.save(os.path.join(colorPath,f))
            depthFlag=not depthFlag
