from .utils import *

import numpy as np


class RGBDContainer:
    '''
        对单帧RGBD图像的封装
    '''
    
    def __init__(self,tsdfContainer,colorFile,depthFile):
        self.tsdfC=tsdfContainer
        self.colorFile=colorFile
        self.depthFile=depthFile
        self.cropFile=expandFileName(depthFile,suffix="_crop")
        self.facePointsFile=expandFileName(depthFile,suffix="_face",ext=".pts")
    
    def prepareImgPair(self):
        color,depth=self.tsdfC.getImgPair(self.colorFile,self.depthFile,depth16bit=self.tsdfC.to16BitDepth)
        self.color=color
        self.depth=depth
    
    def prepareDepth(self):
        depth,depthMask=depthPreprocess(
            self.depth,
            dmin=self.tsdfC.minDepth,dmax=self.tsdfC.maxDepth
        )
        self.depthFilt=depth
        self.depthFullMask=depthMask
    
    def prepareFaceFeatures(self):
        depth=self.depth
        if hasattr(self,"depthFilt"):
            depth=self.depthFilt
        success,faceLocations,faceLandmarks,facePoints,facePointsFilt,depthRate=processFaceImgPair(
            self.color, depth,
            depthThreshold=self.tsdfC.maxDepth,rateThreshold=self.tsdfC.minDepthRate
        )
        self.faceFeaturesSuccess=success
        if self.faceFeaturesSuccess:
            self.faceLocations=faceLocations
            self.faceLandmarks=faceLandmarks
            self.facePoints=facePoints
            self.facePointsFilt=facePointsFilt
            self.depthRate=depthRate
        return self.faceFeaturesSuccess
    
    def prepareDepthMasks(self,cropEyeMouth=True):
        depth,masked,maskedReverse,masksList=processFaceDepth(
            np.asarray(self.depthFilt),self.depthFullMask,
            self.faceLandmarks,self.facePoints,
            cropEyeMouth=cropEyeMouth,
            circleRate=self.tsdfC.faceCircleRate,maxMeanDiff=self.tsdfC.maxMeanDiff
        )
        self.cropEyeMouth=cropEyeMouth
        self.depthFilt=depth
        self.depthMasked=masked
        self.depthMaskedReverse=maskedReverse
        self.depthFullMask=masksList[0]
        self.depthFaceMask=masksList[1]
        self.depthCircleMask=masksList[2]
        self.depthFaceConvexHull=masksList[3]
        
        self.facePointsFilt,_=depthFilter(self.facePoints,self.depthFilt,dthreshold=self.depthFilt.max()+1)
    
    def prepareAll(self,cropEyeMouth=True):
        self.prepareImgPair()
        self.prepareDepth()
        if self.prepareFaceFeatures():
            self.prepareDepthMasks(cropEyeMouth)
        else:
            print(f"{self.colorFile},{self.depthFile} - 识别脸部特征失败")
        return self.faceFeaturesSuccess
            
    def getCrop(self):
        self.depthMasked=self.tsdfC.getImg(self.cropFile,subpath="crop",useO3D=False)
        self.depthFilt=self.tsdfC.getImg(self.depthFile,subpath="crop",useO3D=False)
        self.facePoints=np.loadtxt(
            self.tsdfC.getFilePath(self.facePointsFile,"crop"),dtype=np.int32,delimiter=",",encoding="utf-8"
        )
        self.facePointsFilt,self.depthRate=depthFilter(self.facePoints,self.depthFilt,dthreshold=self.depthFilt.max()+1)
        self.faceFeaturesSuccess=True
        return self.depthMasked
    
    def saveCrop(self):
        result=self.tsdfC.saveImg(self.cropFile,np.uint16(self.depthMasked),"crop") #遮罩图
        result&=self.tsdfC.saveImg(self.depthFile,np.uint16(self.depthFilt),"crop") #过滤后的深度图
        np.savetxt(
            self.tsdfC.getFilePath(self.facePointsFile,"crop"),self.facePoints,     #深度过滤前的特征点
            fmt="%d",header=f"{self.colorFile} 的人脸特征点",
            delimiter=",",encoding="utf-8"
        )
        return result

    def cropExists(self):
        result=self.tsdfC.checkImgExists(self.cropFile,subpath="crop")
        result&=self.tsdfC.checkImgExists(self.depthFile,subpath="crop")
        result&=self.tsdfC.checkImgExists(self.facePointsFile,subpath="crop")
        return result
    
    def _getRGBD(self,name,depth):
        if depth is None:
            return None
        rgbd=self.tsdfC.getRGBD(self.color,depth,scale=self.tsdfC.depthScale,trunc=self.tsdfC.depthTrunc)
        setattr(self,name,rgbd)
        return rgbd
    
    def getRGBD(self):
        return self._getRGBD("rgbd",getattr(self,"depth",None) )
        
    def getRGBDFilt(self):
        return self._getRGBD("rgbdFilt",getattr(self,"depthFilt",None) )
    
    def getRGBDMasked(self):
        if not hasattr(self,"depthMasked") and self.cropExists():
            self.getCrop()
        return self._getRGBD("rgbdMasked",getattr(self,"depthMasked",None) )  
    
    def getRGBDMaskedReverse(self):
        return self._getRGBD("rgbdMaskedReverse",getattr(self,"depthMaskedReverse",None) )