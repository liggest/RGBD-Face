from .utils import *

import numpy as np
import cv2
import open3d as o3d


class PointsMatcher:
    '''
        特征点匹配，基类
    '''
    def __init__(self,tsdfC,**kw):
        self.tsdfC=tsdfC
    
    def getMatchPoints(self,rgbdCSrc,rgbdCTar):
        return False,None,None # 应该输出 (n,2) 形状的点集
    
class ORBMatcher(PointsMatcher):
    '''
        ORB特征点
    '''
    def __init__(self,tsdfC,**kw):
        
        self.intrinsic=tsdfC.intrinsic
        self.orb=cv2.ORB_create(
            scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,
            WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,nfeatures=100,patchSize=31
        )
        super(ORBMatcher,self).__init__(tsdfC,**kw)
        
    def getMatchPoints(self,rgbdCSrc,rgbdCTar):
        maskSrc= (rgbdCSrc.depthMasked>self.tsdfC.minDepth) & (rgbdCSrc.depthMasked<self.tsdfC.maxDepth)
        maskTar= (rgbdCTar.depthMasked>self.tsdfC.minDepth) & (rgbdCTar.depthMasked<self.tsdfC.maxDepth)
        imgSrc= getMaskedImg2(imgAsIntGray(rgbdCSrc.color),maskSrc)
        imgTar= getMaskedImg2(imgAsIntGray(rgbdCTar.color),maskTar)
        kpSrc,descSrc=getORBPoints(imgSrc,self.orb)
        kpTar,descTar=getORBPoints(imgTar,self.orb)
        success,pointsSrc,pointsTar=getORBmatches(kpSrc,descSrc,kpTar,descTar)
        if not success:
            return super(ORBMatcher,self).getMatchPoints(rgbdCSrc,rgbdCTar)
        rePointsSrc,rePointsTar=refineORBmatches(pointsSrc,pointsTar,self.intrinsic)
        if rePointsSrc is None or rePointsTar is None:
            return super(ORBMatcher,self).getMatchPoints(rgbdCSrc,rgbdCTar)
        #pltCorrespondences(imgSrc,imgTar,rePointsSrc,rePointsTar,color=(255,))
        return True,np.int_(rePointsSrc),np.int_(rePointsTar)
        
class FaceMatcher(PointsMatcher):
    '''
        人脸特征点
    '''
    def __init__(self,tsdfC,**kw):
        super(FaceMatcher,self).__init__(tsdfC,**kw)
        
    def getMatchPoints(self,rgbdCSrc,rgbdCTar):
        #print(rgbdCSrc.facePointsFile,rgbdCTar.facePointsFile)
        #print(rgbdCSrc.facePointsFilt.shape,rgbdCTar.facePointsFilt.shape)
        corrIdxs=getCorrespondence(rgbdCSrc.facePointsFilt,rgbdCTar.facePointsFilt)
        pointsSrc=np.int_(rgbdCSrc.facePointsFilt[corrIdxs])
        pointsTar=np.int_(rgbdCTar.facePointsFilt[corrIdxs])
        #pointsSrc =np.vstack(rgbdCSrc.facePointsFilt[corrIdxs])
        #pointsTar =np.vstack(rgbdCTar.facePointsFilt[corrIdxs])
        return True,pointsSrc,pointsTar

class Methods:
    '''
        各种位姿估计方法
    '''
    
    defaultT=np.identity(4)
    defaultInfo=np.identity(6)
    defaultReturn=False,defaultT,defaultInfo

    minFitness=1e-3     # Fitness的阈值，Fitness太小则被认为是对齐失败
    maxRMSE=2e-2 #1e-1  # RMSE的阈值，RMSE太大则被认为是对齐失败
    
    @staticmethod
    def _getRGBDPair(rgbdCSrc,rgbdCTar):
        rgbdSrc=rgbdCSrc.getRGBDMasked() or rgbdCSrc.getRGBDFilt()
        rgbdTar=rgbdCTar.getRGBDMasked() or rgbdCTar.getRGBDFilt()
        return rgbdSrc,rgbdTar
    
    @staticmethod
    def _getPcdPair(tsdfC,rgbdCSrc,rgbdCTar):
        rgbdSrc,rgbdTar=Methods._getRGBDPair(rgbdCSrc,rgbdCTar)
        pcdSrc=o3d.geometry.PointCloud.create_from_rgbd_image(rgbdSrc,tsdfC.intrinsic)
        pcdTar=o3d.geometry.PointCloud.create_from_rgbd_image(rgbdTar,tsdfC.intrinsic)
        return pcdSrc,pcdTar
    
    @staticmethod
    def _getCPcdPair(tsdfC,rgbdCSrc,rgbdCTar):
        mSuccess,points2DSrc,points2DTar=tsdfC.matcher.getMatchPoints(rgbdCSrc,rgbdCTar)
        if not mSuccess:
            print(f"基于 {type(tsdfC.matcher)} 的特征点匹配未成功")
            return False,None,None,None
        rgbdSrc= rgbdCSrc.getRGBDFilt()
        rgbdTar= rgbdCTar.getRGBDFilt()
        points3DSrc=get3Dpoints(points2DSrc,rgbdSrc.depth,tsdfC.intrinsic)
        points3DTar=get3Dpoints(points2DTar,rgbdTar.depth,tsdfC.intrinsic)
        cpcdSrc,cpcdTar,corres=constructPcdPair(points3DSrc,points3DTar)
        return True,cpcdSrc,cpcdTar,corres
    
    @staticmethod
    def _getInfo(needInfo,pcdSrc,pcdTar,threshold,T):
        if needInfo:
            info=o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                pcdSrc, pcdTar, threshold, T)
        else:
            info=Methods.defaultInfo
        return info
    
    @staticmethod
    def _checkResult(result):
        if (result is None) or (result.fitness<Methods.minFitness or result.inlier_rmse>Methods.maxRMSE):
            print("配准结果较差")
            print(result)
            return False
        return True
    
    @staticmethod
    def DoNothing(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        return Methods.defaultReturn
    
    @staticmethod
    def Ransac(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        mSuccess,cpcdSrc,cpcdTar,corres=Methods._getCPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        if not mSuccess:
            return Methods.defaultReturn
        ransac=RANSAC(cpcdSrc,cpcdTar,corres,threshold)
        if not Methods._checkResult(ransac):
            return Methods.defaultReturn
        T=ransac.transformation
        info=Methods._getInfo(needInfo,cpcdSrc,cpcdTar,threshold,T)
        return True,T,info
    
    @staticmethod
    def RGBDOdometry(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=True):
        rgbdSrc,rgbdTar=Methods._getRGBDPair(rgbdCSrc,rgbdCTar)
        success,T,info=RGBDOdometry(
            rgbdSrc,rgbdTar,
            tsdfC.intrinsic,initT,maxDepthDiff=tsdfC.maxDepthDiff
        )
        return success,T,info
    
    @staticmethod
    def ICP_p2p(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp=ICP_p2p(pcdSrc,pcdTar,initT,threshold=threshold)
        if not Methods._checkResult(icp):
            return Methods.defaultReturn
        T=icp.transformation
        info=Methods._getInfo(needInfo,pcdSrc,pcdTar,threshold,T)
        return True,T,info
    
    @staticmethod
    def ICP_p2plane(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp=ICP_p2plane(pcdSrc,pcdTar,initT,threshold=threshold)
        if not Methods._checkResult(icp):
            return Methods.defaultReturn
        T=icp.transformation
        info=Methods._getInfo(needInfo,pcdSrc,pcdTar,threshold,T)
        return True,T,info
    
    @staticmethod
    def ICP_p2plane_loss(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp=ICP_p2plane(pcdSrc,pcdTar,initT,threshold=threshold,withLoss=True)
        if not Methods._checkResult(icp):
            return Methods.defaultReturn
        T=icp.transformation
        info=Methods._getInfo(needInfo,pcdSrc,pcdTar,threshold,T)
        return True,T,info
    
    @staticmethod
    def ICP_color(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp=ICP_color(pcdSrc,pcdTar,initT,threshold=threshold)
        if not Methods._checkResult(icp):
            return Methods.defaultReturn
        T=icp.transformation
        info=Methods._getInfo(needInfo,pcdSrc,pcdTar,threshold,T)
        return True,T,info
    
    @staticmethod
    def MultiICP_p2p(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=True):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp,info=MultiICP(pcdSrc,pcdTar,
                          voxelSizes=[threshold/2,threshold/5,threshold/10],
                          maxIter=[50,30,14],T=initT,icpFunc=ICP_p2p)
        if icp is None:
            T=Methods.defaultT
        else:
            T=icp.transformation
        return True,T,info
    
    @staticmethod
    def MultiICP_p2plane(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=True):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp,info=MultiICP(pcdSrc,pcdTar,
                          voxelSizes=[threshold/2,threshold/5,threshold/10],
                          maxIter=[50,30,14],T=initT,icpFunc=ICP_p2plane)
        if icp is None:
            T=Methods.defaultT
        else:
            T=icp.transformation
        return True,T,info

    @staticmethod
    def MultiICP_p2plane_loss(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=True):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp,info=MultiICP(pcdSrc,pcdTar,
                          voxelSizes=[threshold/2,threshold/5,threshold/10],
                          maxIter=[50,30,14],T=initT,icpFunc=ICP_p2plane,withLoss=True)
        if icp is None:
            T=Methods.defaultT
        else:
            T=icp.transformation
        return True,T,info
    
    @staticmethod
    def MultiICP_color(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=True):
        pcdSrc,pcdTar=Methods._getPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        icp,info=MultiICP(pcdSrc,pcdTar,
                          voxelSizes=[threshold/2,threshold/5,threshold/10],
                          maxIter=[50,30,14],T=initT,icpFunc=ICP_color)
        if icp is None:
            T=Methods.defaultT
        else:
            T=icp.transformation
        return True,T,info
    
    @staticmethod
    def AlignNose(tsdfC,rgbdCSrc,rgbdCTar,initT,threshold,needInfo=False):
        mSuccess,cpcdSrc,cpcdTar,corres=Methods._getCPcdPair(tsdfC,rgbdCSrc,rgbdCTar)
        if not mSuccess:
            return Methods.defaultReturn
        noseIdx=[]
        current=0
        for i,p in enumerate(rgbdCSrc.facePointsFilt):
            if np.isnan(p).any() or np.isnan(rgbdCTar.facePointsFilt[i,:]).any():
                continue
            if 27<=i<31: # 鼻子所在点的下标
                noseIdx.append(current)
            current+=1
        T=icpWithScaleAndAlign(np.asarray(cpcdSrc.points),np.asarray(cpcdTar.points),noseIdx)
        T=np.vstack([T,np.asarray([0,0,0,1])])
        info=Methods._getInfo(needInfo,cpcdSrc,cpcdTar,threshold,T)
        return True,T,info

