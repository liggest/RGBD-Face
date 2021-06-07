from .utils import *
from .rgbd import *
from .pose import *

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from collections import Iterable

class TSDFContainer:
    '''
        对数据集和重建流程的封装
    '''
    
    default={
        "datasetBase":"./datasets/",
        "dataset":"test",
        "colorset":"rgb",
        "depthset":"depth",
        "intrinsic":"intrinsic.json",
        
        "to16BitDepth":True,
        "depthScale":1000,     #uint16转float时的深度缩放
        "maxDepth":1000,       #深度裁切
        "minDepth":100,
        "minDepthRate":0.75,   #有深度的特征点，最低比例
        "faceCircleRate":1.2,  #圆形人脸遮罩的缩放比例
        "maxMeanDiff":100,     #和脸部深度均值的差距，最终融合时在此范围内的深度会被保留，100就是保留+-10cm
        
        "voxelSize":0.05,      #配准时的threshold
        "maxDepthDiff":0.007,  #0.07?
        
        "tsdfVoxelSize":0.4,
        "sdfTrunc":0.04,       #0.005? 数值太大影响融合速度
        
        "keyframeInterval":5,
        "cropEyeMouth":True,   #（除主帧外）是否裁切眼部、嘴部
        "matcher":FaceMatcher, 
        "registerMethods":[Methods.Ransac,Methods.MultiICP_p2plane],
        "registerMethodsAdjacent":[Methods.RGBDOdometry]
        
    }
    
    def __init__(self,**kw):
        
        for k,v in self.default.items():
            kw.setdefault(k,v)
        
        self.loadConfig(**kw)
        
        self.initTSDF()
        self.initT=np.identity(4)
        
        self.printConfig()
        
    def loadConfig(self,**kw):
        
        self.datasetBase=kw["datasetBase"]
        self.dataset=kw["dataset"]
        self.colorset=kw["colorset"]
        self.depthset=kw["depthset"]
        self.intrinsicFile=kw["intrinsic"]
        self.to16BitDepth=kw["to16BitDepth"]
        self.depthScale=kw["depthScale"]
        self.maxDepth=kw["maxDepth"]
        self.minDepth=kw["minDepth"]
        self.depthTrunc=self.maxDepth/self.depthScale #方便使用
        self.minDepthRate=kw["minDepthRate"]
        self.faceCircleRate=kw["faceCircleRate"]
        self.maxMeanDiff=kw["maxMeanDiff"]

        self.voxelSize=kw["voxelSize"]
        self.tsdfVoxelSize=kw["tsdfVoxelSize"]
        self.sdfTrunc=kw["sdfTrunc"]
        self.keyframeInterval=kw["keyframeInterval"]
        self.maxDepthDiff=kw["maxDepthDiff"]
        self.cropEyeMouth=kw["cropEyeMouth"]
        
        self.datasetPath=os.path.join(self.datasetBase,self.dataset)
        checkPath(self.datasetPath)
        self.colorsetPath=os.path.join(self.datasetPath,self.colorset)
        checkPath(self.colorsetPath)
        self.depthsetPath=os.path.join(self.datasetPath,self.depthset)
        checkPath(self.depthsetPath)
        self.cropsetPath=createPath(os.path.join(self.datasetPath,"crop"))
        checkPath(self.cropsetPath)
        self.intrinsicPath=os.path.join(self.datasetPath,self.intrinsicFile)
        checkPath(self.intrinsicPath)
        self.loadIntrinsic()
        
        matcherTemp=kw["matcher"]
        if isinstance(matcherTemp,PointsMatcher):
            self.matcher=matcherTemp
        else:
            self.matcher=matcherTemp(self,**kw) #这里需要传入matcher类
        
        methodsTemp=kw["registerMethods"]
        if isinstance(methodsTemp,Iterable):
            self.registerMethods=methodsTemp
        else:
            self.registerMethods=[methodsTemp] # 适用于直接传入函数的情况
        
        methodsTemp=kw["registerMethodsAdjacent"]
        if isinstance(methodsTemp,Iterable):
            self.registerMethodsAdjacent=methodsTemp
        else:
            self.registerMethodsAdjacent=[methodsTemp] # 适用于直接传入函数的情况
        
        self.config=kw
        
    def printConfig(self):
        print("="*20)
        for k,v in self.config.items():
            print(f"{k}  =>  {repr(v)}")
            
    def loadIntrinsic(self):
        self.intrinsic=o3d.io.read_pinhole_camera_intrinsic(self.intrinsicPath)
        print("相机内参：")
        print(self.intrinsic.intrinsic_matrix)
    
    def initTSDF(self):
        self.tsdf=o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.tsdfVoxelSize/512,
            sdf_trunc=self.sdfTrunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    
    def getImgPair(self,colorFile,depthFile,depth16bit=True,useO3D=True):
        colorFile=os.path.join(self.colorsetPath,colorFile)
        checkPath(colorFile)
        depthFile=os.path.join(self.depthsetPath,depthFile)
        checkPath(depthFile)
        return getImgPair(colorFile,depthFile,depth16bit,useO3D)
    
    def getRGBD(self,color,depth,scale=1000.0,trunc=1.0,colored=True,depth16bit=True):
        if isinstance(color,str) and isinstance(depth,str):
            color,depth=self.getImgPair(color,depth,depth16bit)
        else:
            color=o3d.geometry.Image( np.asarray(color) )
            depth=o3d.geometry.Image( np.asarray(depth) )
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,depth,depth_scale=scale,depth_trunc=trunc,convert_rgb_to_intensity=not colored
        )
    
    def getFilePath(self,fileName,subpath,inDataset=True):
        path=os.path.join(subpath,fileName)
        if inDataset:
            path=os.path.join(self.datasetPath,path)
        return path
    
    def getImg(self,fileName,subpath="color",inDataset=True,useO3D=True):
        imgPath=self.getFilePath(fileName,subpath,inDataset)
        checkPath(imgPath)
        return getImg(imgPath,useO3D)
        
    def saveImg(self,fileName,img,subpath="crop",inDataset=True):
        imgPath=self.getFilePath(fileName,subpath,inDataset)
        return cv2.imwrite(imgPath,img) #返回是否保存成功
    
    def checkImgExists(self,fileName,subpath="color",inDataset=True):
        # 实际上能检查各种文件是否存在
        return os.path.exists(self.getFilePath(fileName,subpath,inDataset))
        
    def setBaseImgPair(self,colorFile,depthFile):
        rgbdC=RGBDContainer(self,colorFile,depthFile)
        rgbdC.prepareAll(cropEyeMouth=False) # 主帧一定不裁切眼、嘴
        if not rgbdC.faceFeaturesSuccess:
            raise Exception("设置基准图像失败")
        #pltRGBD(rgbdC.getRGBD())
        #plt.show()
        rgbdC.saveCrop()
        self.rgbdC=rgbdC
        self.tsdf.integrate(rgbdC.getRGBDMasked(),self.intrinsic,self.initT)
        
    def addImgPair(self,colorFile,depthFile):
        if not hasattr(self,"rgbdC"):
            self.setBaseImgPair(colorFile,depthFile)
            return True
        rgbdC2=RGBDContainer(self,colorFile,depthFile)
        rgbdC2.prepareAll()
        if not rgbdC2.faceFeaturesSuccess:
            print(f"未添加{rgbdC2.colorFile} , {rgbdC2.depthFile}")
            return False
        #pltRGBD(rgbdC2.getRGBD())
        #plt.show()
        
        registerSuccess,trans,info=self.registerPair3(self.rgbdC,rgbdC2) # adjacent=False
        if not registerSuccess:
            print(f"未添加{rgbdC2.colorFile} , {rgbdC2.depthFile}")
            return False
        try:
            cimg,dimg=pcd2RGBD(self.tsdfPCD,self.intrinsic,extrinsic=trans[:3,:])
        except:
            print("tsdf点云投影至RGBD出错")
            print(f"未添加{rgbdC2.colorFile} , {rgbdC2.depthFile}")
            return False
        #dimg是float
        #printImgInfo(dimg,"dimg")
        maxd=self.depthTrunc
        mind=self.minDepth/self.depthScale
        diff=self.maxMeanDiff/self.depthScale
        #print(mind,maxd,diff)
        depthMask=(dimg<=(maxd-diff)) & (dimg>=max(0,(mind+diff)))
        depthMask=maskDilation(depthMask,size=1)
        depthMask=maskErosion(depthMask,size=2)
        dimg2= getMaskedImg2(rgbdC2.depthMasked,depthMask,reverse=True)
        #printImgInfo(dimg2,"dimg2")
        cimg2= getMaskedImg2(np.asarray(rgbdC2.color),depthMask,reverse=True)
        rgbd2masked=self.getRGBD(cimg2,dimg2,self.depthScale,trunc=self.depthTrunc)
        #pltRGBD(rgbd2masked)
        #plt.show()
        self.tsdf.integrate(rgbd2masked,self.intrinsic,trans)
        return True
        
    def registerPair(self,rgbdSrc,rgbdTar,points2DSrc,points2DTar,adjacent=False):
        T=self.initT
        if not adjacent:
            points3DSrc=get3Dpoints(points2DSrc,rgbdSrc.depth,self.intrinsic)
            points3DTar=get3Dpoints(points2DTar,rgbdTar.depth,self.intrinsic)
            cpcdSrc,cpcdTar,corres=constructPcdPair(points3DSrc,points3DTar)
            ransac=RANSAC(cpcdSrc,cpcdTar,corres)
            if ransac.fitness< 1e-3 or ransac.inlier_rmse> 1e-2:
                print("RANSAC结果较差，略过")
                print(ransac)
                return False,self.initT,np.identity(6)
            T=ransac.transformation
        success,trans,info=RGBDOdometry(
            rgbdSrc,rgbdTar,self.intrinsic,T,maxDepthDiff=self.maxDepthDiff
        )
        return success,trans,info
    
    def registerPair2(self,rgbdCSrc,rgbdCTar,adjacent=False):
        T=self.initT
        rgbdSrc=rgbdCSrc.getRGBDMasked() or rgbdCSrc.getRGBDFilt()
        rgbdTar=rgbdCTar.getRGBDMasked() or rgbdCTar.getRGBDFilt()
        if not adjacent:
            mSuccess,points2DSrc,points2DTar=self.matcher.getMatchPoints(rgbdCSrc,rgbdCTar)
            if not mSuccess:
                print(f"基于 {type(self.matcher)} 的特征点匹配未成功，略过")
                return False,self.initT,np.identity(6)
            rgbdTmpSrc= rgbdCSrc.getRGBDFilt()
            rgbdTmpTar= rgbdCTar.getRGBDFilt()
            points3DSrc=get3Dpoints(points2DSrc,rgbdTmpSrc.depth,self.intrinsic)
            points3DTar=get3Dpoints(points2DTar,rgbdTmpTar.depth,self.intrinsic)
            cpcdSrc,cpcdTar,corres=constructPcdPair(points3DSrc,points3DTar)
            ransac=RANSAC(cpcdSrc,cpcdTar,corres)
            if ransac.fitness< 1e-3 or ransac.inlier_rmse> 1e-2:
                print("RANSAC结果较差，略过")
                print(ransac)
                return False,self.initT,np.identity(6)
            T=ransac.transformation
        success,trans,info=RGBDOdometry(
            rgbdSrc,rgbdTar,self.intrinsic,T,maxDepthDiff=self.maxDepthDiff
        )
        return success,trans,info
    
    def registerPair3(self,rgbdCSrc,rgbdCTar,adjacent=False):
        T=self.initT
        if adjacent:
            methods=self.registerMethodsAdjacent
        else:
            methods=self.registerMethods
        lastMethod=methods[-1]
        methods=methods[:-1]
        for mfunc in methods:
            success,T,_=mfunc(self,rgbdCSrc,rgbdCTar,T,self.voxelSize,needInfo=False)
            if not success:
                print(f"基于 {mfunc.__name__} 的配准未成功，略过")
                return False,self.initT,np.identity(6)
        success,T,info=lastMethod(self,rgbdCSrc,rgbdCTar,T,self.voxelSize,needInfo=True)
        return success,T,info
    
    def getRGBDSet(self,fileSlice=slice(None),lazy=False):
        colorFiles=sorted(os.listdir(self.colorsetPath)) #不要在colorset和depthset里面放图片以外的文件
        depthFiles=sorted(os.listdir(self.depthsetPath))
        rgbdFilePairs=zip(colorFiles[fileSlice],depthFiles[fileSlice])
        return rgbdFilePairs if lazy else list(rgbdFilePairs)
    
    def preprocessRGBDSet(self,fileSlice=slice(None),force=False):
        self.rgbdFiles=self.getRGBDSet(fileSlice)
        cropped=False
        for colorFiles,depthFiles in self.rgbdFiles:
            rgbdC=RGBDContainer(self,colorFiles,depthFiles)
            if rgbdC.cropExists() and not force: # force=False则略过切割，后续程序会尝试读取先前保存过的切割图
                print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 已经处理过")
                yield colorFiles,depthFiles
                continue
            # 反之，force=True则一定切割
            rgbdC.prepareAll(cropEyeMouth=cropped and self.cropEyeMouth)
            if not rgbdC.faceFeaturesSuccess:
                print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 预处理失败")
                continue
            print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 预处理完成")
            rgbdC.saveCrop()
            yield colorFiles,depthFiles
            cropped=True # 只有第一张不切眼、嘴，随后切不切取决于self.cropEyeMouth
    
    def getPoseGraph(self):
        pg=o3d.pipelines.registration.PoseGraph()
        T=self.initT
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))
        start=0
        end=len(self.rgbdFiles)
        for src in range(start,end):
            rgbdCSrc=RGBDContainer(self,*self.rgbdFiles[src])
            rgbdCSrc.prepareImgPair()
            rgbdCSrc.prepareDepth()
            rgbdCSrc.prepareFaceFeatures()
            if not rgbdCSrc.faceFeaturesSuccess:
                print(f"{rgbdCSrc.colorFile} , {rgbdCSrc.depthFile} 预处理失败")
                continue
            #rgbdCSrc.saveCrop()
            rgbdSrc=rgbdCSrc.getRGBDMasked()
            for tar in range(src+1,end):
                if tar-src==1 or (src%self.keyframeInterval==0 and tar%self.keyframeInterval==0):
                    rgbdCTar=RGBDContainer(self,*self.rgbdFiles[tar])
                    rgbdCTar.prepareImgPair()
                    rgbdCTar.prepareDepth()
                    rgbdCTar.prepareFaceFeatures()
                    if not rgbdCTar.faceFeaturesSuccess:
                        print(f"{rgbdCTar.colorFile} , {rgbdCTar.depthFile} 预处理失败")
                        continue
                    rgbdTar=rgbdCTar.getRGBDMasked()
                    
                    corrIdxs=getCorrespondence(rgbdCSrc.facePointsFilt,rgbdCTar.facePointsFilt)
                    facePoints =np.vstack(rgbdCSrc.facePointsFilt[corrIdxs])
                    facePoints2=np.vstack(rgbdCTar.facePointsFilt[corrIdxs])
                    #colorSrc,depthSrc=self.getImgPair(*self.rgbdFiles[src],depth16bit=self.to16BitDepth)
                    #rgbdSrc=self.getRGBD(colorSrc,depthSrc,scale=self.depthScale,trunc=self.depthTrunc)
                    #colorTar,depthTar=self.getImgPair(*self.rgbdFiles[tar],depth16bit=self.to16BitDepth)
                    #rgbdTar=self.getRGBD(colorTar,depthTar,scale=self.depthScale,trunc=self.depthTrunc)
                    print(f"{src} -> {tar}")
                    if tar-src==1: #帧间
                        success,trans,info=self.registerPair(
                            rgbdSrc,rgbdTar,facePoints,facePoints2,adjacent=True
                        )
                        T=trans @ T
                        Tinv=np.linalg.inv(T)
                        #Tinv=reverseT(T)
                        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(Tinv))
                        pg.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=False)
                        )
                    elif src%self.keyframeInterval==0 and tar%self.keyframeInterval==0:
                        success,trans,info=self.registerPair(
                            rgbdSrc,rgbdTar,facePoints,facePoints2,adjacent=False
                        )
                        if success:
                            pg.edges.append(
                                o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=True)
                            )
        return pg

    def getPoseGraph2(self):
        
        def getRGBDC(idx,cropEyeMouth=True):
            nonlocal self
            rgbdC=RGBDContainer(self,*self.rgbdFiles[idx])
            rgbdC.prepareImgPair()
            if rgbdC.cropExists():
                rgbdC.getCrop()
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 已经处理过")
                return rgbdC
            rgbdC.prepareDepth()
            rgbdC.prepareFaceFeatures()
            if rgbdC.faceFeaturesSuccess:
                rgbdC.prepareDepthMasks(cropEyeMouth)
                rgbdC.saveCrop()
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 预处理完成")
            else:
                pass
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 面部相关处理失败")
            return rgbdC
        
        pg=o3d.pipelines.registration.PoseGraph()
        T=self.initT
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))
        start=0
        end=len(self.rgbdFiles)
        lastTar=getRGBDC(start,cropEyeMouth=False) #主帧不切眼、嘴        
        if not lastTar.faceFeaturesSuccess:
            raise Exception("第一组图像就没找到脸")
        self.croppedIdxs=[]
        for src in range(start,end):
            useFeatures=True
            rgbdCSrc=lastTar
            if rgbdCSrc.faceFeaturesSuccess:
                self.croppedIdxs.append(src)
            else:
                useFeatures=False
            tar=src+1
            if tar<end:
                print(f"{src} -> {tar}")
                rgbdCTar=getRGBDC(tar,cropEyeMouth=self.cropEyeMouth) #临近帧的匹配用不到特征点，不关心它是否找到脸并裁切了
                success,trans,info=self.registerPair2(
                    rgbdCSrc,rgbdCTar,adjacent=True
                )
                Ttmp=trans @ T
                Tinv=np.linalg.inv(Ttmp) #可能因为奇异矩阵报错
                T=Ttmp
                pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(Tinv))
                pg.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=False)
                )
                lastTar=rgbdCTar
            if useFeatures and src%self.keyframeInterval==0: #src是关键帧
                ki=self.keyframeInterval
                for tar in range(src+ki,end,ki): #和src之后的关键帧匹配
                    print(f"{src} -> {tar}")
                    rgbdCTar=getRGBDC(tar,cropEyeMouth=self.cropEyeMouth) #脸部处理了才能进一步匹配
                    if rgbdCTar.faceFeaturesSuccess:
                        success,trans,info=self.registerPair2(
                            rgbdCSrc,rgbdCTar,adjacent=False
                        )
                        if success:
                            pg.edges.append(
                                o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=True)
                            )
        return pg
    
    def getPoseGraph3(self):
        
        def getRGBDC(idx,cropEyeMouth=True):
            nonlocal self
            rgbdC=RGBDContainer(self,*self.rgbdFiles[idx])
            rgbdC.prepareImgPair()
            if rgbdC.cropExists():
                rgbdC.getCrop()
                #faceConvexHull=getConvexHullMask(rgbdC.depthFilt,rgbdC.facePoints)
                #faceCircleMask=getFaceCircle(rgbdC.depthFilt,rgbdC.facePoints,faceConvexHull,self.faceCircleRate)
                #rgbdC.depthMasked=getMaskedImg2(rgbdC.depthFilt,faceCircleMask)
                #用不裁切的图匹配试试
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 已经处理过")
                return rgbdC
            rgbdC.prepareDepth()
            rgbdC.prepareFaceFeatures()
            if rgbdC.faceFeaturesSuccess:
                rgbdC.prepareDepthMasks(cropEyeMouth)
                rgbdC.saveCrop()
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 预处理完成")
            else:
                pass
                #print(f"{rgbdC.colorFile} , {rgbdC.depthFile} 面部相关处理失败")
            return rgbdC
        
        pg=o3d.pipelines.registration.PoseGraph()
        T=self.initT
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(T))
        start=0
        end=len(self.rgbdFiles)
        lastTar=getRGBDC(start,cropEyeMouth=False) #主帧不切眼、嘴        
        if not lastTar.faceFeaturesSuccess:
            raise Exception("第一组图像就没找到脸")
        self.croppedIdxs=[]
        for src in range(start,end):
            useFeatures=True
            rgbdCSrc=lastTar
            if rgbdCSrc.faceFeaturesSuccess:
                self.croppedIdxs.append(src)
            else:
                useFeatures=False
            tar=src+1
            if tar<end:
                print(f"{src} -> {tar}")
                rgbdCTar=getRGBDC(tar,cropEyeMouth=self.cropEyeMouth) #临近帧的匹配用不到特征点，不关心它是否找到脸并裁切了
                success,trans,info=self.registerPair3(
                    rgbdCSrc,rgbdCTar,adjacent=True
                )
                Ttmp=trans @ T
                Tinv=np.linalg.inv(Ttmp) #可能因为奇异矩阵报错
                T=Ttmp
                pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(Tinv))
                pg.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=False)
                )
                lastTar=rgbdCTar
            if useFeatures and src%self.keyframeInterval==0: #src是关键帧
                ki=self.keyframeInterval
                for tar in range(src-ki,start-1,-ki): #和src之后的关键帧匹配
                    print(f"{src} -> {tar}")
                    rgbdCTar=getRGBDC(tar,cropEyeMouth=self.cropEyeMouth) #脸部处理了才能进一步匹配
                    if rgbdCTar.faceFeaturesSuccess:
                        success,trans,info=self.registerPair3(
                            rgbdCSrc,rgbdCTar,adjacent=False
                        )
                        if success:
                            pg.edges.append(
                                o3d.pipelines.registration.PoseGraphEdge(src,tar,trans,info,uncertain=True)
                            )
        return pg
        
    def optimizePoseGraph(self,pg):
        method=o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        criteria=o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        #if Methods.RGBDOdometry in self.registerMethods:
        #    plc=0.1 # 优化RGBD里程计量时的推荐值
        #else:
        #    plc=2.0
        option=o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.maxDepthDiff,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.1,  # 优化RGBD里程计量时的推荐值
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(pg,method,criteria,option)
        return pg
    
    def integrateWithPoseGraph(self,pg):
        for i in range(len(pg.nodes)):
            print(f"融合{i}")
            rgbdC=RGBDContainer(self,*self.rgbdFiles[i])
            rgbdC.prepareImgPair()
            rgbd=rgbdC.getRGBDMasked()
            pose=pg.nodes[i].pose
            self.tsdf.integrate(rgbd,self.intrinsic,np.linalg.inv(pose))
        return self.tsdfPCD
    
    @deprecated(msg="已被废弃，请使用 fullProcess3")
    def fullProcess(self,fileSlice=slice(None),forcePreprocess=False):
        #self.rgbdFiles=self.getRGBDSet(fileSlice=fileSlice)
        self.rgbdFiles=list(self.preprocessRGBDSet(fileSlice=fileSlice,force=forcePreprocess) )
        print(f"共{len(self.rgbdFiles)}对RGBD图片")
        pg=self.getPoseGraph()
        pg=self.optimizePoseGraph(pg)
        return self.integrateWithPoseGraph(pg)
    
    def integrateWithPoseGraph2(self,pg):
        for n,idx in enumerate(self.croppedIdxs):
            print(f"融合{n}")
            rgbdC=RGBDContainer(self,*self.rgbdFiles[idx])
            rgbdC.prepareImgPair()
            rgbd=rgbdC.getRGBDMasked()
            pose=pg.nodes[idx].pose
            self.tsdf.integrate(rgbd,self.intrinsic,np.linalg.inv(pose))
        return self.tsdfPCD
    
    @deprecated(msg="已被废弃，请使用 fullProcess3")
    def fullProcess2(self,fileSlice=slice(None)):
        self.rgbdFiles=self.getRGBDSet(fileSlice)
        print(f"共{len(self.rgbdFiles)}对RGBD图片")
        pg=self.getPoseGraph2()
        print(f"成功处理了{len(self.croppedIdxs)}对RGBD图片")
        pg=self.optimizePoseGraph(pg)
        return self.integrateWithPoseGraph2(pg)
    
    def fullProcess3(self,fileSlice=slice(None)):
        self.rgbdFiles=self.getRGBDSet(fileSlice)
        print(f"共{len(self.rgbdFiles)}对RGBD图片")
        pg=self.getPoseGraph3()
        print(f"成功处理了{len(self.croppedIdxs)}对RGBD图片")
        pg=self.optimizePoseGraph(pg)
        return self.integrateWithPoseGraph2(pg)
    
    def integrateWithPoseGraphG(self,pg):
        for n,idx in enumerate(self.croppedIdxs):
            print(f"融合{n}")
            rgbdC=RGBDContainer(self,*self.rgbdFiles[idx])
            rgbdC.prepareImgPair()
            rgbd=rgbdC.getRGBDMasked()
            pose=pg.nodes[idx].pose
            self.tsdf.integrate(rgbd,self.intrinsic,np.linalg.inv(pose))
            yield rgbdC
    
    def fullProcess3ByStep(self,fileSlice=slice(None)):
        self.rgbdFiles=self.getRGBDSet(fileSlice)
        print(f"共{len(self.rgbdFiles)}对RGBD图片")
        pg=self.getPoseGraph3()
        print(f"成功处理了{len(self.croppedIdxs)}对RGBD图片")
        pg=self.optimizePoseGraph(pg)
        return self.integrateWithPoseGraphG(pg)
    
    @property
    def tsdfMesh(self):
        mesh=self.tsdf.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh
    
    @property
    def tsdfPCD(self):
        pcd=o3d.geometry.PointCloud()
        mesh=self.tsdfMesh
        pcd.points=mesh.vertices
        pcd.colors=mesh.vertex_colors
        return pcd
    
    def getSmoothedMesh(self,mesh=None,smoothFunc=smoothMeshTaubin,*args,**kw):
        if mesh:
            return postProcessMesh(mesh,smoothFunc,*args,**kw)
        return postProcessMesh(self.tsdfMesh,smoothFunc,*args,**kw)
        
    def save(self,description="zero",**kw):
        if not hasattr(self,"resultPath"):
            self.resultPath=createPath(os.path.join(self.datasetPath,"result"))
            checkPath(self.resultPath)
        #name=f"{self.dataset} key={self.keyframeInterval} {type(self.matcher).__name__}"
        name=f"{self.dataset} {description}"
        for k,v in kw.items():
            value=self.config.get(v)  #传入 key="keyframeInterval" 或者 keyframeInterval=True等都可
            if value is None:
                value=self.config.get(k)
            if not (value is None): #除了k,v都是None的情况
                name=f"{name} {k}={value}"
        meshPath=self.getFilePath(f"{name} mesh.ply","result")
        pcdPath=self.getFilePath(f"{name}.ply","result")
        configPath=self.getFilePath(f"{name} config.txt","result")
        smoothedMeshPath=self.getFilePath(f"{name} mesh smoothed.ply","result")
        mesh=self.tsdfMesh
        if o3d.io.write_triangle_mesh(meshPath,mesh):
            print(f"保存了网格 {meshPath}")
        else:
            print(f"保存网格失败！")
        if o3d.io.write_point_cloud(pcdPath,self.tsdfPCD):
            print(f"保存了点云 {pcdPath}")
        else:
            print(f"保存点云失败！")
        with open(configPath,"w",encoding="utf-8") as f:
            stdo=sys.stdout
            stde=sys.stderr
            sys.stdout=f
            sys.stderr=f
            self.printConfig()
            sys.stdout=stdo
            sys.stderr=stde
        print(f"保存了配置参数 {configPath}")
        print(f"正在平滑网格……")
        if o3d.io.write_triangle_mesh(smoothedMeshPath,self.getSmoothedMesh(mesh),write_vertex_normals=False):
            print(f"保存了平滑后的网格 {smoothedMeshPath}")
        else:
            print(f"保存平滑后的网格失败！")

