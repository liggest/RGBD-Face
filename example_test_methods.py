from dataclasses import dataclass,field
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import time

from RGBDFace import TSDFContainer,RGBDContainer,FaceMatcher
from RGBDFace.utils import get3Dpoints,constructPcdPair,evaluateAndDraw
from RGBDFace.utils import ICP_p2plane,MultiICP,RANSAC,RGBDOdometry

'''
    测试位姿估计方法时所用的程序
    其实这些代码在jupyter notebook里面执行也不错……
'''

@dataclass
class RegistrationResult:
    ''' 记录配准结果 '''
    name: str =field(compare=False)
    totalTime: float =field(init=False,default=0)
    avgTime: float =field(init=False,default=0)
    avgFitness: float =field(init=False,default=0)
    avgInlierRMSE: float =field(init=False,default=0)
    avgInlierCount: int =field(init=False,default=0)
    avgFitness_P: float =field(init=False,default=0)
    avgInlierRMSE_P: float =field(init=False,default=0)
    avgInlierCount_P: int =field(init=False,default=0)
        
    def add(self,ev,ev_P):
        self.avgFitness+=ev.fitness
        self.avgInlierRMSE+=ev.inlier_rmse
        self.avgInlierCount+=np.asarray(ev.correspondence_set).shape[0]
        self.avgFitness_P+=ev_P.fitness
        self.avgInlierRMSE_P+=ev_P.inlier_rmse
        self.avgInlierCount_P+=np.asarray(ev_P.correspondence_set).shape[0]
    
    def mean(self,n):
        self.avgTime=self.totalTime/n
        self.avgFitness/=n
        self.avgInlierRMSE/=n
        self.avgInlierCount/=n
        self.avgFitness_P/=n
        self.avgInlierRMSE_P/=n
        self.avgInlierCount_P/=n

class RegistrationTester:
    ''' 测试用 '''
    #colored=None
    #threshold=0.05
    results={}
    
    def __init__(self,threshold=0.05,colored=None):
        self.threshold=threshold
        self.colored=colored
    
    def test(self,name="test1",n=20):
        def warpper(func):
            result=RegistrationResult(name)
            start=time.time()
            for i in range(n):
                ev,ev_P=func(self.threshold,self.colored)
                result.add(ev,ev_P)
            end=time.time()
            result.totalTime=end-start
            result.mean(n)
            RegistrationTester.results[name]=result
            print(result)
            return func
        return warpper

# ==============
#  准备测试用数据
# ==============

# 数据集对象
container= TSDFContainer(dataset="xxx")
rgbdCSrc =RGBDContainer(container,"IMG_xxxx.jpeg","IMG_xxxx.png") # 第一组RGBD图（匹配源）
rgbdCTar=RGBDContainer(container,"IMG_xxxx.jpeg","IMG_xxxx.png") # 第二组RGBD图（匹配目标）
rgbdCSrc.prepareAll(cropEyeMouth=True)  # 预处理 cropEyeMouth 决定是否裁切眼、嘴
rgbdCTar.prepareAll(cropEyeMouth=True)

# 人脸特征点
fm=FaceMatcher(container,**container.config) 
mSuccess,points2DSrc,points2DTar=fm.getMatchPoints(rgbdCSrc,rgbdCTar)
if not mSuccess:
    raise Exception(f"基于 {type(fm)} 的特征点匹配未成功")

# 人脸特征点云
rgbdTmpSrc= rgbdCSrc.getRGBDFilt()
rgbdTmpTar= rgbdCTar.getRGBDFilt()
points3DSrc=get3Dpoints(points2DSrc,rgbdTmpSrc.depth,container.intrinsic)
points3DTar=get3Dpoints(points2DTar,rgbdTmpTar.depth,container.intrinsic)
cpcdSrc,cpcdTar,corres=constructPcdPair(points3DSrc,points3DTar)

# 稠密点云
pcdSrc=o3d.geometry.PointCloud.create_from_rgbd_image(rgbdCSrc.getRGBDMasked(),container.intrinsic)
pcdTar=o3d.geometry.PointCloud.create_from_rgbd_image(rgbdCTar.getRGBDMasked(),container.intrinsic)

# 预处理后RGBD图像
rgbdSrc=rgbdCSrc.getRGBDMasked()
rgbdTar=rgbdCTar.getRGBDMasked()

# =============
#  测试过程示例
# =============

# 实例化 Rtest
colored=None
threshold=container.voxelSize # 0.05
Rtest=RegistrationTester(threshold=container.voxelSize,colored=colored)

# 未配准
@Rtest.test("0")
def testZero(threshold,colored):
    ev  =evaluateAndDraw(cpcdSrc,cpcdTar,np.identity(4),colored=colored)
    ev_P=evaluateAndDraw(pcdSrc ,pcdTar ,np.identity(4),colored=colored)
    return ev,ev_P

# 多重point-to-plane ICP
@Rtest.test("MultiICPp2plane_P")
def test(threshold,colored):
    icp,info=MultiICP(pcdSrc,pcdTar,
                      voxelSizes=[0.03,0.01,0.005],maxIter=[50,30,14],T=np.identity(4),icpFunc=ICP_p2plane)
    if icp is None:
        T=np.identity(4)
    else:
        T=icp.transformation
    ev  =evaluateAndDraw(cpcdSrc,cpcdTar,T,colored=colored)
    ev_P=evaluateAndDraw(pcdSrc ,pcdTar ,T,colored=colored)
    return ev,ev_P

# RANSAC
@Rtest.test("RANSAC")
def test(threshold,colored):
    ransac=RANSAC(cpcdSrc,cpcdTar,corres,threshold)
    ev  =evaluateAndDraw(cpcdSrc,cpcdTar,ransac.transformation,colored=colored)
    ev_P=evaluateAndDraw(pcdSrc ,pcdTar ,ransac.transformation,colored=colored)
    return ev,ev_P

# RGBD Odometry
@Rtest.test("Odometry")
def test(threshold,colored):
    success,odoTrans,info=RGBDOdometry(
        rgbdSrc,rgbdTar,
        container.intrinsic,T=np.identity(4),maxDepthDiff=container.maxDepthDiff
    )
    ev  =evaluateAndDraw(cpcdSrc,cpcdTar,odoTrans,colored=colored)
    ev_P=evaluateAndDraw(pcdSrc ,pcdTar ,odoTrans,colored=colored)
    return ev,ev_P

# RANSAC 后接 RGBD Odometry
@Rtest.test("RANSAC_Odometry")
def test(threshold,colored):
    ransac=RANSAC(cpcdSrc,cpcdTar,corres,threshold)
    success,odoTrans,info=RGBDOdometry(
        rgbdSrc,rgbdTar,
        container.intrinsic,T=ransac.transformation,maxDepthDiff=container.maxDepthDiff
    )
    ev  =evaluateAndDraw(cpcdSrc,cpcdTar,odoTrans,colored=colored)
    ev_P=evaluateAndDraw(pcdSrc ,pcdTar ,odoTrans,colored=colored)
    return ev,ev_P

print(Rtest.results.keys()) # 列出所有测试名称

def pltTest(names=[],normalize=True):
    ''' 
        绘制测试结果
            names 为要绘制的测试名称
            
            normalize 决定绘图时是否对数据做归一化
    '''
    ln=len(names)
    data=np.zeros( (ln,7) )
    x=np.arange(ln)
    halfWidth=0.15
    width=0.3
    for i,n in enumerate(names):
        r=RegistrationTester.results.get(n)
        if r:
            data[i,0]=r.avgTime
            data[i,1]=r.avgFitness
            data[i,2]=r.avgInlierRMSE
            data[i,3]=r.avgInlierCount
            data[i,4]=r.avgFitness_P
            data[i,5]=r.avgInlierRMSE_P
            data[i,6]=r.avgInlierCount_P
    if normalize:
        for i in range(1,7):
            dmax=np.max(data[:,i])
            dmin=np.min(data[:,i])
            data[:,i]= (data[:,i]-dmin)/(dmax-dmin)
            data[:,i]= data[:,i]*0.9+0.1
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.bar(x-halfWidth,data[:,1],width=width,label="Fitness")
    plt.bar(x+halfWidth,data[:,2],width=width,label="InlierRMSE")
    plt.xticks(x,names,rotation=-72)
    #plt.plot(names,data[:,3],label="InlierCount")
    plt.legend()
    plt.subplot(1,2,2)
    plt.bar(x-halfWidth,data[:,4],width=width,label="Fitness_P")
    plt.bar(x+halfWidth,data[:,5],width=width,label="InlierRMSE_P")
    #plt.plot(names,data[:,6],label="InlierCount_P")
    plt.xticks(x,names,rotation=-72)
    plt.legend()
    plt.show()
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    plt.bar(x,data[:,0],width=width,label="Time")
    plt.xticks(x,names,rotation=-72)
    plt.legend()
    plt.show()

# =============
#  绘制结果示例
# =============

pltTest([
    '0','MultiICPp2plane_P','RANSAC','Odometry','RANSAC_Odometry'
])