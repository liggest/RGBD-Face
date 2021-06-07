import numpy as np
import open3d as o3d
import cv2
import copy

def constructPcdPair(points3D,points3D2):
    '''
        通过两3D点集建立点云，并得到两个点云对应点（下标相同点）的下标列表
    '''
    cpcd=o3d.geometry.PointCloud()
    cpcd2=o3d.geometry.PointCloud()
    corres=o3d.utility.Vector2iVector()
    for i,p in enumerate(points3D):
        cpcd.points.append(points3D[i,:])
        cpcd2.points.append(points3D2[i,:])
        corres.append([i,i])
    return cpcd,cpcd2,corres

def getNormals(pcd,r=0.1,maxNN=30):
    '''
        为点云估计法线，会拷贝一个新的并返回点云
    '''
    if not pcd.has_normals():
        pcd=copy.deepcopy(pcd)
        search=o3d.geometry.KDTreeSearchParamHybrid(radius=r,max_nn=maxNN)
        pcd.estimate_normals(search_param=search)
    return pcd

def reverseT(T):
    '''
        得到变换矩阵T的反变换
    '''
    Tr=np.zeros_like(T)
    Rinv=T[:3,:3].T
    Tr[:3,:3]=Rinv
    #Tr[:3,3]=-T[:3,3]
    Tr[:3,3]= -Rinv @ T[:3,3]
    Tr[3,3]=1
    return Tr

def ICP_p2p(pcd,pcd2,T=np.identity(4),threshold=0.01,maxIter=30):
    '''
        point-to-point ICP
    '''
    icp=o3d.pipelines.registration.registration_icp(
        pcd,pcd2,threshold,T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=maxIter)
    )
    return icp

def ICP_p2plane(pcd,pcd2,T=np.identity(4),threshold=0.01,maxIter=30,withLoss=False):
    '''
        point-to-plane ICP
    '''
    pcd=getNormals(pcd)
    pcd2=getNormals(pcd2)
    if withLoss:
        if hasattr(withLoss, "weight"):
            loss=withLoss
        else:
            loss=o3d.pipelines.registration.TukeyLoss(k=0.1)
        method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    else:
        method=o3d.pipelines.registration.TransformationEstimationPointToPlane()

    icp=o3d.pipelines.registration.registration_icp(
        pcd,pcd2,threshold,T,method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=maxIter)
    )
    return icp

def ICP_color(pcd,pcd2,T=np.identity(4),threshold=0.01,maxIter=30):
    '''
        point-to-plane ICP   
        适用于彩色点云
    '''
    pcd=getNormals(pcd)
    pcd2=getNormals(pcd2)
    method=o3d.pipelines.registration.TransformationEstimationForColoredICP()
    icp=o3d.pipelines.registration.registration_colored_icp(
        pcd,pcd2,threshold,T,method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=maxIter)
    )
    return icp


def MultiICP(pcd,pcd2,voxelSizes=[0.05],maxIter=[30],T=np.identity(4),icpFunc=ICP_p2plane,**kw):
    '''
        多重 ICP，每次先对点云降采样，再根据选定的icp方法配准   
        可选：ICP_p2p、ICP_p2plane、ICP_color
    '''
    #voxelRadius=[0.04,0.02,0.01]
    #maxIter=[50,30,14]
    currentT=T
    lenIter=len(maxIter)
    if not lenIter:
        return None,np.identity(6)
    for scale in range(lenIter):
        iterTimes=maxIter[scale]
        radius=voxelSizes[scale]
        
        pcdDown=pcd.voxel_down_sample(radius)
        pcdDown2=pcd2.voxel_down_sample(radius)
        
        pcdDown.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=30) # 教程里创建了两个对象，不知道能不能只用一个？
        )
        pcdDown2.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2, max_nn=30)
        )
        icp=icpFunc(pcdDown,pcdDown2,currentT,radius*1.4,iterTimes,**kw) # radius*1.4
        currentT=icp.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(pcdDown, pcdDown2, radius * 1.4,currentT)
    return icp,info

def icpNoScale(src,tar):
    '''
        基于对应点的配准（不估计缩放）
        输入3D点src、tar，相同下标的点应为对应点
    '''
    meanSrc = np.mean(src, axis=0)
    meanTar = np.mean(tar, axis=0)
    src1=src-meanSrc.reshape( (1,3) )
    tar1=tar-meanTar.reshape( (1,3) )
    
    W= tar1.T @ src1        
    #W= src1.T @ tar1

    U,sigma,VT = np.linalg.svd(W)
    if np.linalg.det(U) * np.linalg.det(VT) <0:
        U[:,2]*=-1
    
    R= U @ VT
    #t= tar - (R @ src.T).T
    #t= np.mean(t,axis=0).reshape( (3,1) )
    t= meanTar - R @ meanSrc
    T= np.hstack([R,t.reshape( (3,1) )])
    return T

def icpWithScale(src,tar):
    '''
        基于对应点的配准（估计缩放）
        输入3D点src、tar，相同下标的点应为对应点
    '''
    meanSrc = np.mean(src, axis=0)
    meanTar = np.mean(tar, axis=0)
    src1=src-meanSrc.reshape( (1,3) )
    tar1=tar-meanTar.reshape( (1,3) )
    
    W= tar1.T @ src1        

    U,sigma,VT = np.linalg.svd(W)
    if np.linalg.det(U) * np.linalg.det(VT) <0:
        U[:,2]*=-1
    
    R= U @ VT
    s = np.sum(np.abs(tar1)) / np.sum(np.abs(R @ src1.T ))
    R*=s
    t= meanTar - R @ meanSrc
    T= np.hstack([R,t.reshape( (3,1) )])
    return T

def rigidRANSAC(src,tar,n=10,iterTimes=50,threshold=0.05,icpFunc=icpNoScale):
    '''
        基于对应点的RANSAC配准
        src、tar => 3D点，相同下标的点应为对应点
        n => 配准成功所需的最小合群点数（以及单次采样数）
        iterTimes => 迭代次数
        threshold => 距离阈值，用于判定合群点
        icpFunc => icpNoScale、icpWithScale
    '''
    plen = src.shape[0]
    T=np.identity(4)
    inliers = []
    maxInliers = 0
    if plen<n:
        return False,T,inliers
    
    def computeInliers(T,src,tar):
        p1=np.hstack([src,np.ones( (plen,1) )])
        p2=tar
        Tp1=T @ p1.T
        distance=np.linalg.norm(Tp1.T - p2,axis=1)
        idx=np.where(distance<threshold)[0]
        return idx
    
    for _ in range(iterTimes):
        samples = np.random.randint(0, plen, size=n)
        currentT = icpFunc(src[samples,:], tar[samples,:])
        currentInliers = computeInliers(currentT,src,tar)
        if not currentInliers.shape[0]:
            continue
        lenInliers = len(currentInliers)
        if lenInliers>maxInliers:
            maxInliers=lenInliers
            inliers=currentInliers
    if maxInliers <= n:
        return False,T,inliers
    T=icpFunc(src[inliers,:], tar[inliers,:])
    T=np.vstack([T,np.asarray([0,0,0,1])])
    return True,T,inliers

def icpWithScaleAndAlign(src,tar,alignIdx=None,tmask=[True,True,False]):
    '''
        基于对应点的配准（估计缩放）
        输入3D点src、tar，相同下标的点应为对应点
        配准结果按照两点集里alignIdx下标的点对齐位移t（比如提供鼻子下标来对齐鼻子）
        可以提供tmask来指定x、y、z方向的对齐系数
    '''

    meanSrc = np.mean(src, axis=0)
    meanTar = np.mean(tar, axis=0)
    src1=src-meanSrc.reshape( (1,3) )
    tar1=tar-meanTar.reshape( (1,3) )
    
    W= tar1.T @ src1

    U,sigma,VT = np.linalg.svd(W)
    if np.linalg.det(U) * np.linalg.det(VT) <0:
        U[:,2]*=-1
    
    R= U @ VT
    s = np.sum(np.abs(tar1)) / np.sum(np.abs(R @ src1.T ))
    R*=s
    t= meanTar - R @ meanSrc
    if not alignIdx is None:
        tmask=np.asarray(tmask)
        alignIdx=np.asarray(alignIdx)
        tAlign= np.mean(tar[alignIdx], axis=0)- R @ np.mean(src[alignIdx], axis=0)
        t= tAlign*tmask + t*(1-tmask)

    T= np.hstack([R,t.reshape( (3,1) )])
    return T

def RANSAC(pcd,pcd2,correspondence,threshold=0.01):
    '''
        基于对应点的RANSAC
    '''
    try:
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000,1000)
    except:
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) #12.0版本是这个
    ransac=o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd,pcd2,correspondence,threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        ransac_n=6,
        criteria=criteria
    )
    return ransac

def RGBDOdometry(rgbdSrc,rgbdTar,intrinsic,T=np.identity(4),maxDepthDiff=0.07):
    '''
        RGBD里程计
    '''
    option=o3d.pipelines.odometry.OdometryOption()
    term=o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    option.max_depth_diff=maxDepthDiff
    success,trans,info=o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbdSrc,rgbdTar,intrinsic,T,term,option
    )
    return success,trans,info
    
def getORBPoints(color,orb=None):
    '''
        得到图像的ORB关键点和描述子
    '''

    if not orb:
        orb=cv2.ORB_create(
            scaleFactor=1.2,nlevels=8,edgeThreshold=31,firstLevel=0,
            WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE,nfeatures=100,patchSize=31
        )

    kp,desc=orb.detectAndCompute(color,None)
    return kp,desc

def getORBmatches(kp1,desc1,kp2,desc2):
    '''
        匹配两ORB结果，判断是否匹配，得到匹配的对应点
    '''
    if len(kp1)==0 or len(kp2)==0:
        return False,None,None
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=bf.match(desc1,desc2)

    if len(matches)==0:
        return False,None,None

    points1=[]
    points2=[]
    for m in matches:
        points1.append(kp1[m.trainIdx].pt)
        points2.append(kp2[m.queryIdx].pt)
    return True,np.asarray(points1),np.asarray(points2) # nx2

def refineORBmatches(points1,points2,intrinsic):
    '''
        使用5点RANSAC优化ORB匹配对应点
    '''
    intrinsicT=intrinsic.intrinsic_matrix
    fx=intrinsicT[0,0]
    fy=intrinsicT[1,1]
    f=(fx+fy)/2.0
    cx=intrinsicT[0,2]
    cy=intrinsicT[1,2]

    points1Int=np.int32(points1+0.5) #四舍五入
    points2Int=np.int32(points2+0.5) #四舍五入
    E,mask=cv2.findEssentialMat(
        points1Int,points2Int,focal=f,pp=(cx,cy),method=cv2.RANSAC,prob=0.999,threshold=1.0
    )
    if mask is None:
        return None,None
    mask=mask.flatten()
    return points1[mask],points2[mask] # nx2
