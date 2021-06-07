import numpy as np
import open3d as o3d
import cv2

#from .face import processDepth

def get16BitDepth(depth,useO3D=True):
    '''
        处理从iPhone中得到的深度图格式
    '''
    d=np.asarray(depth)
    dr=np.float_(d[:,:,0])
    dg=np.float_(d[:,:,1])
    dz=(dr*255+dg)/10.0 #从三通道转换为单通道深度图
    if useO3D:
        depth16=o3d.geometry.Image(np.uint16(dz))
        return depth16
    else:
        return dz

def depthXY2XYZ(x,y,depth,intrinsic):
    '''
        粗略的图像坐标+深度值转三维坐标
    '''
    imatrix=intrinsic.intrinsic_matrix
    fx,fy=imatrix[0,0],imatrix[1,1]
    cx,cy=imatrix[0,2],imatrix[1,2]
    x3d=depth*(x-cx)/fx
    y3d=depth*(y-cy)/fy
    z3d=depth
    return x3d,y3d,z3d

def get3Dpoints(points2D,depth,intrinsic):
    '''
        通过深度图和相机内参，得到2D点集对应的3D点集（要求2D点为整数）
    '''
    depth=np.asarray(depth)
    imatrix=intrinsic.intrinsic_matrix
    cxy=np.asarray([imatrix[0,2],imatrix[1,2]]).reshape((1,2))
    fxy=np.asarray([imatrix[0,0],imatrix[1,1]]).reshape((1,2))
    d=depth[points2D[:,1],points2D[:,0]].reshape((-1,1))
    points3D= d*(points2D-cxy)/fxy
    points3D=np.hstack([points3D,d])
    return points3D

def get3Dpoints2(points2D,depth,intrinsic):
    '''
        通过深度图和相机内参，得到2D点集对应的3D点集（2D点对应的深度通过双线性插值得到）
    '''
    points3D=np.zeros((points2D.shape[0],3))
    depth=np.asarray(depth)
    for i,p in enumerate(points2D):
        #这里感觉可以在求深度时做插值
        #于是做双线性插值
        pi=np.int_( np.round_(p) )
        xi,yi=pi
        pdiff=p-pi
        xdiff,ydiff=pdiff
        d00=depth[yi  ,xi  ]
        d01=depth[yi  ,xi+1]
        d10=depth[yi+1,xi  ]
        d11=depth[yi+1,xi+1]
        d= (1-ydiff)*( (1-xdiff)*d00+xdiff*d01 ) + ydiff*( (1-xdiff)*d10+xdiff*d11 ) 
        points3D[i,:]=np.asarray(depthXY2XYZ(p[0],p[1],d,intrinsic))
    return points3D

def depthPreprocess(depth,dmin=100,dmax=1000,filtSize=6,bifiltSize=5):
    '''
        对16bit深度图做预处理（平滑滤波、双边滤波），返回滤波后深度图和其遮罩
    '''
    depth=np.asarray(depth)
    depth0=depth.copy()
    mask0= (depth0<dmax) & (depth0>dmin)
    masked=depth0[mask0]
    maxd=masked.max()
    mind=masked.min()
    depth0[depth0<mind]=mind
    depth0[depth0>maxd]=maxd
    depth1= (depth0-mind)/(maxd-mind) # 标准化

    # 平滑滤波
    ffsize=filtSize*filtSize
    H=np.ones((filtSize,filtSize),np.float) / ffsize 
    binary=mask0.astype(np.float)
    binaryfilt=cv2.filter2D(binary,-1,H)
    depthfilt=cv2.filter2D(depth1,-1,H)
    meanfilt= (depthfilt*ffsize)/ ((binaryfilt*ffsize)+ 1e-4) # 这样好像能减少一些边界的影响
    meanfilt[np.isnan(meanfilt)]=0
    #meanfilt=depthfilt * masktest

    # 双边滤波
    depth1_5=cv2.bilateralFilter(
        meanfilt.astype(np.float32),d=bifiltSize,sigmaColor=100,sigmaSpace=100
    )
    depth2=depth1_5*(maxd-mind)+mind

    maskchange=np.abs(depth2-depth*mask0)<5
    maskall=maskchange | mask0
    depth3=depth2 * maskall
    #depth3=np.zeros_like(depth2)
    #depth3[maskall]=depth2[maskall]
    return depth3,maskall

def pcd2RGBD(pcd,intrinsic,extrinsic=None,imSize=(640,480)):
    '''
        点云投影到RGBD
    '''
    if isinstance(intrinsic,o3d.camera.PinholeCameraIntrinsic):
        intrinsic=intrinsic.intrinsic_matrix
    if extrinsic is None:
        extrinsic=np.asarray([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],])
    T= intrinsic @ extrinsic # 3x3 @ 3x4 = 3x4
    points=np.asarray(pcd.points)
    pointlen=points.shape[0]
    ones=np.ones( (pointlen,1) )
    points=np.hstack( [points,ones] ) # nx4
    points2D= T @ points.T # 3x4 @ 4xn = 3xn [xz,yz,z]
    depth=points2D[2,:] # 1xn z
    points2D=points2D/depth # [x,y,1]
    sortedIdx=np.argsort(depth)[::-1] #从大到小对深度排序
    #pxy=np.int_(np.round(points2D[::-1,:])) # 四舍五入取整，顺序是 [1,y,x]
    pxy=np.int_(np.round(points2D[::-1,sortedIdx])) # 将点的顺序变成排序后的（深度大的点靠前）
    depth=depth[sortedIdx] # 将深度的顺序变成排序后的
    #==
    colorImg=np.zeros((imSize[0],imSize[1],3),dtype=np.float64)
    depthImg=np.zeros(imSize)
    idxs=np.where( (0<=pxy[2,:]) & (pxy[2,:]<imSize[1]) & (0<=pxy[1,:]) & (pxy[1,:]<imSize[0]) & (depth>0))[0]
    #选择在图像区域内，且有深度的点
    pxy=pxy[1:,idxs] # [y,x] # 筛选这些点
    colorImg[pxy[0,:],pxy[1,:]]= np.asarray(pcd.colors)[sortedIdx][idxs] #因为点顺序变过了，故颜色也要先变成深度排序后的顺序，再筛选、赋值
    depthImg[pxy[0,:],pxy[1,:]]= depth[idxs] #深度也筛选
    return colorImg,depthImg