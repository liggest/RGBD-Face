import numpy as np
# import open3d as o3d
import face_recognition
# import cv2
from skimage import draw #,morphology


# from .depth import depthPreprocess
from .image import getConvexHullMask,getMaskedImg2, maskClosing,maskDilation,maskErosion

def faceLandmarks(image):
    '''
        人脸边框+特征点
    '''
    image=np.asarray(image)
    locations=face_recognition.face_locations(image)
    if not locations:
        return None,None
    landmarks=face_recognition.face_landmarks(image,face_locations=locations)
    return locations,landmarks

def getLandmarkPoints(landmarks):
    '''
        将特征点变成列表形式
    '''
    facePoints=None
    for face in landmarks:
        for v in face.values():
            if facePoints is None:
                facePoints=np.asarray(v)
            else:
                facePoints=np.vstack([facePoints,np.asarray(v)])
    return facePoints

def depthFilter(points,depth,dthreshold=1.0):
    '''
        过滤无深度的点，计算有深度的点在所有点中的比例
    '''
    depth=np.asarray(depth)
    imSize=depth.shape # h,w
    l=len(points)
    pnum=0
    #resultPoints=[None]*l
    resultPoints=[ np.asarray([np.nan,np.nan]) ]*l
    for i,p in enumerate(points):
        if 0<=p[1]<imSize[0] and 0<=p[0]<imSize[1] and 0<depth[p[1],p[0]]<=dthreshold:
            resultPoints[i]=p
            pnum+=1
    #resultPoints=np.asarray(resultPoints,dtype=np.object)
    resultPoints=np.asarray(resultPoints) # dtype:float64
    return resultPoints,pnum/l

def processFaceRGBD(rgbd,depthThreshold=1.0,rateThreshold=0.75):
    '''
        从RGBD图像中得到人脸边框、特征点（语义字典）、特征点（点集）、深度筛选过的特征点、有深度特征点的比例
    '''
    return processFaceImgPair(rgbd.color,rgbd.depth,depthThreshold=depthThreshold,rateThreshold=rateThreshold)

def processFaceImgPair(color,depth,depthThreshold=1000,rateThreshold=0.75):
    '''
        从color、depth图像对中得到人脸边框、特征点（语义字典）、特征点（点集）、深度筛选过的特征点、有深度特征点的比例
    '''
    locations,landmarks=faceLandmarks(color)
    if locations is None:
        print("未检测到人脸")
        return False,None,None,None,None,None
    facePoints=getLandmarkPoints(landmarks)
    facePointsFilt,depthRate=depthFilter(facePoints,depth,dthreshold=depthThreshold)
    if depthRate<rateThreshold:
        print("脸部特征点中有深度的点所占比例小于阈值")
        return False,None,None,None,None,None
    #facePoints=np.asarray(facePoints,dtype=np.object)
    return True,locations,landmarks,facePoints,facePointsFilt,depthRate

def getCorrespondence(points,points2):
    '''
        得到两点集的对应点（下标相同且不为[nan,nan]）
    '''
    minlen=min( points.shape[0],points2.shape[0] )
    points=points[:minlen,:]
    points2=points2[:minlen,:]
    return np.where( ~np.isnan(points).any(axis=1) & ~np.isnan(points2).any(axis=1) )[0]


# def getCorrespondence2(points,points2):
#     '''
#         得到两点集的对应点（下标相同且不为None）
#     '''
#     minlen=min(points.shape[0],points2.shape[0])
#     corrIdxs=[]
#     for i in range(minlen):
#         if not(points[i] is None or points2[i] is None):
#             corrIdxs.append(i)
#     return np.asarray(corrIdxs)


def getFaceCircle(img,facePoints,faceConvexHull,rRate=1.2):
    '''
        通过人脸特征点，得到包裹脸的圆形遮罩
    '''
    idxs=np.asarray( np.where(faceConvexHull>0.5) )
    centroid=np.sum(idxs,axis=1)/idxs.shape[1] #质心
    distances=np.linalg.norm(facePoints-centroid[::-1],axis=1) #[x y]
    maxDistance=distances.max()
    r=maxDistance*rRate
    center=np.int_(centroid) # [y x]
    cr,cc=draw.disk(center,r,shape=img.shape) # [y x]
    mask=np.zeros_like(img,dtype=np.bool)
    mask[cr,cc]=True

    #试着在圆上接个方块
    # p1=np.asarray( [center[0],center[1]-int(r)],dtype=np.int64 )
    # p2=np.asarray( [-1,center[1]+int(r)],dtype=np.int64 )
    # cr,cc=draw.rectangle(start=p1,end=p2,shape=img.shape)
    # mask[cr,cc]=True
    mask[:center[0]+1,center[1]-int(r):center[1]+int(r)+1]=True

    return mask

def getFaceMask(depth,landmarks,facePoints=None,circleRate=1.2):
    '''
        通过人脸特征点，得到包裹脸的圆形遮罩（除去眼、嘴）
    '''
    leftEye= np.asarray( landmarks[0]["left_eye"] )
    rightEye= np.asarray( landmarks[0]["right_eye"] )
    mouth= set(landmarks[0]["top_lip"]).union( set(landmarks[0]["bottom_lip"]) )
    mouth=np.asarray(list(mouth))
    leftEyeMask=getConvexHullMask(depth,leftEye)
    rightEyeMask=getConvexHullMask(depth,rightEye)
    mouthMask=getConvexHullMask(depth,mouth)
    fullMask=leftEyeMask | rightEyeMask | mouthMask
    # disk=morphology.disk(10)
    # disk=morphology.disk(12)
    # fullMaskDila=morphology.binary_dilation(fullMask,selem=disk) 
    fullMaskDila=maskDilation(fullMask,size=12) #膨胀嘴、脸遮罩

    if facePoints is None:
        facePoints=getLandmarkPoints(landmarks)
    
    faceConvexHull=getConvexHullMask(depth,facePoints)
    faceCircleMask=getFaceCircle(np.asarray(depth),facePoints,faceConvexHull,rRate=circleRate)
    faceMask=faceCircleMask ^ fullMaskDila #取相异的部分，即圆里扣去眼、嘴
    return faceCircleMask,faceMask,faceConvexHull

def getFaceMeanMask(depth,faceConvexHull,depthMask,maxMeanDiff=100):
    '''
        以人脸特征点组成的凸包为范围，求人脸的深度均值，得到能够滤除均值+-maxMeanDiff范围外深度的遮罩
    '''
    # disk=morphology.disk(5)
    # disk=morphology.disk(10)
    # faceConvexHulle=morphology.binary_erosion(faceConvexHull,selem=disk) 
    faceConvexHulle=maskErosion(faceConvexHull,size=10) #腐蚀
    idxs=np.where(faceConvexHulle>0.5)
    faceMean= np.mean( depth[idxs[0],idxs[1]] )
    faceMeanMask=np.asarray( (depth>faceMean-maxMeanDiff )&(depth<faceMean+maxMeanDiff ) & depthMask )
    return faceMeanMask


# def processDepth(depth,landmarks,facePoints,cropEyeMouth=True,
#     dmin=100,dmax=1000,filtSize=6,bifiltSize=5,circleRate=1.2,maxMeanDiff=100):
#     '''
#         预处理深度图像，使用圆形遮罩切出头部，视情况施加遮罩切除眼、嘴   
#         返回处理后的完整深度图、头部、剩余部分、遮罩列表（完整深度遮罩、脸部遮罩（切除眼、嘴）、脸部圆形遮罩）        
#     '''
#     depth,depthMask=depthPreprocess(depth,dmin,dmax,filtSize,bifiltSize)
#     if cropEyeMouth:
#         faceCircleMask,faceMask,faceConvexHull=getFaceMask(depth,landmarks,facePoints,circleRate)
#     else:
#         faceConvexHull=getConvexHullMask(depth,facePoints)
#         faceCircleMask=getFaceCircle(depth,facePoints,faceConvexHull,circleRate)
#         faceMask=faceCircleMask

#     faceMeanMask=getFaceMeanMask(depth,faceConvexHull,depthMask,maxMeanDiff=maxMeanDiff) #在脸部均值 +-10cm以外的都会被切掉
    
#     depthMask &=faceMeanMask
#     disk=morphology.disk(2)
#     depthMask=morphology.binary_erosion(depthMask,selem=disk) #腐蚀
#     depthTmp=getMaskedImg2(depth,depthMask)

#     masked=getMaskedImg2(depthTmp,faceMask)
#     maskedReverse=getMaskedImg2(depthTmp,faceCircleMask,reverse=True)
#     depthTmp=depthTmp.astype(np.uint16)
#     masked=masked.astype(np.uint16)
#     maskedReverse=maskedReverse.astype(np.uint16)
#     return depthTmp,masked,maskedReverse,[depthMask,faceMask,faceCircleMask,faceConvexHull]

def processFaceDepth(depth,depthMask,landmarks,facePoints,cropEyeMouth=True,
    circleRate=1.2,maxMeanDiff=100):
    '''
        预处理深度图像，使用圆形遮罩切出头部，视情况施加遮罩切除眼、嘴   
        返回处理后的完整深度图、头部、剩余部分、遮罩列表（完整深度遮罩、脸部遮罩（切除眼、嘴）、脸部圆形遮罩）        
    '''
    #depth,depthMask=depthPreprocess(depth,dmin,dmax,filtSize,bifiltSize)
    if cropEyeMouth:
        faceCircleMask,faceMask,faceConvexHull=getFaceMask(depth,landmarks,facePoints,circleRate)
    else:
        faceConvexHull=getConvexHullMask(depth,facePoints)
        faceCircleMask=getFaceCircle(depth,facePoints,faceConvexHull,circleRate)
        faceMask=faceCircleMask

    faceMeanMask=getFaceMeanMask(depth,faceConvexHull,depthMask,maxMeanDiff=maxMeanDiff) #在脸部均值 +-10cm以外的都会被切掉
    
    depthMask &=faceMeanMask
    # disk=morphology.disk(2)
    # depthMask=morphology.binary_erosion(depthMask,selem=disk) #腐蚀
    # depthMask=maskClosing(depthMask,size=4)
    depthMask=maskErosion(depthMask,size=2) #腐蚀
    depthTmp=getMaskedImg2(depth,depthMask)

    masked=getMaskedImg2(depthTmp,faceMask)
    maskedReverse=getMaskedImg2(depthTmp,faceCircleMask,reverse=True)
    depthTmp=depthTmp.astype(np.uint16)
    masked=masked.astype(np.uint16)
    maskedReverse=maskedReverse.astype(np.uint16)
    return depthTmp,masked,maskedReverse,[depthMask,faceMask,faceCircleMask,faceConvexHull]
