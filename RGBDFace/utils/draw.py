import copy
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import open3d as o3d

def drawPoints(image,points,radius=2,color=(255,255,255)):
    '''
        在图像上绘制特征点
    '''
    if isinstance(image,Image.Image):
        im=image
    else:
        im=Image.fromarray(np.asarray(image))
    draw=ImageDraw.Draw(im)
    for point in points:
        if not np.isnan(point).any():
            draw.ellipse((point[0]-radius,point[1]-radius,point[0]+radius,point[1]+radius),fill=color)
    return im

def pltCorrespondences(imgSrc,imgTar,pointsSrc,pointsTar,radius=2,color=(255,255,255)):
    '''
        绘制两张图像、它们匹配的特征点、匹配点间的连线
    '''
    imgSrc=np.asarray(drawPoints(imgSrc,pointsSrc,radius=radius,color=color))
    imgTar=np.asarray(drawPoints(imgTar,pointsTar,radius=radius,color=color))
    srcSize=imgSrc.shape
    tarSize=imgTar.shape
    hSrc,wSrc=srcSize[:2]
    hTar,wTar=tarSize[:2]
    totalw=wSrc+wTar
    h=max(hSrc,hTar)
    if len(srcSize)>2:
        cSrc=srcSize[2]
        cTar=tarSize[2]
        c=max(cSrc,cTar)
        im=np.zeros( (h,totalw,c) ,dtype=imgSrc.dtype)
        im[:hSrc,:wSrc,:]=imgSrc
        im[:hTar,wSrc:totalw,:]=imgTar
    else:
        im=np.zeros( (h,totalw) )
        im[:hSrc,:wSrc]=imgSrc
        im[:hTar,wSrc:totalw]=imgTar
    
    for i in range(pointsSrc.shape[0]):
        if not ( np.isnan(pointsSrc[i,:]).any() or np.isnan(pointsTar[i,:]).any() ):
            xSrc=pointsSrc[i,0]
            ySrc=pointsSrc[i,1]
            xTar=pointsTar[i,0] + wSrc
            yTar=pointsTar[i,1]
            plt.plot( [xSrc,xTar],[ySrc,yTar],color=np.random.random(3)/2+0.5,lw=1.0 )
    plt.imshow(im)
    #plt.show()


def printImgInfo(img,name,**kw):
    '''
        绘制图像并打印信息（图像名称、尺寸、最值、数据类型）
    '''
    print(name,img.shape,img.min(),"-",img.max(),img.dtype)
    plt.imshow(img,**kw)
    plt.show()

def pltRGBD(rgbd):
    '''
        用plt绘制rgbd
    '''
    if isinstance(rgbd,list) or isinstance(rgbd,tuple):
        color,depth=rgbd[0],rgbd[1]
    else:
        color,depth=rgbd.color,rgbd.depth
    plt.subplot(1,2,1)
    plt.imshow(color)
    plt.subplot(1,2,2)
    plt.imshow(depth,vmin=0,vmax=1,cmap='gray')

def drawPCDPair(p1,p2,T):
    '''
        用Open3D绘制两个点云（单色）
    '''
    pc1=copy.deepcopy(p1)
    pc2=copy.deepcopy(p2)
    pc1.paint_uniform_color([0.84,0.24,0])
    pc2.paint_uniform_color([0.64,0.64,0])
    pc1.transform(T)
    o3d.visualization.draw_geometries(
        [pc1,pc2]
    )

def drawColoredPCDPair(p1,p2,T):
    '''
        用Open3D绘制两个点云（彩色）
    '''
    pc1=copy.deepcopy(p1)
    pc1.transform(T)
    o3d.visualization.draw_geometries(
        [pc1,p2]
    )

def drawCorrPcdPair(cpcd,cpcd2,corres,T):
    '''
       绘制两点云、它们匹配点间的连线 
    '''
    corresLines=o3d.utility.Vector2iVector()
    cpcdtmp=copy.deepcopy(cpcd)
    cpcdtmp2=copy.deepcopy(cpcd2)
    cpcdtmp.paint_uniform_color([0.84,0.24,0])
    cpcdtmp2.paint_uniform_color([0.64,0.64,0])
    cpcdtmp.transform(T)
    linePoints=o3d.utility.Vector3dVector()
    count=0
    for l in corres:
        linePoints.append(cpcdtmp.points[l[0]])
        linePoints.append(cpcdtmp2.points[l[1]])
        corresLines.append([count,count+1])
        count+=2
    pairLines=o3d.geometry.LineSet(linePoints,corresLines)
    #print(pairLines)
    o3d.visualization.draw_geometries(
        [cpcdtmp,cpcdtmp2,pairLines]
    )

def evaluateAndDraw(pcd,pcd2,T,threshold=0.01,colored=True):
    '''
       评估两点云的配准，并按需绘制它们   
       colored: None 不绘制  True 彩色  False 单色
    '''
    ev=o3d.pipelines.registration.evaluate_registration(
        pcd,pcd2,threshold,T
    )
    if not colored is None:
        if colored:
            drawColoredPCDPair(pcd,pcd2,T)
        else:
            drawPCDPair(pcd,pcd2,T)
        print(ev)
    return ev
